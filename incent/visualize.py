import numpy as np
from anndata import AnnData
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


def generalized_procrustes_analysis(
    X,
    Y,
    pi,
    output_params=False,
    matrix=False,
    allow_reflection=True,
    topk=None,
    eps=1e-12,
):
    """
    Robust post-OT rigid alignment.

    Compared with the old version, this does:
    1) coarse weighted rigid fit from the full soft pi
    2) hard partial one-to-one projection of pi using Hungarian + reject option
    3) final rigid fit on the hard support only

    Parameters
    ----------
    X : (n, 2) array
        Source coordinates.
    Y : (m, 2) array
        Target coordinates.
    pi : (n, m) array
        Soft OT coupling.
    output_params : bool
        If True, also returns transform parameters.
    matrix : bool
        If True, returns rotation/reflection matrix R.
        Else returns theta = atan2(R[1,0], R[0,0]).
        For reflected solutions, theta alone is not fully descriptive,
        so matrix=True is recommended.
    allow_reflection : bool
        Allow flips/reflections in the rigid fit.
    topk : int or None
        Number of target candidates kept per source row.
        If None, chosen automatically.
    eps : float
        Small numerical constant.

    Returns
    -------
    If output_params=False:
        X_aligned, Y_aligned

    If output_params=True and matrix=False:
        X_aligned, Y_aligned, theta, src_center, t

    If output_params=True and matrix=True:
        X_aligned, Y_aligned, R, src_center, t

    Notes
    -----
    The rigid transform is:
        x_aligned = R @ x + t
    implemented in row-vector form as:
        X_aligned = X @ R.T + t
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    pi = np.asarray(pi, dtype=float)

    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == 2 and Y.shape[1] == 2
    assert pi.shape == (X.shape[0], Y.shape[0])

    def _auto_topk(m):
        return max(3, min(10, int(np.ceil(np.log2(max(m, 4))))))

    def _target_scale(Y_):
        if Y_.shape[0] < 2:
            return 1.0
        nn = NearestNeighbors(n_neighbors=2).fit(Y_)
        dists, _ = nn.kneighbors(Y_)
        s = float(np.median(dists[:, 1]))
        return max(s, 1.0)

    def _fit_soft_rigid(X_, Y_, W_, allow_reflection_=True):
        total = float(W_.sum())
        if total <= eps:
            raise ValueError("The transport plan mass is zero! Cannot align.")

        Wn = W_ / total
        a = Wn.sum(axis=1)
        b = Wn.sum(axis=0)

        mx = (a[:, None] * X_).sum(axis=0)
        my = (b[:, None] * Y_).sum(axis=0)

        Xc = X_ - mx
        Yc = Y_ - my

        H = Xc.T @ Wn @ Yc
        U, _, Vt = np.linalg.svd(H)
        R_ = Vt.T @ U.T

        if (not allow_reflection_) and (np.linalg.det(R_) < 0):
            Vt[-1, :] *= -1
            R_ = Vt.T @ U.T

        t_ = my - R_ @ mx
        return R_, t_, mx, my

    def _fit_pairs_rigid(Xm, Ym, w=None, allow_reflection_=True):
        if Xm.shape[0] == 0:
            raise ValueError("No matched pairs found for rigid fit.")

        if w is None:
            w = np.ones(Xm.shape[0], dtype=float)
        else:
            w = np.asarray(w, dtype=float)

        wsum = float(w.sum())
        if wsum <= eps:
            raise ValueError("Matched-pair weights sum to zero.")

        wn = w / wsum
        mx = (wn[:, None] * Xm).sum(axis=0)
        my = (wn[:, None] * Ym).sum(axis=0)

        Xc = Xm - mx
        Yc = Ym - my

        H = Xc.T @ (wn[:, None] * Yc)
        U, _, Vt = np.linalg.svd(H)
        R_ = Vt.T @ U.T

        if (not allow_reflection_) and (np.linalg.det(R_) < 0):
            Vt[-1, :] *= -1
            R_ = Vt.T @ U.T

        t_ = my - R_ @ mx
        return R_, t_, mx, my

    # ------------------------------------------------------------------
    # 1) Coarse soft rigid fit from full pi
    # ------------------------------------------------------------------
    pi_mass = float(pi.sum())
    if pi_mass <= eps:
        raise ValueError("The transport plan mass is zero! Cannot align.")

    R0, t0, src_center0, _ = _fit_soft_rigid(X, Y, pi, allow_reflection_=allow_reflection)

    # ------------------------------------------------------------------
    # 2) Hard partial one-to-one projection of pi
    # ------------------------------------------------------------------
    n, m = pi.shape
    row_mass = pi.sum(axis=1)
    active_rows = np.where(row_mass > eps)[0]

    if len(active_rows) < 2:
        # Too little support to do anything more robust than the soft fit
        X_aligned = X @ R0.T + t0
        Y_aligned = Y.copy()

        if not output_params:
            return X_aligned, Y_aligned

        theta = float(np.arctan2(R0[1, 0], R0[0, 0]))
        if matrix:
            return X_aligned, Y_aligned, R0, src_center0, t0
        return X_aligned, Y_aligned, theta, src_center0, t0

    s = _target_scale(Y)
    topk = _auto_topk(m) if topk is None else max(1, min(int(topk), m))

    all_real_cols = set()
    cand_cols_per_row = []
    real_costs_per_row = []

    for i in active_rows:
        cond = pi[i] / max(row_mass[i], eps)

        cand = np.argsort(-cond)[:topk]
        cand = cand[cond[cand] > eps]

        if cand.size == 0:
            cand = np.array([int(np.argmax(pi[i]))], dtype=int)

        x_pred = X[i] @ R0.T + t0
        resid = np.linalg.norm(Y[cand] - x_pred[None, :], axis=1) / s
        prob_cost = -np.log(cond[cand] + eps)

        # Both terms are dimensionless:
        # -log(prob) rewards confident OT mass
        # residual/s rewards rigid consistency with the coarse transform
        cost = prob_cost + resid

        cand_cols_per_row.append(cand)
        real_costs_per_row.append(cost)
        all_real_cols.update(cand.tolist())

    all_real_cols = np.array(sorted(all_real_cols), dtype=int)
    real_col_to_idx = {j: idx for idx, j in enumerate(all_real_cols)}

    nr = len(active_rows)
    nc_real = len(all_real_cols)
    BIG = 1e9
    C = np.full((nr, nc_real + nr), BIG, dtype=float)

    for r, (i, cand, cost) in enumerate(zip(active_rows, cand_cols_per_row, real_costs_per_row)):
        for j, c in zip(cand, cost):
            C[r, real_col_to_idx[int(j)]] = float(c)

        # Private dummy for this row -> allows unmatched source rows
        med = float(np.median(cost))
        mad = float(np.median(np.abs(cost - med)))
        reject_cost = med + 1.4826 * mad
        if not np.isfinite(reject_cost):
            reject_cost = med + 1.0
        if reject_cost <= float(np.min(cost)):
            reject_cost = float(np.min(cost) + max(0.5, mad + 0.5))

        C[r, nc_real + r] = reject_cost

    row_ind, col_ind = linear_sum_assignment(C)

    pairs = []
    weights = []

    for r, c in zip(row_ind, col_ind):
        if c >= nc_real:
            continue  # matched to private dummy -> leave row unmatched

        i = int(active_rows[r])
        j = int(all_real_cols[c])
        pairs.append((i, j))
        weights.append(row_mass[i])

    # ------------------------------------------------------------------
    # 3) Final rigid fit on hard support only
    # ------------------------------------------------------------------
    if len(pairs) >= 2:
        src_idx = np.array([i for i, _ in pairs], dtype=int)
        tgt_idx = np.array([j for _, j in pairs], dtype=int)
        w = np.asarray(weights, dtype=float)

        R, t, src_center, _ = _fit_pairs_rigid(
            X[src_idx],
            Y[tgt_idx],
            w=w,
            allow_reflection_=allow_reflection,
        )
    else:
        R, t, src_center = R0, t0, src_center0

    X_aligned = X @ R.T + t
    Y_aligned = Y.copy()

    if not output_params:
        return X_aligned, Y_aligned

    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    if matrix:
        return X_aligned, Y_aligned, R, src_center, t
    return X_aligned, Y_aligned, theta, src_center, t


def stack_slices_pairwise(
    slices: List[AnnData],
    pis: List[np.ndarray],
    output_params: bool = False,
    matrix: bool = False
) -> Tuple[List[AnnData], Optional[List[float]], Optional[List[np.ndarray]]]:
    """
    Align spatial coordinates of sequential pairwise slices.

    In other words, align:

        slices[0] --> slices[1] --> slices[2] --> ...

    Args:
        slices: List of slices.
        pis: List of pi (``pairwise_align()`` output) between consecutive slices.
        output_params: If ``True``, addtionally return angles of rotation (theta) and translations for each slice.
        matrix: If ``True`` and output_params is also ``True``, the rotation is
            return as a matrix instead of an angle for each slice.

    Returns:
        - List of slices with aligned spatial coordinates.

        If ``output_params = True``, additionally return:

        - List of angles of rotation (theta) for each slice.
        - List of translations [x_translation, y_translation] for each slice.
    """
    assert len(slices) == len(pis) + 1, "'slices' should have length one more than 'pis'. Please double check."
    assert len(slices) > 1, "You should have at least 2 layers."
    new_coor = []
    thetas = []
    translations = []
    if not output_params:
        S1, S2  = generalized_procrustes_analysis(slices[0].obsm['spatial'], slices[1].obsm['spatial'], pis[0])
    else:
        S1, S2,theta,tX,tY  = generalized_procrustes_analysis(slices[0].obsm['spatial'], slices[1].obsm['spatial'], pis[0],output_params=output_params, matrix=matrix)
        thetas.append(theta)
        translations.append(tX)
        translations.append(tY)
    new_coor.append(S1)
    new_coor.append(S2)
    for i in range(1, len(slices) - 1):
        if not output_params:
            x, y = generalized_procrustes_analysis(new_coor[i], slices[i+1].obsm['spatial'], pis[i])
        else:
            x, y,theta,tX,tY = generalized_procrustes_analysis(new_coor[i], slices[i+1].obsm['spatial'], pis[i],output_params=output_params, matrix=matrix)
            thetas.append(theta)
            translations.append(tY)
        new_coor.append(y)

    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm['spatial'] = new_coor[i]
        new_slices.append(s)

    if not output_params:
        return new_slices
    else:
        return new_slices, thetas, translations


def visualize_alignment(sliceA, sliceB, pi12):
    slices, pis = [sliceA, sliceB], [pi12]
    new_slices = stack_slices_pairwise(slices, pis)

    slice_colors = ['#e41a1c','#377eb8']

    xI_new = new_slices[0].obsm['spatial'][:, 0]
    yI_new = new_slices[0].obsm['spatial'][:, 1]

    xJ_new = new_slices[1].obsm['spatial'][:, 0]
    yJ_new = new_slices[1].obsm['spatial'][:, 1]

    print("====================\nAligned slices")

    plt.scatter(xI_new,yI_new,s=1,alpha=0.5, label='source', c=slice_colors[0])
    plt.scatter(xJ_new,yJ_new,s=1,alpha=0.5, label = 'target', c=slice_colors[1])
    plt.axis("off")
    plt.legend()
    plt.show()

    return new_slices


def visualize_cluster_alignment(sliceA: AnnData, sliceB: AnnData, details: dict, show_lines: bool = True, line_alpha: float = 0.5, line_width: float = 1.0):
    """
    Visualizes the coarse cluster-level OT alignment from `hierarchical_pairwise_align`.
    Shows the spatial layout of clusters, scales/translates sliceA rigidly onto sliceB 
    based on the cluster-OT plan, and draws mapping connections.

    Args:
        sliceA: Source AnnData slice.
        sliceB: Target AnnData slice.
        details: The secondary dictionary returned directly by `hierarchical_pairwise_align` when `return_details=True`.
    """
    coarse_plan = details['coarse_plan']
    cluster_labels_A = details['cluster_labels_A']
    cluster_labels_B = details['cluster_labels_B']
    
    coords_A = sliceA.obsm['spatial']
    coords_B = sliceB.obsm['spatial']
    
    unique_A = np.unique(cluster_labels_A)
    unique_B = np.unique(cluster_labels_B)
    
    centroids_A = np.zeros((len(unique_A), 2))
    for idx, c in enumerate(unique_A):
        centroids_A[idx] = coords_A[cluster_labels_A == c].mean(axis=0)
        
    centroids_B = np.zeros((len(unique_B), 2))
    for idx, c in enumerate(unique_B):
        centroids_B[idx] = coords_B[cluster_labels_B == c].mean(axis=0)

    # Perform rigid Procrustes alignment on the clusters driven by the coarse transport plan
    # Strip out dummy row/cols from coarse_plan if they exist (unbalanced matching budget padding)
    clean_plan = coarse_plan[:len(unique_A), :len(unique_B)]
    
    cen_A_aligned, cen_B_aligned, R, src_center, t = generalized_procrustes_analysis(
        centroids_A, centroids_B, clean_plan, 
        output_params=True, matrix=True, allow_reflection=True
    )
    
    # Align the full coordinate set of slice A for visual context
    coords_A_aligned = coords_A @ R.T + t
    
    plt.figure(figsize=(10, 10))
    slice_colors = ['#e41a1c', '#377eb8']
    
    # Plot background generic cells (faint)
    plt.scatter(coords_A_aligned[:, 0], coords_A_aligned[:, 1], s=2, alpha=0.15, label='Slice A cells', c=slice_colors[0])
    plt.scatter(coords_B[:, 0], coords_B[:, 1], s=2, alpha=0.15, label='Slice B cells', c=slice_colors[1])
    
    # Plot matched cluster centroids
    plt.scatter(cen_A_aligned[:, 0], cen_A_aligned[:, 1], s=60, edgecolors='white', linewidths=1, label='Slice A clusters', c=slice_colors[0], marker='o', zorder=5)
    plt.scatter(cen_B_aligned[:, 0], cen_B_aligned[:, 1], s=60, edgecolors='white', linewidths=1, label='Slice B clusters', c=slice_colors[1], marker='s', zorder=5)
    
    if show_lines:
        # Draw assignment lines matching clusters
        # Normalize plan for line visibility
        max_val = np.max(clean_plan)
        if max_val > 0:
            norm_plan = clean_plan / max_val
            for i in range(clean_plan.shape[0]):
                for j in range(clean_plan.shape[1]):
                    weight = norm_plan[i, j]
                    if weight > 0.05:  # Filter out noise traces
                        plt.plot(
                            [cen_A_aligned[i, 0], cen_B_aligned[j, 0]],
                            [cen_A_aligned[i, 1], cen_B_aligned[j, 1]],
                            c='gray', alpha=weight * line_alpha, lw=weight * line_width * 3, zorder=1
                        )
                        
    plt.axis("equal")
    plt.axis("off")
    plt.title("Hierarchical Cluster-Level Unbalanced Alignment", fontsize=14)
    
    # Custom elegant legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), markerscale=2, loc='best')
    
    plt.show()

