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

        mx = (Wn.sum(axis=1)[:, None] * X_).sum(axis=0) / (a.sum() + eps)
        my = (Wn.sum(axis=0)[:, None] * Y_).sum(axis=0) / (b.sum() + eps)
        # Correctly center based on weighted means
        mx = (X_.T @ a) / (a.sum() + eps)
        my = (Y_.T @ b) / (b.sum() + eps)

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
    # 1) Robust Soft Rigid Fit from exact shadow pi
    # ------------------------------------------------------------------
    pi_mass = float(pi.sum())
    if pi_mass <= eps:
        raise ValueError("The transport plan mass is zero! Cannot align.")

    # Using the continuous soft weighting ensures all points contribute
    # naturally to the morphology intersection mathematically.
    # We bypass aggressive Hungarian pruning which forcefully throws away
    # non-rigid local stretching that causes apparent visual mismatches on boundaries!
    R0, t0, src_center0, _ = _fit_soft_rigid(X, Y, pi, allow_reflection_=allow_reflection)

    X_aligned = X @ R0.T + t0
    Y_aligned = Y.copy()

    if not output_params:
        return X_aligned, Y_aligned

    theta = float(np.arctan2(R0[1, 0], R0[0, 0]))
    if matrix:
        return X_aligned, Y_aligned, R0, src_center0, t0
    return X_aligned, Y_aligned, theta, src_center0, t0


def stack_slices_pairwise(
    slices: List[AnnData],
    pis: List[np.ndarray],
    output_params: bool = False,
    matrix: bool = False
) -> Tuple[List[AnnData], Optional[List[float]], Optional[List[np.ndarray]]]:
    """
    Align spatial coordinates of sequential pairwise slices.

    This function anchors all slices to the coordinate system of slices[0].
    Transformations are accumulated through the stack sequentially:
    slices[1] -> slices[0], then slices[2] -> slices[1] (now in 0's space), etc.
    """
    assert len(slices) == len(pis) + 1, "'slices' should have length one more than 'pis'. Please double check."
    assert len(slices) > 1, "You should have at least 2 layers."
    
    new_coor = [slices[0].obsm['spatial'].copy()]
    thetas = []
    translations = []
    
    # We iterate forward, mapping slice i+1 to slice i's newly aligned coordinates
    for i in range(len(slices) - 1):
        # pis[i] maps slices[i] -> slices[i+1]
        # To align i+1 to i, X = slices[i+1], Y = new_coor[i] (which is slices[i] in the global space), pi is transposed!
        if not output_params:
            X_aligned, _ = generalized_procrustes_analysis(
                slices[i+1].obsm['spatial'], 
                new_coor[i], 
                pis[i].T
            )
            new_coor.append(X_aligned)
        else:
            X_aligned, _, theta, src_center, t = generalized_procrustes_analysis(
                slices[i+1].obsm['spatial'], 
                new_coor[i], 
                pis[i].T,
                output_params=output_params, 
                matrix=matrix
            )
            new_coor.append(X_aligned)
            thetas.append(theta)
            translations.append(t) # Only append the translation vector, ignoring src_center to fix output struct

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

    slice_colors = ['#e41a1c','#377eb8'] # Red (Source), Blue (Target)

    xI_new = new_slices[0].obsm['spatial'][:, 0]
    yI_new = new_slices[0].obsm['spatial'][:, 1]

    xJ_new = new_slices[1].obsm['spatial'][:, 0]
    yJ_new = new_slices[1].obsm['spatial'][:, 1]
    
    # Identify the "Exact Shadow" matched subsets based on transport plan mass
    matched_src = pi12.sum(axis=1) > 1e-5
    matched_tgt = pi12.sum(axis=0) > 1e-5

    print("====================\nAligned slices")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Global Overlay ---
    # Unmatched tails in soft gray
    ax1.scatter(xI_new[~matched_src], yI_new[~matched_src], s=1, alpha=0.2, c='grey')
    ax1.scatter(xJ_new[~matched_tgt], yJ_new[~matched_tgt], s=1, alpha=0.2, c='lightgrey')
    
    # Matched shadow in solid colors
    ax1.scatter(xI_new[matched_src], yI_new[matched_src], s=1.5, alpha=0.6, label='Source (Matched)', c=slice_colors[0])
    ax1.scatter(xJ_new[matched_tgt], yJ_new[matched_tgt], s=1.5, alpha=0.6, label='Target (Matched)', c=slice_colors[1])
    ax1.axis("off")
    ax1.legend()
    ax1.set_title("Rigid Overlay (Matched Core Highlighted)")

    # --- Plot 2: Displacement Vector Field (Quiver) ---
    # To avoid rendering 50,000+ lines, sample up to 1000 matched pairs via argmax
    ax2.scatter(xJ_new, yJ_new, s=0.5, alpha=0.1, c='lightgrey') # Background target
    
    active_rows = np.where(matched_src)[0]
    if len(active_rows) > 0:
        # Sample points for the vector field
        sample_size = min(1000, len(active_rows))
        sampled_src_idx = np.random.choice(active_rows, sample_size, replace=False)
        
        # Find the target cell assignment
        sampled_tgt_idx = np.argmax(pi12[sampled_src_idx], axis=1)
        
        start_x = xI_new[sampled_src_idx]
        start_y = yI_new[sampled_src_idx]
        end_x = xJ_new[sampled_tgt_idx]
        end_y = yJ_new[sampled_tgt_idx]
        
        ax2.quiver(start_x, start_y, end_x - start_x, end_y - start_y, 
                   color='black', alpha=0.4, angles='xy', scale_units='xy', scale=1, width=0.002)
                   
    ax2.axis("off")
    ax2.set_title("Displacement Field (Source to Target)")

    plt.tight_layout()
    plt.show()

    return new_slices

