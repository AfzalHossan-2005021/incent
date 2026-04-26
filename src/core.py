import ot
import heapq
import torch
import numpy as np

from anndata import AnnData
from collections import Counter
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from typing import Optional, Tuple, Union, Dict, Any, List
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from .utils import select_backend, fused_gromov_wasserstein_incent, to_dense_array, extract_data_matrix, jensenshannon_divergence_backend, to_backend


def pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    reg_compact: float = 0.001,
    armijo: bool = True,
    radius: Optional[float] = None,
    use_rep: Optional[str] = None,
    G_init = None,
    a_distribution = None,
    b_distribution = None,
    numItermax: int = 6000,
    use_gpu: bool = True,
    data_type = np.float32,
    epsilon: float = 1e-6,
    verbose: bool = False,
    gpu_verbose: bool = True,
    unbalanced: bool = True,
    **kwargs) -> NDArray[np.floating]:
    """
    Compute the INCENT optimal-transport alignment between two single-cell
    spatial transcriptomics slices.

    This solves a fused Gromov-Wasserstein (FGW) problem in which the linear
    cost combines gene-expression dissimilarity and cell-type mismatch, the
    quadratic cost combines pairwise spatial distances, and the joint
    objective is augmented with a multiscale rotation-invariant neighborhood
    descriptor cost ``M2`` and (optionally) a "Form A" spatial-compactness
    regularizer.

    Args:
        sliceA: Source slice (``AnnData``). Requires ``obsm['spatial']`` and
            ``obs['cell_type_annot']``.
        sliceB: Target slice (``AnnData``). Same requirements as ``sliceA``.
        alpha: Mixing weight on the GW (spatial-structure) term in
            ``[0, 1]``. ``alpha=0`` reduces to a pure linear OT in the
            feature cost ``M1 + gamma * M2``; ``alpha=1`` to a pure GW
            problem.
        beta: Mixing weight on the cell-type-mismatch term inside ``M1``.
            ``M1 = (1 - beta) * cosine_distance(X_A, X_B) + beta * 1[label_A != label_B]``.
        gamma: Weight on the multiscale neighborhood cost ``M2`` added to
            ``M1`` when forming the linear cost.
        reg_compact: Coefficient of the spatial-compactness regularizer
            applied inside :func:`fused_gromov_wasserstein_incent`. Set to
            ``0`` to disable.
        armijo: If True, use Armijo line-search inside conditional gradient.
        radius: Optional fixed radius for the neighborhood descriptor; when
            ``None``, multiscale radii are estimated from local spacing.
        use_rep: If ``None``, gene-expression distance is computed from
            ``slice.X``; otherwise from ``slice.obsm[use_rep]``.
        G_init: Optional initial transport plan ``(n_A, n_B)``. Defaults to
            the outer product of the marginals.
        a_distribution: Optional source marginal. Defaults to uniform when
            ``unbalanced=False``; not supported when ``unbalanced=True``.
        b_distribution: Optional target marginal. Same constraints as
            ``a_distribution``.
        numItermax: Maximum number of conditional-gradient iterations.
        use_gpu: If True and CUDA is available, use the PyTorch backend.
        data_type: Backend tensor dtype. Default ``np.float32``.
        epsilon: Numerical-stability constant added in distance/cosine
            computations.
        verbose: If True, FGW-OT prints inner-loop diagnostics.
        gpu_verbose: If True, print backend selection messages.
        unbalanced: If True, augment the marginals with a label-aware
            "dummy" cell on whichever side has fewer cells (per-type budget),
            which absorbs unmatched mass.
        **kwargs: Forwarded to :func:`fused_gromov_wasserstein_incent`
            (e.g. ``numItermaxEmd``, ``tol_rel``, ``tol_abs``).

    Returns:
        ``pi`` of shape ``(n_A, n_B)``: the soft transport plan from cells
        of ``sliceA`` to cells of ``sliceB``.

    Notes:
        Use :func:`incent.calculate_performance_metrics` after this call to
        obtain initial/final neighborhood, gene-expression, and cell-type
        matching metrics; that helper replaces the previous ``return_obj``
        flag.
    """
    
    # Determine if gpu or cpu is being used
    use_gpu, nx = select_backend(use_gpu=use_gpu, gpu_verbose=gpu_verbose)
    
    
    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.")   
    
    # ────────────────────── Calculate spatial distances ──────────────────────
    D_A, D_B = calculate_spatial_distance(sliceA, sliceB, nx, data_type=data_type, eps=epsilon)
    

    # ────────────────────── Calculate gene expression dissimilarity ──────────────────────
    cosine_dist_gene_expr = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, eps=epsilon)


    # ────────────────────── Calculate cell-type mismatch penalty ──────────────────────
    cell_type_mismatch = calculate_cell_type_mismatch(sliceA, sliceB)


    # Combine gene expression dissimilarity and cell-type mismatch penalty into a single cost matrix M1
    M1_combined = (1 - beta) * cosine_dist_gene_expr + beta * cell_type_mismatch
    M1 = to_backend(M1_combined, nx, data_type=data_type)


    # ────────────────────── Calculate neighborhood dissimilarity ──────────────────────
    js_dist_neighborhood = calculate_neighborhood_dissimilarity(
        sliceA,
        sliceB,
        radius=radius,                 # optional; if None multiscale radii are estimated
        nx=nx,
        data_type=data_type,
        eps=epsilon,
        radii=None,                    # or pass an explicit list, e.g. [20, 35, 50]
        radius_k=6,
        radius_multipliers=(2.5, 4.0, 6.0),
        n_shells=3,
        harmonics=(0, 1, 2),
        harmonic_weights={1: 1.25, 2: 1.5},
        distance_decay="linear",
        include_self=False,
    )
    M2 = to_backend(js_dist_neighborhood, nx, data_type=data_type)


    # ── Dummy cell augmentation ────────────────────────────────────────────
    # Only add a dummy on the side that actually has a deficit.
    # _has_dummy_src: True if source has fewer cells → need dummy source (birth)
    # _has_dummy_tgt: True if target has fewer cells → need dummy target (death)
    _has_dummy_src = False
    _has_dummy_tgt = False

    if unbalanced:
        ns, nt = sliceA.shape[0], sliceB.shape[0]
        labels_A = sliceA.obs['cell_type_annot'].values
        labels_B = sliceB.obs['cell_type_annot'].values
        counts_A = Counter(labels_A)
        counts_B = Counter(labels_B)
        all_types = set(counts_A.keys()) | set(counts_B.keys())
        _budget = sum(max(counts_A.get(k, 0), counts_B.get(k, 0)) for k in all_types)
        _w_dummy_src = _budget - ns   # extra weight for dummy source cell (birth)
        _w_dummy_tgt = _budget - nt   # extra weight for dummy target cell (death)
        _has_dummy_src = _w_dummy_src > 0
        _has_dummy_tgt = _w_dummy_tgt > 0

        print(f"[unbalanced] budget={_budget}, "
              f"dummy_src={'YES (birth)' if _has_dummy_src else 'NO'} w={_w_dummy_src}, "
              f"dummy_tgt={'YES (death)' if _has_dummy_tgt else 'NO'} w={_w_dummy_tgt}")

        _ns_aug = ns + (1 if _has_dummy_src else 0)
        _nt_aug = nt + (1 if _has_dummy_tgt else 0)

        # ---- Augment D_A if dummy source needed ----
        if _has_dummy_src:
            zeros_col = nx.zeros((ns, 1), type_as=D_A)
            D_A_temp = nx.concatenate([D_A, zeros_col], axis=1)
            zeros_row = nx.zeros((1, _ns_aug), type_as=D_A)
            D_A = nx.concatenate([D_A_temp, zeros_row], axis=0)

        # ---- Augment D_B if dummy target needed ----
        if _has_dummy_tgt:
            zeros_col = nx.zeros((nt, 1), type_as=D_B)
            D_B_temp = nx.concatenate([D_B, zeros_col], axis=1)
            zeros_row = nx.zeros((1, _nt_aug), type_as=D_B)
            D_B = nx.concatenate([D_B_temp, zeros_row], axis=0)
        
        # ---- Augment M1: add dummy row/col only where needed ----
        if _has_dummy_tgt:
            mean_col = nx.mean(M1, axis=1)
            mean_col = nx.reshape(mean_col, (-1, 1))
            M1 = nx.concatenate([M1, mean_col], axis=1)
        if _has_dummy_src:
            mean_row = nx.mean(M1, axis=0)
            mean_row = nx.reshape(mean_row, (1, -1))
            M1 = nx.concatenate([M1, mean_row], axis=0)
        if _has_dummy_src and _has_dummy_tgt:
            M1[-1, -1] = 0.0

        # ---- Augment M2: add dummy row/col only where needed ----
        if _has_dummy_tgt:
            mean_col = nx.mean(M2, axis=1)
            mean_col = nx.reshape(mean_col, (-1, 1))
            M2 = nx.concatenate([M2, mean_col], axis=1)
        if _has_dummy_src:
            mean_row = nx.mean(M2, axis=0)
            mean_row = nx.reshape(mean_row, (1, -1))
            M2 = nx.concatenate([M2, mean_row], axis=0)
        if _has_dummy_src and _has_dummy_tgt:
            M2[-1, -1] = 0.0

    # Free up memory from CPU and intermediate tensors created prior to OT
    del cosine_dist_gene_expr, cell_type_mismatch, M1_combined, js_dist_neighborhood
    if isinstance(nx, ot.backend.TorchBackend):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # init distributions
    if a_distribution is None:
        if unbalanced:
            if _has_dummy_src:
                a_vals = np.full(ns + 1, 1.0 / _budget, dtype=np.float64)
                a_vals[-1] = float(_w_dummy_src) / _budget
                a = nx.from_numpy(a_vals)
            else:
                a = nx.ones((ns,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            # uniform distribution, a = array([1/n, 1/n, ...])
            a = nx.ones((sliceA.shape[0],))/sliceA.shape[0]
    else:
        if unbalanced:
            raise ValueError("Custom a_distribution is not supported with unbalanced=True.")
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        if unbalanced:
            if _has_dummy_tgt:
                b_vals = np.full(nt + 1, 1.0 / _budget, dtype=np.float64)
                b_vals[-1] = float(_w_dummy_tgt) / _budget
                b = nx.from_numpy(b_vals)
            else:
                b = nx.ones((nt,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        if unbalanced:
            raise ValueError("Custom b_distribution is not supported with unbalanced=True.")
        b = nx.from_numpy(b_distribution)

    a = to_backend(a, nx, data_type=data_type)
    b = to_backend(b, nx, data_type=data_type)
    
    
    # Run OT
    if G_init is not None:
        if unbalanced and (_has_dummy_src or _has_dummy_tgt):
            # Pad user-provided (ns x nt) G_init to augmented dims
            _gi = G_init.cpu().numpy() if hasattr(G_init, 'cpu') else np.array(G_init, dtype=data_type)
            _gi_aug = np.zeros((_ns_aug, _nt_aug), dtype=data_type)
            _gi_aug[:ns, :nt] = _gi
            G_init = to_backend(_gi_aug, nx, data_type=data_type)
    
    pi = fused_gromov_wasserstein_incent(M1 + gamma * M2, D_A, D_B, a, b, G_init = G_init, alpha= alpha, reg_compact=reg_compact, armijo=armijo, numItermax=numItermax, verbose=verbose, **kwargs)
    pi = nx.to_numpy(pi)

    # ── Dummy cell: strip dummy row/col, renormalize, report birth/death ────
    if unbalanced and (_has_dummy_src or _has_dummy_tgt):
        pi_full = pi.copy()

        # Compute birth / death mass before stripping
        birth_mass = float(pi_full[-1, :nt].sum()) if _has_dummy_src else 0.0
        death_mass = float(pi_full[:ns, -1].sum()) if _has_dummy_tgt else 0.0

        # Strip only the dummy row/col that was actually added
        if _has_dummy_src and _has_dummy_tgt:
            pi = pi_full[:ns, :nt]
        elif _has_dummy_src:
            pi = pi_full[:ns, :]      # strip dummy source row only
        elif _has_dummy_tgt:
            pi = pi_full[:, :nt]      # strip dummy target col only

        print(f"[unbalanced] death_mass: {death_mass:.6f}, birth_mass: {birth_mass:.6f}")

    if isinstance(nx, ot.backend.TorchBackend):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pi


def estimate_characteristic_spacing(adata, k=6, spatial_key="spatial"):
    """
    Robust local spacing estimate: median distance to the k-th nearest neighbor.
    Useful for choosing radii that scale with local cell density.
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return 1.0

    k_eff = min(k + 1, n)  # +1 because nearest neighbor includes self at distance 0
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k_eff)

    kth = dists[:, -1]
    kth = kth[np.isfinite(kth) & (kth > 0)]
    if kth.size == 0:
        return 1.0
    return float(np.median(kth))


def equal_area_shell_edges(radius, n_shells):
    """
    Equal-area shells reduce the trivial bias that outer shells cover more area.
    """
    return radius * np.sqrt(np.linspace(0.0, 1.0, n_shells + 1))


def distance_weights(dist, radius, mode="linear", sigma=None):
    """
    Distance weighting for neighbors inside a radius.
    """
    if mode is None or mode == "uniform":
        return np.ones_like(dist, dtype=np.float64)

    if mode == "linear":
        return np.maximum(0.0, 1.0 - dist / radius)

    if mode == "gaussian":
        s = radius / 2.0 if sigma is None else float(sigma)
        if s <= 0:
            raise ValueError("sigma must be > 0")
        return np.exp(-(dist ** 2) / (2.0 * s * s))

    raise ValueError("distance_decay must be one of: None, 'uniform', 'linear', 'gaussian'")


def neighborhood_distribution_fourier(
    adata,
    radius,
    cell_types=None,
    n_shells=3,
    shell_edges=None,
    harmonics=(0, 1, 2),
    harmonic_weights=None,
    distance_decay="linear",
    sigma=None,
    include_self=False,
    area_normalize=True,
    add_empty_bin=True,
    l1_normalize=True,
    dtype=np.float32,
    spatial_key="spatial",
    label_key="cell_type_annot",
    return_metadata=False,
):
    """
    Rotation-invariant neighborhood descriptor.

    For each focal cell, each cell type, and each radial shell:
      m=0 -> abundance
      m=1 -> one-sidedness
      m=2 -> bilateral / opposite-half structure

    Output is nonnegative and suitable for Jensen-Shannon after normalization.
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    labels = adata.obs[label_key].astype(str).to_numpy()

    if cell_types is None:
        cell_types = np.array(sorted(np.unique(labels)), dtype=str)
    else:
        cell_types = np.array(cell_types, dtype=str)

    label_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    missing = sorted(set(labels) - set(cell_types))
    if missing:
        raise ValueError(f"Labels missing from cell_types: {missing}")

    label_idx = np.array([label_to_idx[x] for x in labels], dtype=np.int32)

    harmonics = tuple(sorted(set(int(h) for h in harmonics)))
    if any(h < 0 for h in harmonics):
        raise ValueError("harmonics must be non-negative integers")
    if 0 not in harmonics:
        harmonics = (0,) + harmonics

    if harmonic_weights is None:
        harmonic_weights = {}
    else:
        harmonic_weights = {int(k): float(v) for k, v in harmonic_weights.items()}

    if shell_edges is None:
        shell_edges = equal_area_shell_edges(radius, n_shells)
    else:
        shell_edges = np.asarray(shell_edges, dtype=np.float64)
        if not np.isclose(shell_edges[0], 0.0):
            raise ValueError("shell_edges must start at 0")
        if not np.isclose(shell_edges[-1], radius):
            raise ValueError("shell_edges must end at radius")
        if np.any(np.diff(shell_edges) <= 0):
            raise ValueError("shell_edges must be strictly increasing")

    n_cells = coords.shape[0]
    n_types = len(cell_types)
    n_shells_eff = len(shell_edges) - 1
    n_harm = len(harmonics)

    shell_areas = np.pi * (shell_edges[1:] ** 2 - shell_edges[:-1] ** 2)

    # group index = type * n_shells + shell
    n_groups = n_types * n_shells_eff
    group_shell_idx = np.tile(np.arange(n_shells_eff), n_types)
    group_area = shell_areas[group_shell_idx]

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=radius)

    n_core = n_groups * n_harm
    n_total = n_core + (1 if add_empty_bin else 0)
    features = np.zeros((n_cells, n_total), dtype=np.float64)

    for i, nbr in enumerate(neighbor_lists):
        nbr = np.asarray(nbr, dtype=np.int32)

        if not include_self:
            nbr = nbr[nbr != i]

        if nbr.size == 0:
            if add_empty_bin:
                features[i, -1] = 1.0
            continue

        rel = coords[nbr] - coords[i]
        dist = np.linalg.norm(rel, axis=1)
        theta = np.arctan2(rel[:, 1], rel[:, 0])

        shell_idx = np.searchsorted(shell_edges[1:], dist, side="left")
        valid = (shell_idx >= 0) & (shell_idx < n_shells_eff)

        if not np.all(valid):
            nbr = nbr[valid]
            dist = dist[valid]
            theta = theta[valid]
            shell_idx = shell_idx[valid]

        if nbr.size == 0:
            if add_empty_bin:
                features[i, -1] = 1.0
            continue

        w = distance_weights(dist, radius=radius, mode=distance_decay, sigma=sigma)
        group_idx = label_idx[nbr] * n_shells_eff + shell_idx

        local = np.zeros((n_groups, n_harm), dtype=np.float64)

        for h_pos, m in enumerate(harmonics):
            if m == 0:
                mag = np.bincount(group_idx, weights=w, minlength=n_groups).astype(np.float64)
            else:
                ang = m * theta
                real = np.bincount(group_idx, weights=w * np.cos(ang), minlength=n_groups)
                imag = np.bincount(group_idx, weights=w * np.sin(ang), minlength=n_groups)
                mag = np.hypot(real, imag)

            if area_normalize:
                mag = mag / np.maximum(group_area, 1e-12)

            mag *= harmonic_weights.get(m, 1.0)
            local[:, h_pos] = mag

        flat = local.reshape(-1)

        if flat.sum() == 0:
            if add_empty_bin:
                features[i, -1] = 1.0
        else:
            if add_empty_bin:
                features[i, :-1] = flat
            else:
                features[i] = flat

    if l1_normalize:
        row_sums = features.sum(axis=1, keepdims=True)
        nz = row_sums[:, 0] > 0
        features[nz] /= row_sums[nz]

    features = features.astype(dtype, copy=False)

    if not return_metadata:
        return features

    metadata = {
        "cell_types": cell_types,
        "shell_edges": shell_edges,
        "harmonics": harmonics,
        "feature_shape": (n_types, n_shells_eff, n_harm),
    }
    return features, metadata


def default_radii_from_spacing(sliceA, sliceB, k=6, multipliers=(2.5, 4.0, 6.0), spatial_key="spatial"):
    sA = estimate_characteristic_spacing(sliceA, k=k, spatial_key=spatial_key)
    sB = estimate_characteristic_spacing(sliceB, k=k, spatial_key=spatial_key)
    base = max(sA, sB)
    return [m * base for m in multipliers]


def neighborhood_distribution_multiscale(
    adata,
    radii,
    cell_types=None,
    n_shells=3,
    harmonics=(0, 1, 2),
    harmonic_weights=None,
    distance_decay="linear",
    sigma=None,
    include_self=False,
    area_normalize=True,
    add_empty_bin_per_scale=False,
    l1_normalize_within_scale=True,
    final_l1_normalize=True,
    dtype=np.float32,
    spatial_key="spatial",
    label_key="cell_type_annot",
    return_metadata=False,
):
    """
    Concatenate rotation-invariant descriptors across multiple radii.
    """
    radii = [float(r) for r in radii]
    if any(r <= 0 for r in radii):
        raise ValueError("all radii must be > 0")

    blocks = []
    meta_blocks = []

    for r in radii:
        feat, meta = neighborhood_distribution_fourier(
            adata=adata,
            radius=r,
            cell_types=cell_types,
            n_shells=n_shells,
            harmonics=harmonics,
            harmonic_weights=harmonic_weights,
            distance_decay=distance_decay,
            sigma=sigma,
            include_self=include_self,
            area_normalize=area_normalize,
            add_empty_bin=add_empty_bin_per_scale,
            l1_normalize=l1_normalize_within_scale,
            dtype=np.float64,
            spatial_key=spatial_key,
            label_key=label_key,
            return_metadata=True,
        )
        blocks.append(feat)
        meta_blocks.append({"radius": r, **meta})

    X = np.concatenate(blocks, axis=1)

    if final_l1_normalize:
        row_sums = X.sum(axis=1, keepdims=True)
        nz = row_sums[:, 0] > 0
        X[nz] /= row_sums[nz]

    X = X.astype(dtype, copy=False)

    if not return_metadata:
        return X

    return X, {"scales": meta_blocks}


def calculate_spatial_distance(sliceA, sliceB, nx, data_type=np.float32, spatial_key = 'spatial', eps=1e-8, norm_k=6):
    """
    Calculate spatial distance between cells in a slice, normalized by robust local spacing.

    Args:
        sliceA: First slice for which to calculate spatial distance.
        sliceB: Second slice for which to calculate spatial distance.
        nx: Backend to use for calculations.
        data_type: Data type for backend tensors.
        spatial_key: Key for the spatial coordinates.
        eps: Small constant to avoid division by zero.
        norm_k: Kth neighbor to use for characteristic spacing estimation.
    Returns:
    D_A, D_B: Pairwise spatial distance matrices.
    """
    
    print("Calculating spatial distance between cells in slice A and slice B")

    coordinates_A = np.asarray(sliceA.obsm[spatial_key], dtype=np.float64)
    coordinates_B = np.asarray(sliceB.obsm[spatial_key], dtype=np.float64)

    D_A = euclidean_distances(coordinates_A, coordinates_A)
    D_B = euclidean_distances(coordinates_B, coordinates_B)

    # Normalize by local characteristic spacing instead of global max tissue diameter
    scale_A = estimate_characteristic_spacing(sliceA, k=norm_k, spatial_key=spatial_key)
    scale_B = estimate_characteristic_spacing(sliceB, k=norm_k, spatial_key=spatial_key)
    scale = max(scale_A, scale_B, eps)

    D_A = D_A / scale
    D_B = D_B / scale

    D_A = to_backend(D_A, nx, data_type=data_type)
    D_B = to_backend(D_B, nx, data_type=data_type)

    return D_A, D_B


def calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, eps = 1e-6):
    """
    Calculate cosine distance between gene expression profiles of slice A and slice B.
    
    Args:
    sliceA: First slice.
    sliceB: Second slice.
    use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity
                between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
    eps: Small constant to add to data matrices to avoid zero vectors.
    
    Returns:
    cosine_dist_gene_expr: Cosine distance matrix between gene expression profiles of slice A and slice B.
    """
    
    print("Calculating cosine distance between gene expression profiles of slice A and slice B")
    
    # Extract and prepare data matrices for cosine distance calculation
    A_X = extract_data_matrix(sliceA, use_rep)
    B_X = extract_data_matrix(sliceB, use_rep)

    # Convert to dense arrays and add small constant to avoid zero vectors
    A_X = to_dense_array(A_X) + eps
    B_X = to_dense_array(B_X) + eps

    # Use sklearn's optimized and numerically stable cosine_distances
    cosine_dist_gene_expr = cosine_distances(A_X, B_X)

    return cosine_dist_gene_expr


def calculate_cell_type_mismatch(sliceA, sliceB):
    """
    Calculate the cell-type mismatch penalty between two slices.

    Args:
        sliceA: First slice.
        sliceB: Second slice.

    Returns:
        cell_type_mismatch: Binary matrix indicating cell-type mismatches.
    """

    _lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    _lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)

    cell_type_mismatch = (_lab_A[:, None] != _lab_B[None, :]).astype(np.float64)

    return cell_type_mismatch


def neighborhood_distributions(
    sliceA,
    sliceB,
    radius=None,
    eps=1e-8,
    radii=None,
    radius_k=6,
    radius_multipliers=(2.5, 4.0, 6.0),
    n_shells=3,
    harmonics=(0, 1, 2),
    harmonic_weights=None,
    distance_decay="linear",
    sigma=None,
    include_self=False,
    spatial_key="spatial",
    label_key="cell_type_annot",
):
    """
    Neighborhood dissimilarity using multiscale rotation-invariant descriptors
    and proper re-normalization before Jensen-Shannon distance.
    """
    all_types = np.array(sorted(
        set(sliceA.obs[label_key].astype(str)) |
        set(sliceB.obs[label_key].astype(str))
    ), dtype=str)

    if radii is None:
        if radius is not None:
            radii = [float(radius)]
        else:
            radii = default_radii_from_spacing(
                sliceA,
                sliceB,
                k=radius_k,
                multipliers=radius_multipliers,
                spatial_key=spatial_key,
            )

    featA = neighborhood_distribution_multiscale(
        sliceA,
        radii=radii,
        cell_types=all_types,
        n_shells=n_shells,
        harmonics=harmonics,
        harmonic_weights=harmonic_weights,
        distance_decay=distance_decay,
        sigma=sigma,
        include_self=include_self,
        area_normalize=True,
        add_empty_bin_per_scale=False,
        l1_normalize_within_scale=True,
        final_l1_normalize=True,
        dtype=np.float32,
        spatial_key=spatial_key,
        label_key=label_key,
    )

    featB = neighborhood_distribution_multiscale(
        sliceB,
        radii=radii,
        cell_types=all_types,
        n_shells=n_shells,
        harmonics=harmonics,
        harmonic_weights=harmonic_weights,
        distance_decay=distance_decay,
        sigma=sigma,
        include_self=include_self,
        area_normalize=True,
        add_empty_bin_per_scale=False,
        l1_normalize_within_scale=True,
        final_l1_normalize=True,
        dtype=np.float32,
        spatial_key=spatial_key,
        label_key=label_key,
    )

    # Add epsilon, THEN renormalize.
    featA = featA + eps
    featB = featB + eps
    featA = featA / featA.sum(axis=1, keepdims=True)
    featB = featB / featB.sum(axis=1, keepdims=True)

    return featA, featB


def calculate_neighborhood_dissimilarity(
    sliceA,
    sliceB,
    radius=None,
    nx=None,
    data_type=np.float32,
    eps=1e-8,
    radii=None,
    radius_k=6,
    radius_multipliers=(2.5, 4.0, 6.0),
    n_shells=3,
    harmonics=(0, 1, 2),
    harmonic_weights=None,
    distance_decay="linear",
    sigma=None,
    include_self=False,
    spatial_key="spatial",
    label_key="cell_type_annot",
):
    """
    Neighborhood dissimilarity using multiscale rotation-invariant descriptors
    and proper re-normalization before Jensen-Shannon distance.
    """
    featA, featB = neighborhood_distributions(
        sliceA,
        sliceB,
        radius=radius,
        eps=eps,
        radii=radii,
        radius_k=radius_k,
        radius_multipliers=radius_multipliers,
        n_shells=n_shells,
        harmonics=harmonics,
        harmonic_weights=harmonic_weights,
        distance_decay=distance_decay,
        sigma=sigma,
        include_self=include_self,
        spatial_key=spatial_key,
        label_key=label_key,
    )

    featA = to_backend(featA, nx, data_type=data_type)
    featB = to_backend(featB, nx, data_type=data_type)

    return jensenshannon_divergence_backend(featA, featB)


def _safe_normalize_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = float(x.sum())
    if s <= eps:
        return np.full_like(x, 1.0 / max(len(x), 1), dtype=np.float64)
    return x / s


def _normalize_cost_matrix(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    finite = np.isfinite(M)
    if not np.any(finite):
        return np.zeros_like(M, dtype=np.float64)
    lo = float(np.min(M[finite]))
    hi = float(np.max(M[finite]))
    if hi - lo <= eps:
        out = np.zeros_like(M, dtype=np.float64)
        out[~finite] = 1.0
        return out
    out = (M - lo) / (hi - lo)
    out[~finite] = 1.0
    return out


def _pairwise_js_distance(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)
    P /= P.sum(axis=1, keepdims=True)
    Q /= Q.sum(axis=1, keepdims=True)

    js = np.empty((P.shape[0], Q.shape[0]), dtype=np.float64)
    for i in range(P.shape[0]):
        m = 0.5 * (P[i][None, :] + Q)
        kl_pm = np.sum(P[i][None, :] * (np.log(P[i][None, :]) - np.log(m)), axis=1)
        kl_qm = np.sum(Q * (np.log(Q) - np.log(m)), axis=1)
        js[i] = np.sqrt(np.maximum(0.0, 0.5 * (kl_pm + kl_qm)))
    return js


def _extract_embedding_matrix(adata: AnnData, use_rep: Optional[str] = None) -> np.ndarray:
    X = extract_data_matrix(adata, use_rep)
    if hasattr(X, 'toarray'):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _compute_local_density(adata: AnnData, k: int = 6, spatial_key: str = 'spatial', eps: float = 1e-8) -> np.ndarray:
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n = coords.shape[0]
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    k_eff = min(k + 1, n)
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k_eff)
    local_scale = np.mean(dists[:, 1:], axis=1)
    density = 1.0 / np.maximum(local_scale, eps)
    med = np.median(density[density > 0]) if np.any(density > 0) else 1.0
    return density / max(med, eps)


def _farthest_point_seeds(coords: np.ndarray, n_seeds: int) -> np.ndarray:
    n = coords.shape[0]
    n_seeds = int(max(1, min(n_seeds, n)))
    seeds = [int(np.argmax(np.linalg.norm(coords - coords.mean(axis=0), axis=1)))]
    min_d2 = np.sum((coords - coords[seeds[0]]) ** 2, axis=1)
    while len(seeds) < n_seeds:
        nxt = int(np.argmax(min_d2))
        if nxt in seeds:
            break
        seeds.append(nxt)
        min_d2 = np.minimum(min_d2, np.sum((coords - coords[nxt]) ** 2, axis=1))
    while len(seeds) < n_seeds:
        candidate = len(seeds) % n
        if candidate not in seeds:
            seeds.append(candidate)
    return np.asarray(seeds, dtype=np.int32)


def _build_spatial_knn_graph(coords: np.ndarray, k_neighbors: int = 10) -> Tuple[List[List[int]], List[List[float]], List[Tuple[int, int, float]]]:
    n = coords.shape[0]
    if n == 0:
        return [], [], []
    k_eff = max(1, min(int(k_neighbors), n - 1)) if n > 1 else 0
    if k_eff == 0:
        return [[] for _ in range(n)], [[] for _ in range(n)], []

    tree = cKDTree(coords)
    dists, idx = tree.query(coords, k=k_eff + 1)
    neighbors = [[] for _ in range(n)]
    weights = [[] for _ in range(n)]
    edges = {}

    for i in range(n):
        for j, d in zip(idx[i, 1:], dists[i, 1:]):
            j = int(j)
            d = float(d)
            if i == j:
                continue
            neighbors[i].append(j)
            weights[i].append(d)
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = min(d, edges.get((a, b), d))

    for (i, j), d in list(edges.items()):
        if i not in neighbors[j]:
            neighbors[j].append(i)
            weights[j].append(d)
        if j not in neighbors[i]:
            neighbors[i].append(j)
            weights[i].append(d)

    edge_list = [(i, j, d) for (i, j), d in edges.items()]
    return neighbors, weights, edge_list


def _build_supercell_features(
    adata: AnnData,
    use_rep: Optional[str] = None,
    spatial_key: str = 'spatial',
    label_key: str = 'cell_type_annot',
    density_k: int = 6,
) -> Dict[str, Any]:
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    labels = adata.obs[label_key].astype(str).to_numpy()
    density = _compute_local_density(adata, k=density_k, spatial_key=spatial_key)
    unique_types = np.array(sorted(np.unique(labels)), dtype=str)
    one_hot = np.zeros((coords.shape[0], len(unique_types)), dtype=np.float64)
    label_to_idx = {lab: i for i, lab in enumerate(unique_types)}
    for i, lab in enumerate(labels):
        one_hot[i, label_to_idx[lab]] = 1.0

    embedding = _extract_embedding_matrix(adata, use_rep)
    feat = np.concatenate([
        one_hot,
        density[:, None],
        embedding / np.maximum(np.std(embedding, axis=0, keepdims=True) + 1e-8, 1e-8),
    ], axis=1)

    feat -= feat.mean(axis=0, keepdims=True)
    feat /= np.maximum(feat.std(axis=0, keepdims=True), 1e-8)

    return {
        'coords': coords,
        'labels': labels,
        'density': density,
        'embedding': embedding,
        'feature_matrix': feat,
        'cell_types': unique_types,
    }


def _balanced_region_growing_labels(
    coords: np.ndarray,
    node_features: np.ndarray,
    neighbors: List[List[int]],
    edge_weights: List[List[float]],
    n_clusters: int,
    feature_weight: float = 0.25,
) -> np.ndarray:
    n = coords.shape[0]
    if n_clusters <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)

    n_clusters = int(max(1, min(n_clusters, n)))
    seeds = _farthest_point_seeds(coords, n_clusters)
    target_sizes = np.full(n_clusters, n // n_clusters, dtype=np.int32)
    target_sizes[: n % n_clusters] += 1

    flat_weights = np.concatenate([np.asarray(w, dtype=np.float64) for w in edge_weights if len(w) > 0]) if any(len(w) > 0 for w in edge_weights) else np.array([1.0])
    edge_scale = float(np.median(flat_weights[flat_weights > 0])) if np.any(flat_weights > 0) else 1.0

    assigned = np.full(n, -1, dtype=np.int32)
    cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
    cluster_coord_sum = np.zeros((n_clusters, coords.shape[1]), dtype=np.float64)
    cluster_feat_sum = np.zeros((n_clusters, node_features.shape[1]), dtype=np.float64)
    heap = []

    for c, seed in enumerate(seeds):
        heapq.heappush(heap, (0.0, c, int(seed), int(seed)))

    while heap:
        cost, c, node, parent = heapq.heappop(heap)
        if assigned[node] != -1:
            continue
        if cluster_sizes[c] >= target_sizes[c]:
            continue

        assigned[node] = c
        cluster_sizes[c] += 1
        cluster_coord_sum[c] += coords[node]
        cluster_feat_sum[c] += node_features[node]

        coord_center = cluster_coord_sum[c] / cluster_sizes[c]
        feat_center = cluster_feat_sum[c] / cluster_sizes[c]

        for nbr, edge_d in zip(neighbors[node], edge_weights[node]):
            if assigned[nbr] != -1:
                continue
            spatial_term = float(edge_d) / max(edge_scale, 1e-8)
            frontier_term = np.linalg.norm(coords[nbr] - coord_center) / max(edge_scale, 1e-8)
            feature_term = np.linalg.norm(node_features[nbr] - feat_center) / max(np.sqrt(node_features.shape[1]), 1.0)
            score = cost + spatial_term + 0.15 * frontier_term + feature_weight * feature_term
            heapq.heappush(heap, (score, c, int(nbr), node))

    leftover = np.where(assigned < 0)[0]
    if leftover.size:
        for node in leftover:
            not_full = np.where(cluster_sizes < target_sizes)[0]
            if not_full.size == 0:
                not_full = np.arange(n_clusters)
            centers = cluster_coord_sum[not_full] / np.maximum(cluster_sizes[not_full][:, None], 1)
            costs = np.linalg.norm(centers - coords[node], axis=1)
            c = int(not_full[np.argmin(costs)])
            assigned[node] = c
            cluster_sizes[c] += 1
            cluster_coord_sum[c] += coords[node]
            cluster_feat_sum[c] += node_features[node]

    return assigned.astype(np.int32)


def _cluster_mean_rows(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    out = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size:
            out[c] = X[idx].mean(axis=0)
    return out


def _compute_cluster_statistics(
    adata: AnnData,
    cluster_labels: np.ndarray,
    coords: np.ndarray,
    cell_graph_edges: List[Tuple[int, int, float]],
    density: np.ndarray,
    neighborhood_features: np.ndarray,
    cell_types_union: np.ndarray,
    use_rep: Optional[str] = None,
    spatial_key: str = 'spatial',
    label_key: str = 'cell_type_annot',
    graph_structure_weight: float = 0.65,
) -> Dict[str, Any]:
    n_clusters = int(cluster_labels.max()) + 1
    embedding = _extract_embedding_matrix(adata, use_rep)
    labels = adata.obs[label_key].astype(str).to_numpy()
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters).astype(np.float64)
    centroids = np.zeros((n_clusters, coords.shape[1]), dtype=np.float64)
    radii = np.zeros(n_clusters, dtype=np.float64)
    type_hist = np.zeros((n_clusters, len(cell_types_union)), dtype=np.float64)
    type_to_idx = {ct: i for i, ct in enumerate(cell_types_union)}

    for c in range(n_clusters):
        idx = np.where(cluster_labels == c)[0]
        if idx.size == 0:
            continue
        centroids[c] = coords[idx].mean(axis=0)
        radii[c] = float(np.median(np.linalg.norm(coords[idx] - centroids[c], axis=1))) if idx.size > 1 else 1.0
        for lab in labels[idx]:
            type_hist[c, type_to_idx[lab]] += 1.0

    type_hist = np.vstack([_safe_normalize_vector(row) for row in type_hist])
    mean_embedding = _cluster_mean_rows(embedding, cluster_labels, n_clusters)
    mean_neighborhood = _cluster_mean_rows(neighborhood_features, cluster_labels, n_clusters)

    density_stats = np.zeros((n_clusters, 2), dtype=np.float64)
    for c in range(n_clusters):
        idx = np.where(cluster_labels == c)[0]
        if idx.size:
            density_stats[c, 0] = density[idx].mean()
            density_stats[c, 1] = density[idx].std() if idx.size > 1 else 0.0

    centroid_dist = euclidean_distances(centroids, centroids)
    centroid_dist = _normalize_cost_matrix(centroid_dist)

    adjacency = np.full((n_clusters, n_clusters), np.inf, dtype=np.float64)
    np.fill_diagonal(adjacency, 0.0)
    for i, j, d in cell_graph_edges:
        ci = int(cluster_labels[i])
        cj = int(cluster_labels[j])
        if ci == cj:
            continue
        centroid_edge = float(np.linalg.norm(centroids[ci] - centroids[cj]))
        edge_cost = min(max(float(d), 1e-8), centroid_edge if centroid_edge > 0 else float(d))
        if edge_cost < adjacency[ci, cj]:
            adjacency[ci, cj] = edge_cost
            adjacency[cj, ci] = edge_cost

    graph_dist = adjacency.copy()
    for k in range(n_clusters):
        graph_dist = np.minimum(graph_dist, graph_dist[:, [k]] + graph_dist[[k], :])
    inf_mask = ~np.isfinite(graph_dist)
    graph_dist[inf_mask] = centroid_dist[inf_mask]
    graph_dist = _normalize_cost_matrix(graph_dist)

    structure = graph_structure_weight * graph_dist + (1.0 - graph_structure_weight) * centroid_dist
    structure = _normalize_cost_matrix(structure)

    return {
        'sizes': cluster_sizes,
        'centroids': centroids,
        'radii': radii,
        'type_hist': type_hist,
        'mean_embedding': mean_embedding,
        'mean_neighborhood': np.vstack([_safe_normalize_vector(row + 1e-8) for row in mean_neighborhood]),
        'density_stats': density_stats,
        'structure': structure,
        'adjacency': adjacency,
        'cluster_labels': cluster_labels,
    }


def _compute_cluster_feature_cost(
    stats_A: Dict[str, Any],
    stats_B: Dict[str, Any],
    beta: float,
    gamma: float,
    density_weight: float = 0.2,
    eps: float = 1e-8,
) -> np.ndarray:
    type_cost = _pairwise_js_distance(stats_A['type_hist'] + eps, stats_B['type_hist'] + eps)
    expr_cost = cosine_distances(stats_A['mean_embedding'] + eps, stats_B['mean_embedding'] + eps)
    nbr_cost = _pairwise_js_distance(stats_A['mean_neighborhood'] + eps, stats_B['mean_neighborhood'] + eps)
    density_cost = euclidean_distances(stats_A['density_stats'], stats_B['density_stats'])

    type_cost = _normalize_cost_matrix(type_cost)
    expr_cost = _normalize_cost_matrix(expr_cost)
    nbr_cost = _normalize_cost_matrix(nbr_cost)
    density_cost = _normalize_cost_matrix(density_cost)

    linear_cost = (1.0 - beta) * expr_cost + beta * type_cost
    linear_cost += gamma * nbr_cost + density_weight * density_cost
    return _normalize_cost_matrix(linear_cost)


def _compute_numpy_cell_costs(
    sliceA: AnnData,
    sliceB: AnnData,
    beta: float,
    gamma: float,
    radius: Optional[float] = None,
    use_rep: Optional[str] = None,
    epsilon: float = 1e-6,
    spatial_key: str = 'spatial',
    label_key: str = 'cell_type_annot',
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:    
    use_gpu, nx = select_backend(use_gpu=use_gpu, gpu_verbose=False)
    
    D_A, D_B = calculate_spatial_distance(sliceA, sliceB, nx, data_type=np.float32, eps=epsilon)
    
    scale = max(estimate_characteristic_spacing(sliceA, spatial_key=spatial_key), estimate_characteristic_spacing(sliceB, spatial_key=spatial_key), epsilon)
    D_A = D_A / scale
    D_B = D_B / scale

    expr_cost = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, eps=epsilon)
    type_cost = calculate_cell_type_mismatch(sliceA, sliceB)

    # Use GPU for neighborhood dissimilarity if requested
    nbr_cost = calculate_neighborhood_dissimilarity(
        sliceA,
        sliceB,
        radius=radius,
        nx=nx,
        data_type=np.float32,
        eps=epsilon,
        radii=None,
        radius_k=6,
        radius_multipliers=(2.5, 4.0, 6.0),
        n_shells=3,
        harmonics=(0, 1, 2),
        harmonic_weights={1: 1.25, 2: 1.5},
        distance_decay='linear',
        sigma=None,
        include_self=False,
        spatial_key=spatial_key,
        label_key=label_key,
    )
    
    if hasattr(nbr_cost, 'cpu'):
        nbr_cost = nx.to_numpy(nbr_cost)
    elif hasattr(nbr_cost, 'numpy'):
        nbr_cost = nbr_cost.numpy()

    M = (1.0 - beta) * _normalize_cost_matrix(expr_cost) + beta * _normalize_cost_matrix(type_cost)
    M += gamma * _normalize_cost_matrix(nbr_cost)
    M = _normalize_cost_matrix(M)
    
    # Offload back to CPU to prevent massive VRAM leaks inside hierarchical details
    if hasattr(D_A, 'cpu'):
        D_A = nx.to_numpy(D_A)
    if hasattr(D_B, 'cpu'):
        D_B = nx.to_numpy(D_B)
        
    return D_A, D_B, M



def _sinkhorn_project_kernel(K: np.ndarray, a: np.ndarray, b: np.ndarray, n_iter: int = 50, eps: float = 1e-12, use_gpu: bool = False):
    use_gpu, nx = select_backend(use_gpu=use_gpu, gpu_verbose=False)

    K = to_backend(np.maximum(np.asarray(K, dtype=np.float64), eps), nx)
    a = to_backend(_safe_normalize_vector(a, eps=eps), nx)
    b = to_backend(_safe_normalize_vector(b, eps=eps), nx)
    u = nx.ones(a.shape, type_as=a)
    v = nx.ones(b.shape, type_as=b)
    for _ in range(n_iter):
        Kv = nx.dot(K, v)
        Kv = nx.maximum(Kv, eps)
        u = a / Kv
        KTu = nx.dot(K.T, u)
        KTu = nx.maximum(KTu, eps)
        v = b / KTu
    
    res = (u[:, None] * K) * v[None, :]
    return res



def _build_cell_level_init_from_cluster_plan(
    coarse_plan: np.ndarray,
    cluster_labels_A: np.ndarray,
    cluster_labels_B: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    cluster_centroids_A: np.ndarray,
    cluster_centroids_B: np.ndarray,
    cluster_radii_A: np.ndarray,
    cluster_radii_B: np.ndarray,
    feature_cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    topk_clusters: int = 2,
    min_mass_fraction: float = 0.05,
    spatial_sigma_scale: float = 2.0,
    feature_tau: float = 0.5,
    eps: float = 1e-12,
    **kwargs,
) -> np.ndarray:
    nA, nB = feature_cost.shape
    G = np.zeros((nA, nB), dtype=np.float64)

    unique_A = np.unique(cluster_labels_A)
    for u in unique_A:
        src_idx = np.where(cluster_labels_A == u)[0]
        row = np.asarray(coarse_plan[u], dtype=np.float64)
        if row.sum() <= eps:
            row = np.zeros_like(row)
            row[np.argmin(np.linalg.norm(cluster_centroids_B - cluster_centroids_A[u], axis=1))] = 1.0
        vmax = row.max() if row.size else 0.0
        active = np.where(row >= min_mass_fraction * max(vmax, eps))[0]
        if active.size == 0:
            active = np.argsort(row)[-topk_clusters:]
        elif active.size > topk_clusters:
            active = active[np.argsort(row[active])[-topk_clusters:]]
        active = np.asarray(active, dtype=np.int32)
        active_weights = _safe_normalize_vector(row[active] + eps, eps=eps)

        src_rel = coords_A[src_idx] - cluster_centroids_A[u]
        for weight_uv, v in zip(active_weights, active):
            tgt_idx = np.where(cluster_labels_B == v)[0]
            if tgt_idx.size == 0:
                continue
            pred = src_rel + cluster_centroids_B[v]
            d2 = cdist(pred, coords_B[tgt_idx], metric='sqeuclidean')
            sigma = spatial_sigma_scale * max(cluster_radii_A[u], cluster_radii_B[v], 1e-3)
            spatial_kernel = np.exp(-d2 / (2.0 * sigma * sigma))
            feat_kernel = np.exp(-feature_cost[np.ix_(src_idx, tgt_idx)] / max(feature_tau, 1e-6))
            G[np.ix_(src_idx, tgt_idx)] += weight_uv * spatial_kernel * feat_kernel

    fallback = np.exp(-feature_cost / max(feature_tau, 1e-6))
    row_zero = np.where(G.sum(axis=1) <= eps)[0]
    if row_zero.size:
        G[row_zero] = fallback[row_zero]
    col_zero = np.where(G.sum(axis=0) <= eps)[0]
    if col_zero.size:
        G[:, col_zero] += fallback[:, col_zero]

    return _sinkhorn_project_kernel(G + eps, a, b, n_iter=75, eps=eps, use_gpu=kwargs.get('use_gpu', False))



def hierarchical_pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: Optional[float] = None,
    use_rep: Optional[str] = None,
    target_cluster_size: int = 96,
    n_clusters: Optional[int] = None,
    clustering_k_neighbors: int = 12,
    clustering_feature_weight: float = 0.25,
    graph_structure_weight: float = 0.65,
    coarse_density_weight: float = 0.20,
    coarse_alpha: float = 0.6,
    coarse_max_iter: int = 1000,
    coarse_max_iter_ot: int = 2000,
    fine_alpha: Optional[float] = None,
    fine_max_iter: int = 100,
    fine_max_iter_ot: int = 300,
    init_topk_clusters: int = 2,
    init_min_mass_fraction: float = 0.05,
    init_spatial_sigma_scale: float = 2.0,
    init_feature_tau: float = 0.5,
    spatial_key: str = 'spatial',
    label_key: str = 'cell_type_annot',
    a_distribution = None,
    b_distribution = None,
    unbalanced: bool = True,
    verbose: bool = False,
    return_details: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], Dict[str, Any]]]:
    """
    Hierarchical coarse-to-fine alignment using contiguous supercells and unbalanced FGW
    at both the cluster and cell levels.

    The coarse level builds balanced-ish spatial supercells, computes a cluster graph,
    solves fused unbalanced Gromov-Wasserstein, and then uses the coarse transport plan
    to construct a cell-level initialization for a second unbalanced FGW solve.
    """

    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f'Found empty `AnnData`:\n{s}.')
        if spatial_key not in s.obsm:
            raise KeyError(f'Missing spatial coordinates in `obsm[{spatial_key!r}]`.')
        if label_key not in s.obs:
            raise KeyError(f'Missing cell type labels in `obs[{label_key!r}]`.')

    nA, nB = sliceA.n_obs, sliceB.n_obs
    if n_clusters is None:
        n_clusters = int(np.ceil(max(nA, nB) / max(target_cluster_size, 2)))
    n_clusters = int(max(1, min(n_clusters, nA, nB)))

    if fine_alpha is None:
        fine_alpha = alpha

    if verbose:
        print(f'[hierarchical] n_clusters={n_clusters}, target_cluster_size≈{int(np.ceil(max(nA, nB) / n_clusters))}')

    super_A = _build_supercell_features(sliceA, use_rep=use_rep, spatial_key=spatial_key, label_key=label_key)
    super_B = _build_supercell_features(sliceB, use_rep=use_rep, spatial_key=spatial_key, label_key=label_key)
    nbrs_A, edgew_A, edges_A = _build_spatial_knn_graph(super_A['coords'], k_neighbors=clustering_k_neighbors)
    nbrs_B, edgew_B, edges_B = _build_spatial_knn_graph(super_B['coords'], k_neighbors=clustering_k_neighbors)

    cluster_labels_A = _balanced_region_growing_labels(
        super_A['coords'], super_A['feature_matrix'], nbrs_A, edgew_A, n_clusters=n_clusters, feature_weight=clustering_feature_weight
    )
    cluster_labels_B = _balanced_region_growing_labels(
        super_B['coords'], super_B['feature_matrix'], nbrs_B, edgew_B, n_clusters=n_clusters, feature_weight=clustering_feature_weight
    )

    union_cell_types = np.array(sorted(set(super_A['labels']) | set(super_B['labels'])), dtype=str)

    radii = [float(radius)] if radius is not None else default_radii_from_spacing(sliceA, sliceB, k=6, multipliers=(2.5, 4.0, 6.0), spatial_key=spatial_key)
    neighborhood_A = neighborhood_distribution_multiscale(
        sliceA, radii=radii, cell_types=union_cell_types, n_shells=3, harmonics=(0, 1, 2),
        harmonic_weights={1: 1.25, 2: 1.5}, distance_decay='linear', include_self=False,
        area_normalize=True, add_empty_bin_per_scale=False, l1_normalize_within_scale=True,
        final_l1_normalize=True, dtype=np.float32, spatial_key=spatial_key, label_key=label_key,
    )
    neighborhood_B = neighborhood_distribution_multiscale(
        sliceB, radii=radii, cell_types=union_cell_types, n_shells=3, harmonics=(0, 1, 2),
        harmonic_weights={1: 1.25, 2: 1.5}, distance_decay='linear', include_self=False,
        area_normalize=True, add_empty_bin_per_scale=False, l1_normalize_within_scale=True,
        final_l1_normalize=True, dtype=np.float32, spatial_key=spatial_key, label_key=label_key,
    )

    stats_A = _compute_cluster_statistics(
        sliceA, cluster_labels_A, super_A['coords'], edges_A, super_A['density'], neighborhood_A,
        cell_types_union=union_cell_types, use_rep=use_rep, spatial_key=spatial_key, label_key=label_key,
        graph_structure_weight=graph_structure_weight,
    )
    stats_B = _compute_cluster_statistics(
        sliceB, cluster_labels_B, super_B['coords'], edges_B, super_B['density'], neighborhood_B,
        cell_types_union=union_cell_types, use_rep=use_rep, spatial_key=spatial_key, label_key=label_key,
        graph_structure_weight=graph_structure_weight,
    )

    M_coarse = _compute_cluster_feature_cost(stats_A, stats_B, beta=beta, gamma=gamma, density_weight=coarse_density_weight)
    p = _safe_normalize_vector(stats_A['sizes'])
    q = _safe_normalize_vector(stats_B['sizes'])
    init_coarse = np.outer(p, q)

    if unbalanced:
        # Check deficit relative to the other slice to add dummy
        _has_dummy_src = False
        _has_dummy_tgt = False
        
        _w_dummy_src = max(0, stats_B['sizes'].sum() - stats_A['sizes'].sum())
        _w_dummy_tgt = max(0, stats_A['sizes'].sum() - stats_B['sizes'].sum())
        
        _has_dummy_src = _w_dummy_src > 0
        _has_dummy_tgt = _w_dummy_tgt > 0
        
        _budget = max(stats_A['sizes'].sum(), stats_B['sizes'].sum())

        ns_c, nt_c = len(p), len(q)
        _ns_c_aug = ns_c + (1 if _has_dummy_src else 0)
        _nt_c_aug = nt_c + (1 if _has_dummy_tgt else 0)

        # Augment structures
        struct_A = stats_A['structure'].copy()
        struct_B = stats_B['structure'].copy()
        
        if _has_dummy_src:
            zeros_col = np.zeros((ns_c, 1), dtype=np.float64)
            struct_A = np.concatenate([struct_A, zeros_col], axis=1)
            zeros_row = np.zeros((1, _ns_c_aug), dtype=np.float64)
            struct_A = np.concatenate([struct_A, zeros_row], axis=0)
            
        if _has_dummy_tgt:
            zeros_col = np.zeros((nt_c, 1), dtype=np.float64)
            struct_B = np.concatenate([struct_B, zeros_col], axis=1)
            zeros_row = np.zeros((1, _nt_c_aug), dtype=np.float64)
            struct_B = np.concatenate([struct_B, zeros_row], axis=0)

        # Augment feature cost M_coarse
        if _has_dummy_tgt:
            mean_col = M_coarse.mean(axis=1, keepdims=True)
            M_coarse = np.concatenate([M_coarse, mean_col], axis=1)
        if _has_dummy_src:
            mean_row = M_coarse.mean(axis=0, keepdims=True)
            M_coarse = np.concatenate([M_coarse, mean_row], axis=0)
        if _has_dummy_src and _has_dummy_tgt:
            M_coarse[-1, -1] = 0.0

        # Adjust p and q distributions
        if _has_dummy_src:
            p = np.concatenate([stats_A['sizes'] / _budget, [_w_dummy_src / _budget]])
        else:
            p = stats_A['sizes'] / _budget
            
        if _has_dummy_tgt:
            q = np.concatenate([stats_B['sizes'] / _budget, [_w_dummy_tgt / _budget]])
        else:
            q = stats_B['sizes'] / _budget

        init_coarse = np.outer(p, q)
        
        coarse_out = fused_gromov_wasserstein_incent(
            M=M_coarse,
            C1=struct_A,
            C2=struct_B,
            p=p,
            q=q,
            G_init=init_coarse,
            alpha=coarse_alpha,
            numItermax=coarse_max_iter,
            numItermaxEmd=coarse_max_iter_ot,
            log=return_details,
            verbose=verbose,
            **kwargs,
        )

        if return_details:
            coarse_plan_full, coarse_log = coarse_out
        else:
            coarse_plan_full = coarse_out
            coarse_log = None
            
        # Strip dummy rows/cols added
        if _has_dummy_src and _has_dummy_tgt:
            coarse_plan = coarse_plan_full[:ns_c, :nt_c]
        elif _has_dummy_src:
            coarse_plan = coarse_plan_full[:ns_c, :]
        elif _has_dummy_tgt:
            coarse_plan = coarse_plan_full[:, :nt_c]
        else:
            coarse_plan = coarse_plan_full
            
    else:
        coarse_out = fused_gromov_wasserstein_incent(
            M=M_coarse,
            C1=stats_A['structure'],
            C2=stats_B['structure'],
            p=p,
            q=q,
            G_init=init_coarse,
            alpha=coarse_alpha,
            numItermax=coarse_max_iter,
            numItermaxEmd=coarse_max_iter_ot,
            log=return_details,
            verbose=verbose,
            **kwargs,
        )

        if return_details:
            coarse_plan, coarse_log = coarse_out
        else:
            coarse_plan = coarse_out
            coarse_log = None

    coarse_plan = np.asarray(coarse_plan, dtype=np.float64)

    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    D_A, D_B, M_fine = _compute_numpy_cell_costs(
        sliceA, sliceB, beta=beta, gamma=gamma, radius=radius, use_rep=use_rep, epsilon=1e-6,
        spatial_key=spatial_key, label_key=label_key, use_gpu=use_gpu
    )

    if a_distribution is None:
        a = np.full(nA, 1.0 / max(nA, 1), dtype=np.float64)
    else:
        a = _safe_normalize_vector(np.asarray(a_distribution, dtype=np.float64))
    if b_distribution is None:
        b = np.full(nB, 1.0 / max(nB, 1), dtype=np.float64)
    else:
        b = _safe_normalize_vector(np.asarray(b_distribution, dtype=np.float64))

    G_init = _build_cell_level_init_from_cluster_plan(
        coarse_plan=coarse_plan,
        cluster_labels_A=cluster_labels_A,
        cluster_labels_B=cluster_labels_B,
        coords_A=super_A['coords'],
        coords_B=super_B['coords'],
        cluster_centroids_A=stats_A['centroids'],
        cluster_centroids_B=stats_B['centroids'],
        cluster_radii_A=np.maximum(stats_A['radii'], 1e-3),
        cluster_radii_B=np.maximum(stats_B['radii'], 1e-3),
        feature_cost=M_fine,
        a=a,
        b=b,
        topk_clusters=init_topk_clusters,
        min_mass_fraction=init_min_mass_fraction,
        spatial_sigma_scale=init_spatial_sigma_scale,
        feature_tau=init_feature_tau,
        use_gpu=use_gpu
    )

    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    fine_out = pairwise_align(
        sliceA,
        sliceB,
        alpha=fine_alpha,
        beta=beta,
        gamma=gamma,
        radius=radius,
        use_rep=use_rep,
        G_init=G_init,
        a_distribution=a_distribution,
        b_distribution=b_distribution,
        unbalanced=unbalanced,
        verbose=verbose,
        numItermax=fine_max_iter,
        numItermaxEmd=fine_max_iter_ot,
        use_gpu=use_gpu,
        **kwargs
    )

    fine_plan = fine_out

    fine_plan = np.asarray(fine_plan, dtype=np.float64)

    details = {
        'cluster_labels_A': cluster_labels_A,
        'cluster_labels_B': cluster_labels_B,
        'coarse_plan': coarse_plan,
        'cluster_feature_cost': M_coarse,
        'cluster_structure_A': stats_A['structure'],
        'cluster_structure_B': stats_B['structure'],
        'cell_level_init': G_init,
        'cell_feature_cost': M_fine,
        'cell_structure_A': D_A,
        'cell_structure_B': D_B,
        'coarse_log': coarse_log
    }

    return (fine_plan, details) if return_details else fine_plan
