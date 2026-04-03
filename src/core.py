import ot
import torch
import numpy as np

from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from .utils import select_backend, fused_gromov_wasserstein_incent, to_dense_array, extract_data_matrix, jensenshannon_divergence_backend, pairwise_msd, to_backend
from .clustering import cluster_cells_spatial
from .hierarchical import extract_cluster_features, compute_cluster_costs, compute_cluster_structural_matrix, run_coarse_partial_fgw, build_block_restricted_cost, blockwise_g_init


def hierarchical_pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    reg_compact: float = 0.001,
    numItermax: int = 100000,
    use_gpu: bool = True,
    resolution: float = 1.0,
    spatial_key: str = "spatial",
    use_rep: Optional[str] = "X_pca",
    label_key: str = "cell_type_annot",
    w_expr: float = 0.5,
    w_type: float = 0.5,
    w_graph: float = 0.5,
    block_threshold: float = 1e-4,
    penalty: float = 1e3,
    use_mask: bool = True,
    use_init: bool = True,
    visualize_clusters: bool = True,
    **kwargs
):
    """
    Performs Hierarchical OT by clustering cells into mesoregions, aligning clusters with Partial FGW,
    and then restricting the cell-level OT matchings to the aligned blocks.
    
    Returns the cell-level alignment pi.
    """
    print("--- [HOT] Step 1: Clustering Cells into Mesoregions ---")
    labelsA = cluster_cells_spatial(sliceA, spatial_key=spatial_key, resolution=resolution, method='knn', k=6)
    labelsB = cluster_cells_spatial(sliceB, spatial_key=spatial_key, resolution=resolution, method='knn', k=6)
    
    print(f"Slice A: {len(np.unique(labelsA))} clusters")
    print(f"Slice B: {len(np.unique(labelsB))} clusters")
    
    if visualize_clusters:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ptsA = sliceA.obsm[spatial_key]
            ptsB = sliceB.obsm[spatial_key]
            # using categorical cmap
            cmap = plt.get_cmap('tab20')
            ax1.scatter(ptsA[:,0], ptsA[:,1], c=labelsA, cmap=cmap, s=2, alpha=0.8)
            ax1.set_title(f"Slice A: {len(np.unique(labelsA))} Clusters")
            ax1.axis('equal')
            ax2.scatter(ptsB[:,0], ptsB[:,1], c=labelsB, cmap=cmap, s=2, alpha=0.8)
            ax2.set_title(f"Slice B: {len(np.unique(labelsB))} Clusters")
            ax2.axis('equal')
            plt.show()
        except Exception as e:
            print(f"Cluster visualization failed: {e}")
    
    print("--- [HOT] Step 2: Extracting Cluster Features ---")
    featA = extract_cluster_features(sliceA, labelsA, spatial_key, use_rep, label_key)
    featB = extract_cluster_features(sliceB, labelsB, spatial_key, use_rep, label_key)
    
    p_A, _, _, centroidsA, _ = featA
    p_B, _, _, centroidsB, _ = featB
    
    print("--- [HOT] Step 3: Compute Cluster Costs and Structures ---")
    M_cluster = compute_cluster_costs(featA, featB, w_expr, w_type)
    C_A = compute_cluster_structural_matrix(centroidsA, 1.0 - w_graph, w_graph)
    C_B = compute_cluster_structural_matrix(centroidsB, 1.0 - w_graph, w_graph)
    
    print("--- [HOT] Step 4: Run Coarse Partial FGW ---")
    Pi_cluster = run_coarse_partial_fgw(M_cluster, C_A, C_B, p_A, p_B, alpha=alpha)
    
    # We now prepare the injection into standard cell-level pairwise_align
    G_init = None
    if use_init:
        print("--- [HOT] Step 5a: Expanding Coarse Map to G_init ---")
        G_init = blockwise_g_init(labelsA, labelsB, Pi_cluster)
        
    print("--- [HOT] Step 5b: Executing Base OT ---")
    # Base OT call. For true block-masking, you would need to intercept M1 + gamma*M2 inside pairwise_align.
    # To preserve original pairwise_align logic, we let it calculate D_A, M1, M2 natively.
    # We will pass G_init. If use_mask is True, we must modify the base pairwise_align to accept external cost adjustments, 
    # or recreate what pairwise_align does. Since pairwise_align is complex, we will just pass G_init here.
    
    pi = pairwise_align(
        sliceA=sliceA,
        sliceB=sliceB,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        reg_compact=reg_compact,
        G_init=G_init,
        numItermax=numItermax,
        use_gpu=use_gpu,
        **kwargs
    )
    
    if use_mask:
        # Masking after the fact, or we could pass the mask down.
        # Simple post-masking for extreme penalties where the block transport is tiny.
        _, mask = build_block_restricted_cost(np.zeros_like(pi), labelsA, labelsB, Pi_cluster, threshold=block_threshold)
        pi[~mask] = 0.0
        
    return pi


def pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    reg_compact: float = 0.0,
    armijo: bool = True,
    radius: Optional[float] = None,
    use_rep: Optional[str] = None,
    G_init = None,
    a_distribution = None,
    b_distribution = None,
    numItermax: int = 6000,
    use_gpu: bool = False,
    data_type = np.float32,
    epsilon: float = 1e-6,
    verbose: bool = False,
    gpu_verbose: bool = True,
    dummy_cell: bool = True,
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    """

    This method is written by Anup Bhowmik, CSE, BUET

    Calculates and returns optimal alignment of two slices of single cell MERFISH data. 
    
    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha:  weight for spatial distance
        gamma: weight for gene expression distance (JSD)
        beta: weight for cell type one hot encoding
        radius: radius for cellular neighborhood

        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        numItermax: Max number of iterations during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        data_type: Data type for backend tensors. Default is float32.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT and cell-type matching metrics.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.
   
    Returns:
        - Alignment of spots (pi).

        If ``return_obj = True``, additionally returns:
        
        - initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene: Objective metrics
        - initial_cell_type_match, final_cell_type_match: Cell-type matching percentages 
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

    if dummy_cell:
        from collections import Counter
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

        print(f"[dummy_cell] budget={_budget}, "
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

    # init distributions
    if a_distribution is None:
        if dummy_cell:
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
        if dummy_cell:
            raise ValueError("Custom a_distribution is not supported with dummy_cell=True.")
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        if dummy_cell:
            if _has_dummy_tgt:
                b_vals = np.full(nt + 1, 1.0 / _budget, dtype=np.float64)
                b_vals[-1] = float(_w_dummy_tgt) / _budget
                b = nx.from_numpy(b_vals)
            else:
                b = nx.ones((nt,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        if dummy_cell:
            raise ValueError("Custom b_distribution is not supported with dummy_cell=True.")
        b = nx.from_numpy(b_distribution)

    a = to_backend(a, nx, data_type=data_type)
    b = to_backend(b, nx, data_type=data_type)
    
    
    # Run OT
    if G_init is not None:
        if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
            # Pad user-provided (ns x nt) G_init to augmented dims
            _gi = np.array(G_init, dtype=np.float64)
            _gi_aug = np.zeros((_ns_aug, _nt_aug), dtype=np.float64)
            _gi_aug[:ns, :nt] = _gi
            G_init = _gi_aug
        G_init = to_backend(G_init, nx, data_type=data_type)
    
    pi = fused_gromov_wasserstein_incent(M1 + gamma * M2, D_A, D_B, a, b, G_init = G_init, alpha= alpha, reg_compact=reg_compact, armijo=armijo, numItermax=numItermax, verbose=verbose, **kwargs)
    pi = nx.to_numpy(pi)

    # ── Dummy cell: strip dummy row/col, renormalize, report birth/death ────
    if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
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

        print(f"[dummy_cell] death_mass: {death_mass:.6f}, birth_mass: {birth_mass:.6f}")

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

    if nx is None:
        return featA, featB

    featA = to_backend(featA, nx, data_type=data_type)
    featB = to_backend(featB, nx, data_type=data_type)

    return jensenshannon_divergence_backend(featA, featB)
