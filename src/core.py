import ot
import torch
import numpy as np

from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from .utils import select_backend, fused_gromov_wasserstein_incent, to_dense_array, extract_data_matrix, jensenshannon_divergence_backend, to_backend
from .clustering import cluster_cells_spatial
from .hierarchical import extract_cluster_features, compute_cluster_costs, compute_cluster_structural_matrix, run_coarse_partial_fgw, build_block_restricted_cost, blockwise_g_init, extract_continuous_macro_section


def hierarchical_pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    reg_compact: float = 0.001,
    numItermax: int = 100000,
    use_gpu: bool = True,
    cluster_method: str = 'delaunay',
    cluster_extension_hops: int = 5,
    resolution: float = 1.0,
    macro_section_mass_pct: float = 0.8,
    spatial_key: str = "spatial",
    use_rep: Optional[str] = "X_pca",
    label_key: str = "cell_type_annot",
    w_expr: float = 0.4,
    w_type: float = 0.4,
    w_struct: float = 0.2,
    w_graph: float = 0.5,
    block_threshold: float = 1e-4,
    rand_seed: Optional[int] = 2005021,
    visualize_clusters: bool = True,
    **kwargs
):
    """
    Performs Hierarchical OT by clustering cells into mesoregions, aligning clusters with Partial FGW,
    and then restricting the cell-level OT matchings to the aligned blocks.
    
    Returns the cell-level alignment pi.
    """
    print("--- [HOT] Step 1: Clustering Cells into Mesoregions ---")
    labelsA = cluster_cells_spatial(sliceA, spatial_key=spatial_key, resolution=resolution, method=cluster_method, k=6, seed=rand_seed)
    labelsB = cluster_cells_spatial(sliceB, spatial_key=spatial_key, resolution=resolution, method=cluster_method, k=6, seed=rand_seed)
    
    # Pre-cache global cell types for cluster structure alignment
    all_types = np.array(sorted(set(sliceA.obs[label_key].astype(str)) | set(sliceB.obs[label_key].astype(str))), dtype=str)

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
    featA = extract_cluster_features(sliceA, labelsA, spatial_key, use_rep, label_key, all_types=all_types)
    featB = extract_cluster_features(sliceB, labelsB, spatial_key, use_rep, label_key, all_types=all_types)
    
    p_A, _, _, centroidsA, _, _ = featA
    p_B, _, _, centroidsB, _, _ = featB
    
    print("--- [HOT] Step 3: Compute Cluster Costs and Structures ---")
    # Note: If w_struct is used, w_type is inherently redundant because the 0th harmonic of the cluster 
    # structural feature is exactly its cell type composition. User can set w_type=0.0 when w_struct>0.
    M_cluster = compute_cluster_costs(featA, featB, w_expr, w_type, w_struct)
    C_A = compute_cluster_structural_matrix(centroidsA, 1.0 - w_graph, w_graph)
    C_B = compute_cluster_structural_matrix(centroidsB, 1.0 - w_graph, w_graph)
    
    print("--- [HOT] Step 4: Run Coarse Partial FGW ---")
    Pi_cluster = run_coarse_partial_fgw(M_cluster, C_A, C_B, p_A, p_B, alpha=alpha)
    
    if visualize_clusters:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            
            # Center the coordinates purely for overlap plotting
            cA_plot = centroidsA - np.mean(centroidsA, axis=0)
            cB_plot = centroidsB - np.mean(centroidsB, axis=0)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(cA_plot[:,0], cA_plot[:,1], c='blue', s=20, label='Slice A Clusters (Centered)', zorder=2)
            ax.scatter(cB_plot[:,0], cB_plot[:,1], c='red', s=20, label='Slice B Clusters (Centered)', zorder=2)

            max_pi = np.max(Pi_cluster)
            if max_pi > 0:
                lines = []
                linewidths = []
                for i in range(Pi_cluster.shape[0]):
                    for j in range(Pi_cluster.shape[1]):
                        if Pi_cluster[i, j] > block_threshold:
                            lines.append([(cA_plot[i,0], cA_plot[i,1]), (cB_plot[j,0], cB_plot[j,1])])
                            linewidths.append((Pi_cluster[i, j] / max_pi) * 2.0)      

                lc = LineCollection(lines, colors='k', linewidths=linewidths, alpha=0.5, zorder=1)
                ax.add_collection(lc)

            ax.set_title("Macro-Level Cluster Matching ($Pi_{cluster}$)")
            ax.axis('equal')
            ax.legend()
            plt.show()
        except Exception as e:
            print(f"Cluster matching visualization failed: {e}")

    # We now prepare the injection into standard cell-level pairwise_align
    print("--- [HOT] Step 5: Extract Continuous Macro Sections ---")
    idx_A, idx_B, dist_A, dist_B = extract_continuous_macro_section(
        sliceA, sliceB, labelsA, labelsB, Pi_cluster, mass_pct=macro_section_mass_pct,
        spatial_key=spatial_key, extension_hops=cluster_extension_hops
    )
    
    print(f"Selected {len(idx_A)}/{sliceA.shape[0]} cells from A, {len(idx_B)}/{sliceB.shape[0]} cells from B.")

    if visualize_clusters:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ptsA = sliceA.obsm[spatial_key]
            ptsB = sliceB.obsm[spatial_key]
            
            # Highlight chosen core dynamically
            core_A_mask = np.zeros(sliceA.shape[0], dtype=bool)
            core_A_mask[idx_A] = True
            core_B_mask = np.zeros(sliceB.shape[0], dtype=bool)
            core_B_mask[idx_B] = True
            
            # Plot distance out from core (distance is 0 inside core)
            sc1 = ax1.scatter(ptsA[:,0], ptsA[:,1], c=dist_A, cmap='viridis_r', s=4, alpha=0.9)
            ax1.scatter(ptsA[core_A_mask,0], ptsA[core_A_mask,1], c='red', s=2, alpha=0.5, label='Selected Core')
            ax1.set_title("Slice A: Macro Selection & Distances")
            ax1.axis('equal')
            ax1.legend()
            fig.colorbar(sc1, ax=ax1, label='Distance to Core', fraction=0.046, pad=0.04)
            
            sc2 = ax2.scatter(ptsB[:,0], ptsB[:,1], c=dist_B, cmap='viridis_r', s=4, alpha=0.9)
            ax2.scatter(ptsB[core_B_mask,0], ptsB[core_B_mask,1], c='red', s=2, alpha=0.5, label='Selected Core')
            ax2.set_title("Slice B: Macro Selection & Distances")
            ax2.axis('equal')
            ax2.legend()
            fig.colorbar(sc2, ax=ax2, label='Distance to Core', fraction=0.046, pad=0.04)
            
            plt.show()
        except Exception as e:
            print(f"Sub-selection visualization failed: {e}")

    # Only run base OT on the selected continuous matching blocks
    sub_sliceA = sliceA[idx_A].copy()
    sub_sliceB = sliceB[idx_B].copy()

    # The selected sections will have an initial plan that decays based on distance from the core
    sub_N, sub_M = sub_sliceA.shape[0], sub_sliceB.shape[0]
    G_init_sub = None
    if sub_N > 0 and sub_M > 0:
        sigma_A = max(1e-5, np.max(dist_A) / 2.0)
        sigma_B = max(1e-5, np.max(dist_B) / 2.0)
        
        weight_A = np.exp(- (dist_A[idx_A]**2) / (2 * sigma_A**2))
        weight_B = np.exp(- (dist_B[idx_B]**2) / (2 * sigma_B**2))
        
        G_init_sub = np.outer(weight_A, weight_B)
        G_init_sub /= np.sum(G_init_sub)

    print("--- [HOT] Step 6: Executing Base OT on Selected Sections ---")
    pi_sub = pairwise_align(
        sliceA=sub_sliceA,
        sliceB=sub_sliceB,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        reg_compact=reg_compact,
        G_init=G_init_sub,
        numItermax=numItermax,
        use_gpu=use_gpu,
        unbalanced=True,
        **kwargs
    )

    # Rest of the region will have zero initial/final plan
    pi_full = np.zeros((sliceA.shape[0], sliceB.shape[0]))
    if sub_N > 0 and sub_M > 0:
        grid_A, grid_B = np.ix_(idx_A, idx_B)
        pi_full[grid_A, grid_B] = pi_sub

    print("--- [HOT] Step 7: Global Refinement via Overlap Projection ---")
    from .visualize import stack_slices_pairwise
    try:
        # 1. Geometrically align full slices using the partial block solution
        aligned_slices = stack_slices_pairwise([sliceA, sliceB], [pi_full], output_params=False)
        coords_A_aligned = np.asarray(aligned_slices[0].obsm[spatial_key])
        coords_B_aligned = np.asarray(aligned_slices[1].obsm[spatial_key])
        
        # 2. Determine overlapping regions based on Mutual Convex Hull Intersection
        from scipy.spatial import Delaunay
        
        # Build spatial triangulations which implicitly model the arbitrary Convex Hull 
        # (the exact polygonal "shrink-wrap" of the tissue) rather than axis-aligned rectangles.
        tri_A = Delaunay(coords_A_aligned)
        tri_B = Delaunay(coords_B_aligned)
        
        # A cell is mathematically in the overlap if its coordinate physically 
        # falls inside the triangulated geometric boundaries of the opposite slice.
        overlap_mask_A = tri_B.find_simplex(coords_A_aligned) >= 0
        overlap_mask_B = tri_A.find_simplex(coords_B_aligned) >= 0
        
        # Slices with complex concave curves might still flag empty middle space. 
        # A safety filter checks distance to nearest neighbor to prune out "empty hull" mappings.
        tree_A = cKDTree(coords_A_aligned)
        tree_B = cKDTree(coords_B_aligned)
        s_A = estimate_characteristic_spacing(sliceA, spatial_key=spatial_key)
        s_B = estimate_characteristic_spacing(sliceB, spatial_key=spatial_key)
        
        # Max distance permitted is ~10x local spacing to allow continuity without bridging voids
        tau = max(s_A, s_B) * 10.0
        
        dist_A_to_B, _ = tree_B.query(coords_A_aligned)
        dist_B_to_A, _ = tree_A.query(coords_B_aligned)
        
        # Logical AND: Must be inside the polygon shape AND within reasonable tissue distance
        overlap_mask_A = overlap_mask_A & (dist_A_to_B <= tau)
        overlap_mask_B = overlap_mask_B & (dist_B_to_A <= tau)
        
        # Fallback if overlap vanishes perfectly
        if not np.any(overlap_mask_A): overlap_mask_A[:] = True
        if not np.any(overlap_mask_B): overlap_mask_B[:] = True
        
        # 3. Apply strict zero-weights to the unmatched (non-overlapping) parts
        # Weights are exactly 1.0 inside the solid overlap, and 0.0 in the pure tails.
        weight_A_full = np.zeros(sliceA.shape[0], dtype=np.float64)
        weight_A_full[overlap_mask_A] = 1.0
        
        weight_B_full = np.zeros(sliceB.shape[0], dtype=np.float64)
        weight_B_full[overlap_mask_B] = 1.0

        if visualize_clusters:
            try:
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                ptsA_full = sliceA.obsm[spatial_key]
                ptsB_full = sliceB.obsm[spatial_key]
                
                sc1 = ax1.scatter(ptsA_full[:,0], ptsA_full[:,1], c=weight_A_full, cmap='magma', s=2, alpha=0.9)
                ax1.set_title("Slice A: Native Space Overlap Weights")
                ax1.axis('equal')
                fig.colorbar(sc1, ax=ax1, label='Weight / Confidence')
                
                sc2 = ax2.scatter(ptsB_full[:,0], ptsB_full[:,1], c=weight_B_full, cmap='magma', s=2, alpha=0.9)
                ax2.set_title("Slice B: Native Space Overlap Weights")
                ax2.axis('equal')
                fig.colorbar(sc2, ax=ax2, label='Weight / Confidence')
                
                plt.suptitle("Global Overlap Projection Weights (Decaying from Center)")
                plt.show()
            except Exception as e:
                print(f"Overlap weight visualization failed: {e}")
            
        print(f"--- [HOT] Step 8: Executing Final Base OT on Matched Portions (A: {np.sum(overlap_mask_A)}, B: {np.sum(overlap_mask_B)}) ---")
        
        idx_A_matched = np.where(overlap_mask_A)[0]
        idx_B_matched = np.where(overlap_mask_B)[0]
        
        sliceA_matched = sliceA[idx_A_matched].copy()
        sliceB_matched = sliceB[idx_B_matched].copy()

        # In the matched subset, initial bias is uniform
        G_init_matched = np.outer(np.ones(len(idx_A_matched)), np.ones(len(idx_B_matched)))
        G_init_sum = np.sum(G_init_matched)
        if G_init_sum > 0:
            G_init_matched /= G_init_sum

        pi_matched_final = pairwise_align(
            sliceA=sliceA_matched,
            sliceB=sliceB_matched,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            reg_compact=reg_compact,
            G_init=G_init_matched,
            numItermax=numItermax,
            use_gpu=use_gpu,
            unbalanced=True,
            **kwargs
        )
        
        pi_full_final = np.zeros((sliceA.shape[0], sliceB.shape[0]), dtype=np.float64)
        grid_A, grid_B = np.ix_(idx_A_matched, idx_B_matched)
        pi_full_final[grid_A, grid_B] = pi_matched_final

        return pi_full_final
        
    except Exception as e:
        print(f"Global refinement failed: {e}. Returning block-restricted pi_full.")
        return pi_full

def pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    reg_compact: float = 0.01,
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
    unbalanced: bool = True,
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
    
    # init distributions
    if a_distribution is None:
        # uniform distribution, a = array([1/n, 1/n, ...])
        a = nx.from_numpy(np.ones(sliceA.shape[0], dtype=np.float64) / sliceA.shape[0])
    else:
        a = to_backend(a_distribution, nx, data_type=data_type)

    if b_distribution is None:
        # uniform distribution, b = array([1/m, 1/m, ...])
        b = nx.from_numpy(np.ones(sliceB.shape[0], dtype=np.float64) / sliceB.shape[0])
    else:
        b = to_backend(b_distribution, nx, data_type=data_type)

    a = to_backend(a, nx, data_type=data_type)
    b = to_backend(b, nx, data_type=data_type)
    
    
    # Run OT
    if G_init is not None:
        G_init = to_backend(G_init, nx, data_type=data_type)
    
    pi = fused_gromov_wasserstein_incent(M1 + gamma * M2, D_A, D_B, a, b, unbalanced=unbalanced, G_init = G_init, alpha= alpha, reg_compact=reg_compact, armijo=armijo, numItermax=numItermax, verbose=verbose, **kwargs)
    pi = nx.to_numpy(pi)

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
