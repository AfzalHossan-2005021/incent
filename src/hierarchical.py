import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance_matrix

import ot
import logging


def extract_cluster_features(adata, labels, spatial_key="spatial", feature_key="X", label_key="cell_type_annot"):
    """
    Extract cluster-level features for coarse mapping.
    
    Args:
        adata: AnnData object.
        labels: np.ndarray of cluster labels for each cell.
        spatial_key: key in obsm for coords.
        feature_key: 'X' for adata.X, else obsm key for latent features.
        label_key: key in obs for cell type annotations.
        
    Returns:
        masses: np.ndarray (C,) normalized size of each cluster
        mu_expr: np.ndarray (C, D) mean expression/latent vector
        hist_types: np.ndarray (C, T) normalized cell-type histograms
        centroids: np.ndarray (C, 2) average spatial coordinate
    """
    coords = adata.obsm[spatial_key]
    n_cells = coords.shape[0]
    
    if feature_key == "X":
        if sp.issparse(adata.X):
            expr = adata.X.toarray()
        else:
            expr = adata.X
    else:
        expr = adata.obsm[feature_key]
        
    ctypes = adata.obs[label_key].astype(str).values
    unique_types = np.unique(ctypes)
    type_map = {t: i for i, t in enumerate(unique_types)}
    n_types = len(unique_types)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    masses = np.zeros(n_clusters)
    mu_expr = np.zeros((n_clusters, expr.shape[1]))
    hist_types = np.zeros((n_clusters, n_types))
    centroids = np.zeros((n_clusters, 2))
    
    for c_i, c in enumerate(unique_labels):
        mask = (labels == c)
        c_size = np.sum(mask)
        masses[c_i] = c_size / float(n_cells)
        
        if c_size > 0:
            mu_expr[c_i, :] = np.mean(expr[mask], axis=0)
            centroids[c_i, :] = np.mean(coords[mask], axis=0)
            
            c_types = ctypes[mask]
            counts = np.bincount([type_map[t] for t in c_types], minlength=n_types)
            hist_types[c_i, :] = counts / float(c_size) # normalized histogram
            
    return masses, mu_expr, hist_types, centroids, unique_labels


def compute_cluster_costs(featuresA, featuresB, w_expr=0.5, w_type=0.5):
    """
    Compute inter-cluster cost matrix M_cluster between two slices.
    
    Args:
        featuresA: Tuple of (masses, mu_expr, hist_types, centroids) from slice A
        featuresB: Tuple of (masses, mu_expr, hist_types, centroids) from slice B
        
    Returns:
        M_cluster: np.ndarray (C_A, C_B) cost matrix
    """
    _, mu_expr_A, hist_types_A, _, _ = featuresA
    _, mu_expr_B, hist_types_B, _, _ = featuresB
    
    # Cosine distance for continuous expression
    from sklearn.metrics.pairwise import cosine_distances
    M_expr = cosine_distances(mu_expr_A, mu_expr_B)
    
    # Jensen-Shannon for categorical cell types (histograms are sum=1)
    from scipy.spatial.distance import jensenshannon
    M_type = np.zeros((hist_types_A.shape[0], hist_types_B.shape[0]))
    for i in range(hist_types_A.shape[0]):
        for j in range(hist_types_B.shape[0]):
            # js returns distance, bounded 0-1. Can square for JS divergence if desired.
            M_type[i, j] = jensenshannon(hist_types_A[i], hist_types_B[j])
            
    M_cluster = w_expr * M_expr + w_type * M_type
    return M_cluster


def compute_cluster_structural_matrix(centroids, w_euc=1.0, w_graph=0.0):
    """
    Compute intra-slice structure matrix C.
    
    Args:
        centroids: (C, 2) coords of clusters.
    """
    # Simple euclidean distance for now if w_graph is 0
    C_euc = distance_matrix(centroids, centroids)
    
    if w_graph > 0:
        from scipy.spatial import Delaunay
        # build adj matrix
        n = centroids.shape[0]
        adj = np.zeros((n, n))
        try:
            tri = Delaunay(centroids)
            indptr, indices = tri.vertex_neighbor_vertices
            for i in range(n):
                for j in indices[indptr[i]:indptr[i+1]]:
                    adj[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            
            C_graph = dijkstra(sp.csr_matrix(adj), directed=False)
            # handle infinite dists
            C_graph[np.isinf(C_graph)] = np.max(C_graph[~np.isinf(C_graph)]) * 2
        except Exception as e:
            logging.warning(f"Delaunay failed: {e}. Falling back to 100% euclidean.")
            C_graph = C_euc
            
        C = w_euc * C_euc + w_graph * C_graph
    else:
        C = C_euc
        
    return C


def run_coarse_partial_fgw(M_cluster, C_A, C_B, p_A, p_B, alpha=0.5, m=None, reg_m=1.0):
    """
    Solves cluster-level partial/unbalanced FGW.
    """
    if m is None:
        m = min(np.sum(p_A), np.sum(p_B)) * 0.999
        
    C_A_norm = C_A / (np.max(C_A) + 1e-8)
    C_B_norm = C_B / (np.max(C_B) + 1e-8)
    M_norm = M_cluster / (np.max(M_cluster) + 1e-8)
    
    logging.info("Running Unbalanced FGW...")
    # reg_marginals controls how much marginal relaxation is allowed (lower = more mass can be dropped)
    pi_samp, pi_feat, log = ot.gromov.fused_unbalanced_gromov_wasserstein(
        Cx=C_A_norm, Cy=C_B_norm, wx=p_A, wy=p_B, M=M_norm, alpha=alpha, reg_marginals=reg_m, log=True, max_iter=500
    )
    
    return pi_samp


def build_block_restricted_cost(M_cell, labels_A, labels_B, Pi_cluster, threshold=1e-4, penalty=1e6):
    """
    Blocks out cell pairs where their corresponding macro-regions aren't aligned.
    
    Args:
        M_cell: (N, M) original feature cost matrix for cells.
        labels_A: (N,) macro cluster assignment for A.
        labels_B: (M,) macro cluster assignment for B.
        Pi_cluster: (C_A, C_B) soft transport matrix from coarse step.
        threshold: minimum macro transport to consider the blocks alive.
        penalty: amount to add to M_cell for blocked pairs.
        
    Returns:
        M_penalty: (N, M) new cost matrix
        mask: (N, M) boolean mask of active blocks
    """
    n, m = M_cell.shape
    M_penalty = M_cell.copy()
    mask = np.zeros((n, m), dtype=bool)
    
    for i in range(Pi_cluster.shape[0]):
        for j in range(Pi_cluster.shape[1]):
            if Pi_cluster[i, j] > threshold:
                idx_A = np.where(labels_A == i)[0]
                idx_B = np.where(labels_B == j)[0]
                if len(idx_A) > 0 and len(idx_B) > 0:
                    # Mark active
                    grid_A, grid_B = np.ix_(idx_A, idx_B)
                    mask[grid_A, grid_B] = True
                    
    M_penalty[~mask] += penalty
    return M_penalty, mask


def blockwise_g_init(labels_A, labels_B, Pi_cluster):
    """
    Expands the coarse Pi_cluster into an initial transport map G_init for cells.
    
    Args:
        labels_A: (N,)
        labels_B: (M,)
        Pi_cluster: (C_A, C_B)
        
    Returns:
        G_init: (N, M)
    """
    N = len(labels_A)
    M = len(labels_B)
    G_init = np.zeros((N, M))
    
    for i in range(Pi_cluster.shape[0]):
        idx_A = np.where(labels_A == i)[0]
        nA = len(idx_A)
        if nA == 0: continue
            
        for j in range(Pi_cluster.shape[1]):
            idx_B = np.where(labels_B == j)[0]
            nB = len(idx_B)
            if nB == 0: continue
                
            val = Pi_cluster[i, j] / (nA * nB)
            grid_A, grid_B = np.ix_(idx_A, idx_B)
            G_init[grid_A, grid_B] = val
            
    # normalize row sums to 1/N
    G_sums = G_init.sum()
    if G_sums > 0:
        G_init = G_init / G_sums
        
    return G_init


def extract_continuous_macro_section(sliceA, sliceB, labels_A, labels_B, Pi_cluster, spatial_key='spatial', extension_hops=2):
    """
    Identifies a continuous, highly-matched section from the clustering alignment
    by performing balanced region growing on the cluster adjacency graph.
    Returns the cell indices for the macro-region in both slices, and boundary distances.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    N, M = sliceA.shape[0], sliceB.shape[0]
    
    total_mass = np.sum(Pi_cluster)
    if total_mass == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    num_clusters_A = Pi_cluster.shape[0]
    num_clusters_B = Pi_cluster.shape[1]
    
    # 1. Precompute cluster centroids and absolute cell counts
    centroids_A = np.zeros((num_clusters_A, 2))
    sizes_A = np.zeros(num_clusters_A)
    for c in range(num_clusters_A):
        idx = np.where(labels_A == c)[0]
        if len(idx) > 0:
            centroids_A[c] = sliceA.obsm[spatial_key][idx].mean(axis=0)
            sizes_A[c] = len(idx)

    centroids_B = np.zeros((num_clusters_B, 2))
    sizes_B = np.zeros(num_clusters_B)
    for c in range(num_clusters_B):
        idx = np.where(labels_B == c)[0]
        if len(idx) > 0:
            centroids_B[c] = sliceB.obsm[spatial_key][idx].mean(axis=0)
            sizes_B[c] = len(idx)

    # 2. Build Contiguous Adjacency Graphs (K=6 physical cluster neighbors)
    nn_A = NearestNeighbors(n_neighbors=min(6, num_clusters_A)).fit(centroids_A)
    adj_A = nn_A.kneighbors_graph(mode='connectivity').toarray()
    
    nn_B = NearestNeighbors(n_neighbors=min(6, num_clusters_B)).fit(centroids_B)
    adj_B = nn_B.kneighbors_graph(mode='connectivity').toarray()

    # 3. Seed initialization from absolute peak confidence cluster pair
    idx_max = np.argmax(Pi_cluster)
    seed_A, seed_B = np.unravel_index(idx_max, Pi_cluster.shape)
    
    core_A, core_B = {seed_A}, {seed_B}
    sum_cells_A, sum_cells_B = sizes_A[seed_A], sizes_B[seed_B]
    
    # Grow to target 75% coverage max - forces edge trim while allowing wide bulk bodies
    target_coverage = 0.75 
    ratio_target = N / float(M)
    min_score = np.max(Pi_cluster) * 0.005 # Allows extending into weakly-moderate matches
    
    # 4. Region Growing Expansion
    while (sum_cells_A / N < target_coverage) or (sum_cells_B / M < target_coverage):
        # Look around outer boundary of current core
        cand_A = set()
        for c in core_A: 
            cand_A.update(np.where(adj_A[c])[0])
        cand_A -= core_A
        
        cand_B = set()
        for c in core_B: 
            cand_B.update(np.where(adj_B[c])[0])
        cand_B -= core_B
        
        if not cand_A and not cand_B:
            break
            
        # Score neighboring candidates solely by Pi flow connected inward into the opposing chosen core
        best_cand_A, best_score_A = None, -1
        for c in cand_A:
            score = sum(Pi_cluster[c, j] for j in core_B)
            if score > best_score_A:
                best_score_A = score
                best_cand_A = c
                
        best_cand_B, best_score_B = None, -1
        for c in cand_B:
            score = sum(Pi_cluster[i, c] for i in core_A)
            if score > best_score_B:
                best_score_B = score
                best_cand_B = c
                
        can_add_A = best_cand_A is not None and best_score_A > min_score
        can_add_B = best_cand_B is not None and best_score_B > min_score
        
        if not can_add_A and not can_add_B:
            break
            
        current_ratio = sum_cells_A / max(1, sum_cells_B)
        added = False
        
        # 5. Dynamically balance growth to preserve N/M physical shape proportions
        if can_add_A and can_add_B:
            if current_ratio < ratio_target:
                core_A.add(best_cand_A)
                sum_cells_A += sizes_A[best_cand_A]
            else:
                core_B.add(best_cand_B)
                sum_cells_B += sizes_B[best_cand_B]
            added = True
        elif can_add_A:
            core_A.add(best_cand_A)
            sum_cells_A += sizes_A[best_cand_A]
            added = True
        elif can_add_B:
            core_B.add(best_cand_B)
            sum_cells_B += sizes_B[best_cand_B]
            added = True
            
        if not added:
            break
            
    # 6. Extraction translation
    core_A, core_B = np.array(list(core_A)), np.array(list(core_B))
    
    idx_A = np.where(np.isin(labels_A, core_A))[0]
    idx_B = np.where(np.isin(labels_B, core_B))[0]
    
    # 7. Compute physical edge distances radiating off the prime "Seed" for core decay weight plan
    coords_A, coords_B = sliceA.obsm[spatial_key], sliceB.obsm[spatial_key]
    
    seed_centroid_A = centroids_A[seed_A]
    seed_centroid_B = centroids_B[seed_B]
    
    dist_A = np.linalg.norm(coords_A[idx_A] - seed_centroid_A, axis=1)
    dist_B = np.linalg.norm(coords_B[idx_B] - seed_centroid_B, axis=1)

    return idx_A, idx_B, dist_A, dist_B
