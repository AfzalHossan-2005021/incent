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
    Identifies the largest co-contiguous, highly-matched section from the clustering alignment.
    Returns the cell indices for the extended macro-region in both slices,
    along with their boundary-distances for weight decay.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    import numpy as np

    N, M = sliceA.shape[0], sliceB.shape[0]
    
    total_mass = np.sum(Pi_cluster)
    if total_mass == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    coords_A, coords_B = sliceA.obsm[spatial_key], sliceB.obsm[spatial_key]

    # 1. Compute Cluster Centroids to build Spatial Graphs
    num_clusters_A, num_clusters_B = Pi_cluster.shape
    centroids_A, centroids_B = np.zeros((num_clusters_A, 2)), np.zeros((num_clusters_B, 2))
    valid_A, valid_B = np.zeros(num_clusters_A, dtype=bool), np.zeros(num_clusters_B, dtype=bool)

    for i in range(num_clusters_A):
        mask = (labels_A == i)
        if np.any(mask):
            centroids_A[i] = coords_A[mask].mean(axis=0)
            valid_A[i] = True

    for i in range(num_clusters_B):
        mask = (labels_B == i)
        if np.any(mask):
            centroids_B[i] = coords_B[mask].mean(axis=0)
            valid_B[i] = True

    # 2. Build Structural Adjacency Matrices for Clusters (k=10)
    k_A = min(10, np.sum(valid_A))
    adj_A = np.zeros((num_clusters_A, num_clusters_A), dtype=bool)
    if k_A > 0:
        nn_A = NearestNeighbors(n_neighbors=k_A).fit(centroids_A[valid_A])
        ind_A = nn_A.kneighbors(centroids_A[valid_A], return_distance=False)
        valid_idx_A = np.where(valid_A)[0]
        for i, neighbors in enumerate(ind_A):
            for n in neighbors:
                adj_A[valid_idx_A[i], valid_idx_A[n]] = True
                adj_A[valid_idx_A[n], valid_idx_A[i]] = True

    k_B = min(10, np.sum(valid_B))
    adj_B = np.zeros((num_clusters_B, num_clusters_B), dtype=bool)
    if k_B > 0:
        nn_B = NearestNeighbors(n_neighbors=k_B).fit(centroids_B[valid_B])
        ind_B = nn_B.kneighbors(centroids_B[valid_B], return_distance=False)
        valid_idx_B = np.where(valid_B)[0]
        for i, neighbors in enumerate(ind_B):
            for n in neighbors:
                adj_B[valid_idx_B[i], valid_idx_B[n]] = True
                adj_B[valid_idx_B[n], valid_idx_B[i]] = True

    np.fill_diagonal(adj_A, True)
    np.fill_diagonal(adj_B, True)

    # 3. Select ENLARGED subset of transport masses (Top 95%)
    flat_pi = Pi_cluster.flatten()
    sorted_idx = np.argsort(flat_pi)[::-1]
    sorted_cumsum = np.cumsum(flat_pi[sorted_idx])
    
    cutoff_idx = np.searchsorted(sorted_cumsum, total_mass * 0.95)
    selected_flat_idx = sorted_idx[:cutoff_idx+1]
    
    matches = []
    for idx in selected_flat_idx:
        u, v = np.unravel_index(idx, Pi_cluster.shape)
        if valid_A[u] and valid_B[v]:
            matches.append((u, v))
            
    num_matches = len(matches)
    if num_matches == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    # 4. Enforce Structural Similarity & Continuity via Match-Graph
    # Two matching pairs are connected ONLY if they are contiguous in BOTH spatial slices
    match_adj = np.zeros((num_matches, num_matches), dtype=bool)
    for i in range(num_matches):
        u1, v1 = matches[i]
        for j in range(i+1, num_matches):
            u2, v2 = matches[j]
            if adj_A[u1, u2] and adj_B[v1, v2]:
                match_adj[i, j] = True
                match_adj[j, i] = True

    # Find the largest Structurally Co-Contiguous Component
    n_comp, comp_labels = connected_components(match_adj, directed=False)
    largest = np.bincount(comp_labels).argmax()
    largest_match_indices = np.where(comp_labels == largest)[0]
    
    strong_A = list(set([matches[i][0] for i in largest_match_indices]))
    strong_B = list(set([matches[i][1] for i in largest_match_indices]))
    
    core_cells_A = np.where(np.isin(labels_A, strong_A))[0]
    core_cells_B = np.where(np.isin(labels_B, strong_B))[0]

    if len(core_cells_A) == 0 or len(core_cells_B) == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
    
    # 5. Cell-level Physical Distance Extension
    tree_A = cKDTree(coords_A[core_cells_A])
    tree_B = cKDTree(coords_B[core_cells_B])
    
    dist_A, _ = tree_A.query(coords_A)
    dist_B, _ = tree_B.query(coords_B)
    
    # Compute dynamic extension radius
    def compute_radius(coords_core):
        if len(coords_core) > 1:
            nn = NearestNeighbors(n_neighbors=min(6, len(coords_core))).fit(coords_core)
            dists, _ = nn.kneighbors(coords_core)
            median_dist = np.median(dists[:, 1:])
            # 1 hop roughly equals the median neighbor distance
            # Enlarging the hop multiplier from 2.0 to 4.0 to aggressively sweep in aligned border structures
            return median_dist * (extension_hops * 4.0)
        return 10.0
            
    rad_A = compute_radius(coords_A[core_cells_A])
    rad_B = compute_radius(coords_B[core_cells_B])
    
    idx_A = np.where(dist_A <= rad_A)[0]
    idx_B = np.where(dist_B <= rad_B)[0]
    
    return idx_A, idx_B, dist_A[idx_A], dist_B[idx_B]
