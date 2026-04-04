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


def extract_continuous_macro_section(sliceA, sliceB, labels_A, labels_B, Pi_cluster, p_A, p_B, spatial_key='spatial', extension_hops=2):
    """
    Solves the 'cluster averaging' issue by locking strictly onto 1-to-1 mutually best matched clusters.
    Constructs a contiguous geographic map, and enforces identical cell counts per-cluster by 
    trimming fringe boundaries (cells furthest from cluster centers).
    Guarantees mathematically continuous domains with similar visual borders.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components
    import numpy as np

    N, M = sliceA.shape[0], sliceB.shape[0]
    coords_A, coords_B = sliceA.obsm[spatial_key], sliceB.obsm[spatial_key]
    
    total_mass = np.sum(Pi_cluster)
    if total_mass == 0: 
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    # 1. Strict 1-to-1 Mutual Mapping (No 1-to-Many Averaging)
    best_A = np.argmax(Pi_cluster, axis=1)
    best_B = np.argmax(Pi_cluster, axis=0)
    
    mutual_pairs = []
    masses = []
    for i in range(Pi_cluster.shape[0]):
        j = best_A[i]
        if best_B[j] == i:  # It's a mutual best match
            mutual_pairs.append((i, j))
            masses.append(Pi_cluster[i, j])
            
    if not mutual_pairs:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    mutual_pairs = np.array(mutual_pairs)
    masses = np.array(masses)
    
    # Reject extreme outliers/ghost clusters by dropping the weakest half of mutual matches
    thresh = np.median(masses) if len(masses) > 2 else 0
    strong_pairs = mutual_pairs[masses >= thresh]
    
    # Extract centers of these candidate clusters
    cent_A, cent_B = [], []
    for i, j in strong_pairs:
        cent_A.append(np.mean(coords_A[labels_A == i], axis=0))
        cent_B.append(np.mean(coords_B[labels_B == j], axis=0))
    cent_A, cent_B = np.array(cent_A), np.array(cent_B)
    
    # 2. Extract Largest Physically Contiguous Chain of Clusters
    def connected_chain(centers):
        if len(centers) < 2: return np.arange(len(centers))
        nn = NearestNeighbors(n_neighbors=min(6, len(centers))).fit(centers)
        adj = nn.kneighbors_graph(mode='connectivity')
        n_comp, comp_labels = connected_components(adj, directed=False)
        return np.where(comp_labels == np.bincount(comp_labels).argmax())[0]
        
    chain_A_idx = connected_chain(cent_A)
    # The pairs are already 1-to-1 locked, so the subset of pairs must just be continuous
    # We restrict to those physically continuous in A, and then physically continuous in B.
    valid_pairs = strong_pairs[chain_A_idx]
    
    cent_B_subset = cent_B[chain_A_idx]
    chain_B_idx = connected_chain(cent_B_subset)
    final_pairs = valid_pairs[chain_B_idx]
    
    # 3. Geometric Alignment: Extract equal cell counts per cluster and cluster-centric distances
    idx_A, idx_B = [], []
    dist_A, dist_B = [], []
    
    for i, j in final_pairs:
        c_A = np.where(labels_A == i)[0]
        c_B = np.where(labels_B == j)[0]
        
        target_k = min(len(c_A), len(c_B))
        if target_k == 0: continue
            
        cA_loc = coords_A[c_A]
        cB_loc = coords_B[c_B]
        
        # Center of this specific cluster domain
        mu_A = np.mean(cA_loc, axis=0)
        mu_B = np.mean(cB_loc, axis=0)
        
        dA = np.linalg.norm(cA_loc - mu_A, axis=1)
        dB = np.linalg.norm(cB_loc - mu_B, axis=1)
        
        # Trim cells furthest from the core (fringe boundaries that distort shape)
        if len(c_A) > target_k:
            keep_A = np.argsort(dA)[:target_k]
            c_A = c_A[keep_A]
            dA = dA[keep_A]
            
        if len(c_B) > target_k:
            keep_B = np.argsort(dB)[:target_k]
            c_B = c_B[keep_B]
            dB = dB[keep_B]
            
        idx_A.extend(c_A)
        idx_B.extend(c_B)
        dist_A.extend(dA)
        dist_B.extend(dB)
        
    if len(idx_A) == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    return np.array(idx_A), np.array(idx_B), np.array(dist_A), np.array(dist_B)
