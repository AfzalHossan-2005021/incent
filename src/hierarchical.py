import ot
import logging
import numpy as np
import scipy.sparse as sp

from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components


def extract_cluster_features(adata, labels, spatial_key="spatial", feature_key="X", label_key="cell_type_annot", all_types=None):
    """
    Extract cluster-level features for coarse mapping.
    
    Args:
        adata: AnnData object.
        labels: np.ndarray of cluster labels for each cell.
        spatial_key: key in obsm for coords.
        feature_key: 'X' for adata.X, else obsm key for latent features.
        label_key: key in obs for cell type annotations.
        all_types: global list of cell types.
        
    Returns:
        masses: np.ndarray (C,) normalized size of each cluster
        mu_expr: np.ndarray (C, D) mean expression/latent vector
        hist_types: np.ndarray (C, T) normalized cell-type histograms
        centroids: np.ndarray (C, 2) average spatial coordinate
        unique_labels: np.ndarray (C,)
        mu_struct: np.ndarray (C, T * 3) spatial distribution within the cluster itself, or None
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
    if all_types is None:
        unique_types = np.unique(ctypes)
    else:
        unique_types = np.array(all_types)
        
    type_map = {t: i for i, t in enumerate(unique_types)}
    n_types = len(unique_types)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    masses = np.zeros(n_clusters)
    mu_expr = np.zeros((n_clusters, expr.shape[1]))
    hist_types = np.zeros((n_clusters, n_types))
    centroids = np.zeros((n_clusters, 2))
    
    # 3 harmonics (m=0, 1, 2) per cell type
    mu_struct = np.zeros((n_clusters, n_types * 3))
    
    for c_i, c in enumerate(unique_labels):
        mask = (labels == c)
        c_size = np.sum(mask)
        masses[c_i] = c_size / float(n_cells)
        
        if c_size > 0:
            mu_expr[c_i, :] = np.mean(expr[mask], axis=0)
            centroids[c_i, :] = np.mean(coords[mask], axis=0)
            
            c_types = ctypes[mask]
            # Map types; ignore if they don't exist in target mapping
            mapped_types = [type_map[t] for t in c_types if t in type_map]
            counts = np.bincount(mapped_types, minlength=n_types).astype(np.float64)
            hist_types[c_i, :] = counts / float(c_size) # normalized histogram
            
            # --- Computed Intrinsic Cluster Fourier Context ---
            # Evaluates cell localization RELATIVE to the macro-cluster centroid
            rel_coords = coords[mask] - centroids[c_i]
            thetas = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
            
            local_feat = np.zeros((n_types, 3))
            c_types_idx = np.array(mapped_types)
            
            for h_idx, m in enumerate([0, 1, 2]):
                if m == 0:
                    mag = counts
                else:
                    ang = m * thetas
                    real = np.bincount(c_types_idx, weights=np.cos(ang), minlength=n_types)
                    imag = np.bincount(c_types_idx, weights=np.sin(ang), minlength=n_types)
                    mag = np.hypot(real, imag)
                    
                local_feat[:, h_idx] = mag
            
            flat = local_feat.reshape(-1)
            if flat.sum() > 0:
                flat /= flat.sum()
            mu_struct[c_i, :] = flat
            
    return masses, mu_expr, hist_types, centroids, unique_labels, mu_struct


def compute_cluster_costs(featuresA, featuresB, w_expr=0.5, w_type=0.5, w_struct=0.0):
    """
    Compute inter-cluster cost matrix M_cluster between two slices.
    
    Args:
        featuresA: Tuple of features from slice A
        featuresB: Tuple of features from slice B
        
    Returns:
        M_cluster: np.ndarray (C_A, C_B) cost matrix
    """
    _, mu_expr_A, hist_types_A, _, _, mu_struct_A = featuresA
    _, mu_expr_B, hist_types_B, _, _, mu_struct_B = featuresB
    
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
            
    # Re-weighting based on provided w_struct
    total_w = w_expr + w_type + w_struct
    if total_w == 0: total_w = 1.0
    w_expr_norm = w_expr / total_w
    w_type_norm = w_type / total_w
    w_struct_norm = w_struct / total_w
    
    M_cluster = w_expr_norm * M_expr + w_type_norm * M_type
    
    if mu_struct_A is not None and mu_struct_B is not None and w_struct > 0:
        M_struct = np.zeros((mu_struct_A.shape[0], mu_struct_B.shape[0]))
        for i in range(mu_struct_A.shape[0]):
            for j in range(mu_struct_B.shape[0]):
                # Using Jensen-Shannon since the fourier features are normalized probabilistic distributions over space
                M_struct[i, j] = jensenshannon(mu_struct_A[i], mu_struct_B[j])
        
        M_cluster += w_struct_norm * M_struct
        
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


def extract_continuous_macro_section(sliceA, sliceB, labels_A, labels_B, Pi_cluster, mass_pct=0.85, spatial_key='spatial', extension_hops=2):
    """
    Identifies the largest co-contiguous, highly-matched section from the clustering alignment.
    Returns the cell indices for the extended macro-region in both slices,
    along with their boundary-distances for weight decay.
    """
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

    # 2. Build Structural Adjacency Matrices for Clusters based on true spatial borders
    # Using Delaunay triangulation to find neighboring clusters, then filter by valid clusters and Pi_cluster mass
    # calculate a spatial graph using Delaunay triangulation on all the cells' coordinates. If a Delaunay triangle contains cells from two different clusters, it guarantees they are physically touching at their borders.
    from scipy.spatial import Delaunay
    
    adj_A = np.zeros((num_clusters_A, num_clusters_A), dtype=bool)
    if N >= 3:
        tri_A = Delaunay(coords_A)
        for simplex in tri_A.simplices:
            lab = labels_A[simplex]
            for i in range(3):
                for j in range(i+1, 3):
                    if lab[i] != lab[j] and valid_A[lab[i]] and valid_A[lab[j]]:
                        adj_A[lab[i], lab[j]] = True
                        adj_A[lab[j], lab[i]] = True
                        
    adj_B = np.zeros((num_clusters_B, num_clusters_B), dtype=bool)
    if M >= 3:
        tri_B = Delaunay(coords_B)
        for simplex in tri_B.simplices:
            lab = labels_B[simplex]
            for i in range(3):
                for j in range(i+1, 3):
                    if lab[i] != lab[j] and valid_B[lab[i]] and valid_B[lab[j]]:
                        adj_B[lab[i], lab[j]] = True
                        adj_B[lab[j], lab[i]] = True

    np.fill_diagonal(adj_A, True)
    np.fill_diagonal(adj_B, True)

    # 3. Select ENLARGED subset of transport masses (Top {mass_pct} of total mass)
    flat_pi = Pi_cluster.flatten()
    sorted_idx = np.argsort(flat_pi)[::-1]
    sorted_cumsum = np.cumsum(flat_pi[sorted_idx])
    
    cutoff_idx = np.searchsorted(sorted_cumsum, total_mass * mass_pct)
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
    
    # Compute initial barycenters of the starting core components
    bary_A = np.mean(centroids_A[strong_A], axis=0)
    bary_B = np.mean(centroids_B[strong_B], axis=0)

    # 5. Cluster-level Topological Extension
    for _ in range(extension_hops):
        # Find all topological neighbors of the current strong components
        neighbors_A = np.where(np.any(adj_A[strong_A, :], axis=0))[0]
        neighbors_B = np.where(np.any(adj_B[strong_B, :], axis=0))[0]
        
        # Exclude clusters already in the strong set
        candidates_A = [c for c in neighbors_A if c not in strong_A and valid_A[c]]
        candidates_B = [c for c in neighbors_B if c not in strong_B and valid_B[c]]
        
        if not candidates_A or not candidates_B:
            break
            
        # Find the valid matching pair that is closest to the initial barycenters
        best_pair = None
        best_dist = float('inf')
        
        # Threshold to filter out nearly-zero noise entries from FGW
        min_mass = np.max(Pi_cluster) * 1e-4 
        
        for pA in candidates_A:
            for pB in candidates_B:
                # The pair must still have valid OT confidence to be aligned together!
                if Pi_cluster[pA, pB] > min_mass:
                    dist_A = np.linalg.norm(centroids_A[pA] - bary_A)
                    dist_B = np.linalg.norm(centroids_B[pB] - bary_B)
                    
                    # We minimize the combined distance to the respective barycenters
                    total_dist = dist_A + dist_B
                    if total_dist < best_dist:
                        best_dist = total_dist
                        best_pair = (pA, pB)
                    
        # Add the best geometrically compact pair
        if best_pair is not None:
            strong_A.append(best_pair[0])
            strong_B.append(best_pair[1])
        else:
            break

    # Extract final cells based strictly on expanded cluster membership
    idx_A = np.where(np.isin(labels_A, strong_A))[0]
    idx_B = np.where(np.isin(labels_B, strong_B))[0]

    # Compute physical distances from the newly expanded core to all cells
    core_coords_A = coords_A[idx_A]
    core_coords_B = coords_B[idx_B]

    tree_A = cKDTree(core_coords_A)
    tree_B = cKDTree(core_coords_B)
    
    dist_A, _ = tree_A.query(coords_A)
    dist_B, _ = tree_B.query(coords_B)
    
    return idx_A, idx_B, dist_A, dist_B
