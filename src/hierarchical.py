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
    Identifies the largest continuous, structurally-identical macro-section 
    by automatically detecting and solving major anatomical portions (e.g. 1 vs 2 hemispheres).
    Then uses core-OT point strict mutual overlap and draws matching cell quotas.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import DBSCAN
    from scipy.spatial import cKDTree
    import numpy as np

    N, M = sliceA.shape[0], sliceB.shape[0]
    coords_A, coords_B = sliceA.obsm[spatial_key], sliceB.obsm[spatial_key]

    total_mass = np.sum(Pi_cluster)
    if total_mass == 0: return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    # --- SMART ALIGNMENT: Dynamic Structural Portion Matching ---
    # Automatically detect disconnected tissue portions in physical space
    nn_A = NearestNeighbors(n_neighbors=6).fit(coords_A)
    eps_A = np.median(nn_A.kneighbors(coords_A)[0][:, 1:]) * 5.0
    port_A = DBSCAN(eps=eps_A, min_samples=10).fit_predict(coords_A)
    
    nn_B = NearestNeighbors(n_neighbors=6).fit(coords_B)
    eps_B = np.median(nn_B.kneighbors(coords_B)[0][:, 1:]) * 5.0
    port_B = DBSCAN(eps=eps_B, min_samples=10).fit_predict(coords_B)
    
    unique_port_A = [p for p in np.unique(port_A) if p >= 0]
    unique_port_B = [p for p in np.unique(port_B) if p >= 0]
    
    if len(unique_port_A) > 0 and len(unique_port_B) > 0 and (len(unique_port_A) != len(unique_port_B)):
        import logging
        logging.info(f"Smart Align: Asymmetric portions detected (A:{len(unique_port_A)} vs B:{len(unique_port_B)}). Finding best subspace mapping.")
        mass_matrix = np.zeros((len(unique_port_A), len(unique_port_B)))
        
        for i, pa in enumerate(unique_port_A):
            clusters_pa = np.unique(labels_A[port_A == pa])
            for j, pb in enumerate(unique_port_B):
                clusters_pb = np.unique(labels_B[port_B == pb])
                if len(clusters_pa) > 0 and len(clusters_pb) > 0:
                    grid = np.ix_(clusters_pa, clusters_pb)
                    mass_matrix[i, j] = np.sum(Pi_cluster[grid])
                
        # Smart Select: Keep only the portions holding the maximal OT mass
        keep_port_A, keep_port_B = unique_port_A, unique_port_B
        if len(unique_port_A) < len(unique_port_B):
            best_B_idx = np.argmax(mass_matrix, axis=1)
            keep_port_B = [unique_port_B[j] for j in best_B_idx]
        else:
            best_A_idx = np.argmax(mass_matrix, axis=0)
            keep_port_A = [unique_port_A[i] for i in best_A_idx]
            
        # Zero out the cross-portion mass for anything completely unmatched
        valid_cA = np.unique(labels_A[np.isin(port_A, keep_port_A)])
        valid_cB = np.unique(labels_B[np.isin(port_B, keep_port_B)])
        
        invalid_mask_A = ~np.isin(np.arange(Pi_cluster.shape[0]), valid_cA)
        invalid_mask_B = ~np.isin(np.arange(Pi_cluster.shape[1]), valid_cB)
        Pi_cluster[invalid_mask_A, :] = 0
        Pi_cluster[:, invalid_mask_B] = 0
    # 1. Strict 1-to-1 Mutual Mapping to prevent averaging across multi-mapped clusters
    best_A = np.argmax(Pi_cluster, axis=1)
    best_B = np.argmax(Pi_cluster, axis=0)

    mutual_pairs = []
    masses = []
    for i in range(Pi_cluster.shape[0]):
        j = best_A[i]
        if best_B[j] == i:
            mutual_pairs.append((i, j))
            masses.append(Pi_cluster[i, j])

    # Extract centroids to use as rigid anchor points
    anchor_A, anchor_B = [], []
    if len(mutual_pairs) >= 3:
        # Use top 50% most confident mutual matches to avoid boundary distortion
        mutual_pairs = np.array(mutual_pairs)
        masses = np.array(masses)
        thresh = np.median(masses) if len(masses) > 3 else 0
        strong_pairs = mutual_pairs[masses >= thresh]
        for i, j in strong_pairs:
            anchor_A.append(np.mean(coords_A[labels_A == i], axis=0))
            anchor_B.append(np.mean(coords_B[labels_B == j], axis=0))
    
    # Fallback if too few strict mutual matches exist for 2D rigid transform
    if len(anchor_A) < 3:
        flat_pi = Pi_cluster.flatten()
        sorted_idx = np.argsort(flat_pi)[::-1]
        pairs_A, pairs_B = np.unravel_index(sorted_idx, Pi_cluster.shape)
        
        seen_A, seen_B = set(), set()
        anchor_A, anchor_B = [], []
        for i, j in zip(pairs_A, pairs_B):
            if i not in seen_A and j not in seen_B:
                seen_A.add(i)
                seen_B.add(j)
                anchor_A.append(np.mean(coords_A[labels_A == i], axis=0))
                anchor_B.append(np.mean(coords_B[labels_B == j], axis=0))
                if len(anchor_A) >= max(3, len(mutual_pairs)): break
                
    if len(anchor_A) < 3: 
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    anchor_A, anchor_B = np.array(anchor_A), np.array(anchor_B)

    # 2. Rigid Tissue Alignment (Orthogonal Procrustes)
    X, Y = anchor_A, anchor_B
    mu_X, mu_Y = np.mean(X, axis=0), np.mean(Y, axis=0)
    H = (Y - mu_Y).T @ (X - mu_X)
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    # Slice B geometrically projected cleanly onto Slice A space
    coords_B_aligned = (coords_B - mu_Y) @ R + mu_X

    # 3. Base native geometric density for spatial intersection
    nn = NearestNeighbors(n_neighbors=6).fit(coords_A)
    dists, _ = nn.kneighbors(coords_A)
    # Define rigid spatial overlap bound roughly equal to "extension_hops" cells away
    search_radius = np.median(dists[:, 1:]) * max(2.0, float(extension_hops))

    # 4. Strict Geometric Overlap
    tree_A = cKDTree(coords_A)
    tree_B = cKDTree(coords_B_aligned)

    dist_A_to_B, _ = tree_B.query(coords_A)       # How far is A's cell from B's mapped tissue
    dist_B_to_A, _ = tree_A.query(coords_B_aligned) # How far is B's cell from A's tissue

    # Cells that perfectly exist in both slices simultaneously (identical spatial footprint)
    overlap_A = np.where(dist_A_to_B < search_radius)[0]
    overlap_B = np.where(dist_B_to_A < search_radius)[0]

    if len(overlap_A) < 5 or len(overlap_B) < 5:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)

    # 5. Restrict to a STRICTLY CONTINUOUS physical structure
    # By using radius graph, we absolutely forbid jumping across empty gaps
    def largest_contiguous_block(coords, cell_idx, rad):
        if len(cell_idx) < 2: return cell_idx
        nn_rad = NearestNeighbors(radius=rad * 1.5).fit(coords[cell_idx])
        adj = nn_rad.radius_neighbors_graph(mode='connectivity')
        n_comp, comp_labels = connected_components(adj, directed=False)
        return cell_idx[comp_labels == np.bincount(comp_labels).argmax()]

    contig_A = largest_contiguous_block(coords_A, overlap_A, search_radius)
    contig_B = largest_contiguous_block(coords_B_aligned, overlap_B, search_radius)

    # 6. Force exactly identical numbers of cells for absolutely balanced geometry
    target_count = min(len(contig_A), len(contig_B))
    
    if target_count > 0:
        # If A has 5 more cells, we drop the 5 cells that are structurally furthest from B!
        sort_A = np.argsort(dist_A_to_B[contig_A])
        idx_A = contig_A[sort_A[:target_count]]

        sort_B = np.argsort(dist_B_to_A[contig_B])
        idx_B = contig_B[sort_B[:target_count]]
    else:
        idx_A, idx_B = np.arange(N), np.arange(M)

    return idx_A, idx_B, dist_A_to_B[idx_A], dist_B_to_A[idx_B]
