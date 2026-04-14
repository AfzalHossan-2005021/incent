import logging
import numpy as np
import scipy.sparse as sp

from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance_matrix, Delaunay
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import jensenshannon
from ot.gromov import fused_unbalanced_gromov_wasserstein


def extract_cluster_features(adata, labels, spatial_key="spatial", feature_key="X_pca", label_key="cell_type_annot", all_types=None) -> tuple:
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
        centroids: np.ndarray (C, 2) average spatial coordinate
        mu_expr: np.ndarray (C, D) mean expression/latent vector
        mu_struct: np.ndarray (C, T * 3) spatial distribution within the cluster itself based on 3 Fourier harmonics of the cell type localization
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
    centroids = np.zeros((n_clusters, 2))
    
    def compute_reference_angle(pts):
        centered = pts - pts.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        v1 = vh[0]
        if np.mean((centered @ v1) ** 3) < 0:
            v1 = -v1
        return np.arctan2(v1[1], v1[0])
        
    ref_angle = compute_reference_angle(coords)
    
    # 5 harmonics (m=0 (1), m=1 (2), m=2 (2)) per cell type
    mu_struct = np.zeros((n_clusters, n_types * 5))
    
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
            
            # --- Computed Intrinsic Cluster Fourier Context ---
            # Evaluates cell localization RELATIVE to the macro-cluster centroid
            rel_coords = coords[mask] - centroids[c_i]
            thetas = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
            
            local_feat = np.zeros((n_types, 5))
            c_types_idx = np.array(mapped_types)
            
            feat_idx = 0
            for m in [0, 1, 2]:
                if m == 0:
                    local_feat[:, feat_idx] = counts
                    feat_idx += 1
                else:
                    ang = m * thetas
                    real = np.bincount(c_types_idx, weights=np.cos(ang), minlength=n_types)
                    imag = np.bincount(c_types_idx, weights=np.sin(ang), minlength=n_types)
                    
                    phase_anchor = np.exp(-1j * m * ref_angle)
                    descriptor_complex = (real + 1j * imag) * phase_anchor
                    
                    local_feat[:, feat_idx] = descriptor_complex.real
                    local_feat[:, feat_idx + 1] = descriptor_complex.imag
                    feat_idx += 2
            
            # Make nonnegative for Jensen-Shannon divergence
            local_feat = local_feat - local_feat.min()
            
            flat = local_feat.reshape(-1)
            if flat.sum() > 0:
                flat /= flat.sum()
            mu_struct[c_i, :] = flat
            
    return masses, centroids, mu_expr, mu_struct


def compute_cluster_feature_costs(mu_expr_A, mu_struct_A, mu_expr_B, mu_struct_B, beta=0.75):
    """
    Compute inter-cluster cost matrix M_cluster between two slices.
    
    Args:
        mu_expr_A: np.ndarray (C_A, D) mean expression for slice A
        mu_struct_A: np.ndarray (C_A, T * 3) structural features for slice A
        mu_expr_B: np.ndarray (C_B, D) mean expression for slice B
        mu_struct_B: np.ndarray (C_B, T * 3) structural features for slice B
        beta: weight for structural distance (expression distance is 1 - beta)
        
    Returns:
        M_cluster: np.ndarray (C_A, C_B) cost matrix
    """

    if(beta > 1.0 or beta < 0.0):
        raise ValueError("Beta must be between 0 and 1.")
    
    # Cosine distance for continuous expression
    M_expr = cosine_distances(mu_expr_A, mu_expr_B)
            
    # Jensen-Shannon for cell type histograms (probability distributions)
    M_struct = np.zeros((mu_struct_A.shape[0], mu_struct_B.shape[0]))
    for i in range(mu_struct_A.shape[0]):
        for j in range(mu_struct_B.shape[0]):
            # Using Jensen-Shannon since the fourier features are normalized probabilistic distributions over space
            M_struct[i, j] = jensenshannon(mu_struct_A[i], mu_struct_B[j])

    
    M_cluster = (1.0 - beta) * M_expr + beta * M_struct
        
    return M_cluster


def compute_cluster_structural_matrix(centroids, w_euc=0.5, w_graph=0.5):
    """
    Compute the structural distance matrix C for clusters based on their centroids.
    
    Args:
        centroids: (C, 2) coords of clusters.
        w_euc: weight for direct euclidean distance.
        w_graph: weight for graph-based distance.
    
    Returns:
        C: (C, C) combined structural distance matrix.
    """
    # Simple euclidean distance for now if w_graph is 0
    C_euc = distance_matrix(centroids, centroids)
    
    if w_graph > 0:
        # build adj matrix
        n = centroids.shape[0]
        adj = np.zeros((n, n))

        tri = Delaunay(centroids)
        indptr, indices = tri.vertex_neighbor_vertices
        for i in range(n):
            for j in indices[indptr[i]:indptr[i+1]]:
                adj[i, j] = np.linalg.norm(centroids[i] - centroids[j])
        
        C_graph = dijkstra(sp.csr_matrix(adj), directed=False)
        # handle infinite dists
        C_graph[np.isinf(C_graph)] = np.max(C_graph[~np.isinf(C_graph)]) * 2
        
        w_total = w_euc + w_graph
        if w_total == 0: w_total = 1.0
        w_euc_norm = w_euc / w_total
        w_graph_norm = w_graph / w_total
            
        C = w_euc_norm * C_euc + w_graph_norm * C_graph
    else:
        C = C_euc
        
    return C


def run_coarse_partial_fgw(M_cluster, C_A, C_B, p_A, p_B, alpha=0.5, m=None, reg_m=1.0):
    """
    Solves cluster-level partial/unbalanced FGW.
    """
    if m is None:
        m = min(np.sum(p_A), np.sum(p_B)) * 0.999

    scale = max(C_A.max(), C_B.max()) + 1e-8
        
    C_A_norm = C_A / scale
    C_B_norm = C_B / scale
    M_norm = M_cluster / (np.max(M_cluster) + 1e-8)
    
    logging.info("Running Unbalanced FGW...")
    # reg_marginals controls how much marginal relaxation is allowed (lower = more mass can be dropped)
    pi_samp, pi_feat = fused_unbalanced_gromov_wasserstein(
        Cx=C_A_norm, Cy=C_B_norm, wx=p_A, wy=p_B, M=M_norm, alpha=alpha, reg_marginals=reg_m, max_iter=5000
    )
    
    return pi_samp


def extract_continuous_macro_section(sliceA, sliceB, labels_A, labels_B, Pi_cluster, spatial_key='spatial'):
    """
    Identifies the largest co-contiguous, highly-matched section from the clustering alignment.
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

    # 2. Build Structural Adjacency Matrices for Clusters based on local density interaction radii
    def build_structural_adjacency(coords, labels, valid_mask):
        n_clusters = len(valid_mask)
        adj = np.zeros((n_clusters, n_clusters), dtype=bool)
        np.fill_diagonal(adj, True)
        
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < 2:
            return adj
            
        centroids = np.zeros((len(valid_idx), 2))
        kdtries = {}
        intra_dists = {}
        
        # Determine internal neighborhood spacings per cluster
        for i, c_id in enumerate(valid_idx):
            c_coords = coords[labels == c_id]
            centroids[i] = c_coords.mean(axis=0)
            
            tree = cKDTree(c_coords)
            kdtries[c_id] = tree
            
            if len(c_coords) > 1:
                # 99th percentile of internal 1-NN distances dictates natural maximum spacing
                d, _ = tree.query(c_coords, k=2)
                intra_dists[c_id] = np.percentile(d[:, 1], 99)
            else:
                intra_dists[c_id] = 0.0

        # Centroid Delaunay quickly limits our evaluations to macroscopic topological neighbors
        if len(valid_idx) >= 3:
            tri = Delaunay(centroids)
            candidate_edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        u, v = valid_idx[simplex[i]], valid_idx[simplex[j]]
                        candidate_edges.add((min(u, v), max(u, v)))
        else:
            u, v = valid_idx[0], valid_idx[1]
            candidate_edges = {(min(u, v), max(u, v))}
            
        # Parameter-free geometric verification
        for u, v in candidate_edges:
            # Minimum physical inter-cluster gap via rapid KD-Tree
            min_dists, _ = kdtries[u].query(coords[labels == v], k=1)
            min_gap = np.min(min_dists)
            
            # Clusters touch if the gap is smaller than their combined local topological spacing
            if min_gap <= (intra_dists[u] + intra_dists[v]):
                adj[u, v] = True
                adj[v, u] = True
                
        return adj

    adj_A = build_structural_adjacency(coords_A, labels_A, valid_A)
    adj_B = build_structural_adjacency(coords_B, labels_B, valid_B)

    # 3. Parameter-free statistical expectation for OT plan
    # Under random/independent transport, mass distributes as the outer product of marginals.
    marg_A = np.sum(Pi_cluster, axis=1)
    marg_B = np.sum(Pi_cluster, axis=0)
    expected_pi = np.outer(marg_A, marg_B) / (total_mass + 1e-8)
    
    # True biological assignments must concentrate mass significantly above this mathematical null expectation.
    significant_matches = (Pi_cluster > expected_pi) & (Pi_cluster > 0)
    
    valid_masses = Pi_cluster.copy()
    valid_masses[~significant_matches] = 0.0
    
    if np.max(valid_masses) == 0:
        valid_masses = Pi_cluster # Fallback if null test yields no pairs
        
    # 4. Region Growing via Contiguous Expansion from Most Confident Seed
    start_flat = np.argmax(valid_masses)
    start_A, start_B = np.unravel_index(start_flat, valid_masses.shape)
    
    strong_A = {start_A}
    strong_B = {start_B}
    
    # Precompute individual cluster masses for accurate center-of-mass tracking
    masses_A = np.array([np.sum(labels_A == c) for c in range(num_clusters_A)])
    masses_B = np.array([np.sum(labels_B == c) for c in range(num_clusters_B)])

    # Stop when the smaller slice is fully incorporated into the core
    while len(strong_A) < np.sum(valid_A) and len(strong_B) < np.sum(valid_B):
        # The morphological frontier: valid clusters directly touching the current core but NOT inside it
        frontier_A = {i for i in range(num_clusters_A) if np.any(adj_A[i, list(strong_A)]) and valid_A[i] and i not in strong_A}
        frontier_B = {j for j in range(num_clusters_B) if np.any(adj_B[j, list(strong_B)]) and valid_B[j] and j not in strong_B}
        
        # If we cannot contiguously expand simultaneously in both slices, stop.
        if not frontier_A or not frontier_B:
            break
        
        # Dynamic center of mass of the assembled biological core
        bary_A = np.average(centroids_A[list(strong_A)], axis=0, weights=masses_A[list(strong_A)])
        bary_B = np.average(centroids_B[list(strong_B)], axis=0, weights=masses_B[list(strong_B)])
        
        # Compute current spatial radius of the core to evaluate isometric scaling
        rad_A = np.mean([np.linalg.norm(centroids_A[c] - bary_A) for c in strong_A])
        rad_B = np.mean([np.linalg.norm(centroids_B[c] - bary_B) for c in strong_B])
        rad_A = rad_A if rad_A > 0 else 1.0
        rad_B = rad_B if rad_B > 0 else 1.0
        
        # --- Compute rigid transform to define the core's exact morphological shadow ---
        # We align the already-selected core structures through a weighted Kabsch operation
        sA_list = list(strong_A)
        sB_list = list(strong_B)
        
        P_c = centroids_A[sA_list] - bary_A
        Q_c = centroids_B[sB_list] - bary_B
        
        # Construct the cross-covariance matrix heavily weighted precisely by the OT matching probabilities
        W_cross = valid_masses[np.ix_(sA_list, sB_list)]
        H = P_c.T @ W_cross @ Q_c
        
        # SVD handles extraction of the optimal rotation bridging spatial slice A to slice B
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        
        # Ensure we construct a pure rotation (determinant 1) by rectifying reflection matrices
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        
        best_pair = None
        best_score = -1.0
        
        # Strictly evaluate 1-to-1 candidates from the unused frontiers
        # This guarantees "one cluster considered only once" and balanced increments
        for pA in frontier_A:
            for pB in frontier_B:
                # Must pass null expectation threshold mapping
                if valid_masses[pA, pB] > 0:
                    # Geometrically project candidate pA into Slice B utilizing the core's rigid shadow
                    pA_proj = (centroids_A[pA] - bary_A) @ R + bary_B
                    
                    # Shadow Distance: Deviation of target pB from candidate pA's projected shadow
                    shadow_dist = np.linalg.norm(pA_proj - centroids_B[pB])
                    
                    # Scale down the absolute coordinate distance into a relative penalty based on the core size
                    shadow_penalty = shadow_dist / rad_B
                    
                    # Maximize confidently mapped pair mass while heavily enforcing structural coherence 
                    # (Exponential decay guarantees only clusters mathematically inside the shadow survive)
                    score = valid_masses[pA, pB] * np.exp(-shadow_penalty)
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (pA, pB)
                        
        if best_pair is not None and best_score > 0:
            strong_A.add(best_pair[0])
            strong_B.add(best_pair[1])
        else:
            break
            
    # --- Final Polish: Shadow Alignment Refinement ---
    # Re-evaluate all assigned and unassigned clusters using the final converged rigid body projection.
    # We strip out structurally incoherent clusters that were incorporated early and replace them.
    sA_list = list(strong_A)
    sB_list = list(strong_B)
    bary_A = np.average(centroids_A[sA_list], axis=0, weights=masses_A[sA_list])
    bary_B = np.average(centroids_B[sB_list], axis=0, weights=masses_B[sB_list])
    
    rad_B = np.mean([np.linalg.norm(centroids_B[c] - bary_B) for c in sB_list])
    rad_B = rad_B if rad_B > 0 else 1.0

    P_c = centroids_A[sA_list] - bary_A
    Q_c = centroids_B[sB_list] - bary_B
    W_cross = valid_masses[np.ix_(sA_list, sB_list)]
    H = P_c.T @ W_cross @ Q_c

    U, _, Vt = np.linalg.svd(H)
    R_final = U @ Vt
    if np.linalg.det(R_final) < 0:
        Vt[-1, :] *= -1
        R_final = U @ Vt
        
    # Project all valid_A nodes into B's space using the converged global rigid rotation
    proj_A = (centroids_A - bary_A) @ R_final + bary_B
    
    new_strong_A = set()
    new_strong_B = set()
    
    # Bipartite Assignment over the entire valid field based on the computed geometrical structural shadow
    for pA in range(num_clusters_A):
        if not valid_A[pA]: continue
        
        # Only evaluate valid B clusters
        valid_B_idx = np.where(valid_B)[0]
        
        best_pB = None
        best_score = -1.0
        
        for pB in valid_B_idx:
            if valid_masses[pA, pB] > 0:
                shadow_dist = np.linalg.norm(proj_A[pA] - centroids_B[pB])
                shadow_penalty = shadow_dist / rad_B
                score = valid_masses[pA, pB] * np.exp(-shadow_penalty)
                
                # Only accept pairs with score heavily aligned to the actual shadow (> 10% of pure valid OT mass)
                if score > best_score and score > (valid_masses[pA, pB] * 0.1):
                    best_score = score
                    best_pB = pB
                    
        if best_pB is not None:
            new_strong_A.add(pA)
            new_strong_B.add(best_pB)
            
    strong_A = list(new_strong_A)
    strong_B = list(new_strong_B)

    if len(strong_A) == 0 or len(strong_B) == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M)
        
    # 5. Extract Final Cell Indices
    idx_A = np.where(np.isin(labels_A, strong_A))[0]
    idx_B = np.where(np.isin(labels_B, strong_B))[0]

    # Compute physical distances from the contiguous footprint back to all cells
    tree_A = cKDTree(coords_A[idx_A])
    tree_B = cKDTree(coords_B[idx_B])
    
    dist_A, _ = tree_A.query(coords_A)
    dist_B, _ = tree_B.query(coords_B)
    
    return idx_A, idx_B, dist_A, dist_B

