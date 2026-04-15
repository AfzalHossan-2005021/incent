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
        
    # 4. Region Growing via Contiguous Expansion from Most Confident Edge
    # Seeding by Edge: A single point has undefined rotation, which can cause erratic early growth.
    # We find the highest-confidence paired adjacency (an edge) to lock an initial translation and rotation vector.
    best_edge_score = -1.0
    best_edge = None
    
    valid_A_idx = np.where(valid_A)[0]
    valid_B_idx = np.where(valid_B)[0]
    
    for uA in valid_A_idx:
        for vA in valid_A_idx:
            if uA >= vA or not adj_A[uA, vA]: continue
            
            for uB in valid_B_idx:
                for vB in valid_B_idx:
                    if uB == vB or not adj_B[uB, vB]: continue
                    
                    # Match can be (uA->uB, vA->vB) or (uA->vB, vA->uB)
                    score1 = valid_masses[uA, uB] * valid_masses[vA, vB]
                    score2 = valid_masses[uA, vB] * valid_masses[vA, uB]
                    
                    if score1 > best_edge_score:
                        best_edge_score = score1
                        best_edge = ((uA, uB), (vA, vB))
                    if score2 > best_edge_score:
                        best_edge_score = score2
                        best_edge = ((uA, vB), (vA, uB))
                        
    if best_edge is None or best_edge_score <= 0.0:
        # Fallback to single point if no valid edges exist
        start_flat = np.argmax(valid_masses)
        start_A, start_B = np.unravel_index(start_flat, valid_masses.shape)
        mapped_pairs = [(start_A, start_B)]
    else:
        mapped_pairs = [(best_edge[0][0], best_edge[0][1]), (best_edge[1][0], best_edge[1][1])]
    
    # Precompute individual cluster masses for accurate center-of-mass tracking
    masses_A = np.array([np.sum(labels_A == c) for c in range(num_clusters_A)])
    masses_B = np.array([np.sum(labels_B == c) for c in range(num_clusters_B)])
    
    # Scale parameters for dimensionless geometric comparisons
    tree_A = cKDTree(centroids_A[valid_A])
    d_A, _ = tree_A.query(centroids_A[valid_A], k=2)
    spacing_A = np.median(d_A[:, 1]) if len(d_A) > 0 else 1.0

    tree_B = cKDTree(centroids_B[valid_B])
    d_B, _ = tree_B.query(centroids_B[valid_B], k=2)
    spacing_B = np.median(d_B[:, 1]) if len(d_B) > 0 else 1.0

    def get_largest_connected_pair_component(pairs):
        # Prevent scatter: ensure the mapping graph itself is contiguous
        if not pairs: return []
        n_pairs = len(pairs)
        adj_pairs = np.zeros((n_pairs, n_pairs), dtype=bool)
        for i, (uA, uB) in enumerate(pairs):
            for j, (vA, vB) in enumerate(pairs):
                if i != j and (adj_A[uA, vA] or adj_B[uB, vB]):
                    adj_pairs[i, j] = True
        n_comp, comp_labels = connected_components(adj_pairs, directed=False)
        largest = np.bincount(comp_labels).argmax()
        return [pairs[i] for i in np.where(comp_labels == largest)[0]]

    def contiguous_expansion_pass(current_pairs):
        pairs = list(current_pairs)
        while True:
            sA = {p[0] for p in pairs}
            sB = {p[1] for p in pairs}
            
            if len(sA) >= np.sum(valid_A) or len(sB) >= np.sum(valid_B): break
            
            frontier_A = {i for i in range(num_clusters_A) if valid_A[i] and i not in sA and np.any(adj_A[i, list(sA)])}
            frontier_B = {j for j in range(num_clusters_B) if valid_B[j] and j not in sB and np.any(adj_B[j, list(sB)])}
            
            if not frontier_A or not frontier_B:
                break
                
            sA_list, sB_list = list(sA), list(sB)
            bary_A = np.average(centroids_A[sA_list], axis=0, weights=masses_A[sA_list])
            bary_B = np.average(centroids_B[sB_list], axis=0, weights=masses_B[sB_list])

            best_pair = None
            best_score = float('inf')
            
            for pA in frontier_A:
                for pB in frontier_B:
                    if valid_masses[pA, pB] > 0:
                        norm_dA = np.linalg.norm(centroids_A[pA] - bary_A) / spacing_A
                        norm_dB = np.linalg.norm(centroids_B[pB] - bary_B) / spacing_B
                        
                        # Parameter-free scale-normalized expansion criterion
                        # Replaces rigid SVD tracking and favors biologically contiguous pairs
                        score = norm_dA + norm_dB
                        
                        if score < best_score:
                            best_score = score
                            best_pair = (pA, pB)
                            
            if best_pair is not None:
                pairs.append(best_pair)
            else:
                break
        return pairs

    def trim_worst_outlier(current_pairs):
        if len(current_pairs) < 3:
            return current_pairs, False
            
        sA_list = [p[0] for p in current_pairs]
        sB_list = [p[1] for p in current_pairs]
        
        # Calculate full weighted sums to efficiently compute Leave-One-Out (LOO) barycenters
        weights_A = masses_A[sA_list]
        weights_B = masses_B[sB_list]
        
        sum_pos_A = np.sum(centroids_A[sA_list] * weights_A[:, None], axis=0)
        sum_mass_A = np.sum(weights_A)
        
        sum_pos_B = np.sum(centroids_B[sB_list] * weights_B[:, None], axis=0)
        sum_mass_B = np.sum(weights_B)

        residuals_pairs = {}
        for (pA, pB) in current_pairs:
            # Leave-One-Out (LOO) Barycenter Computation:
            # Preventing "Self-Masking / Leverage Skew". If we use the full barycenter, an extreme outlier 
            # pulls the center of mass toward itself. This artificially shrinks its own discrepancy while 
            # inflating the discrepancy of perfectly healthy points on the opposite side of the tissue, 
            # causing the exact wrong pairs to be trimmed! By evaluating against the core EXCLUDING itself, 
            # true outliers are mathematically exposed.
            
            m_pA = masses_A[pA]
            loo_mass_A = sum_mass_A - m_pA
            if loo_mass_A <= 0: continue
            loo_bary_A = (sum_pos_A - centroids_A[pA] * m_pA) / loo_mass_A
            
            m_pB = masses_B[pB]
            loo_mass_B = sum_mass_B - m_pB
            if loo_mass_B <= 0: continue
            loo_bary_B = (sum_pos_B - centroids_B[pB] * m_pB) / loo_mass_B

            # Scale-normalized LOO barycenter mismatch
            norm_dA = np.linalg.norm(centroids_A[pA] - loo_bary_A) / spacing_A  
            norm_dB = np.linalg.norm(centroids_B[pB] - loo_bary_B) / spacing_B  

            # An outlier shows severe dimensional discrepancy from the LOO core center
            discrepancy = abs(norm_dA - norm_dB)
            
            # Parameter-free biological bound: The discrepancy is scaled by the median cluster spacing.
            # A discrepancy > 1.0 means the radial stretch offset between the source and target 
            # is physically larger than 1 entire neighboring cluster. This is the exact mathematical 
            # definition of a "topological fold" (mapping jumping over a valid adjacent neighbor).
            if discrepancy > 1.0:
                residuals_pairs[(pA, pB)] = discrepancy

        if not residuals_pairs:
            return current_pairs, False  # No outliers found

        # Trim the single highest outlier
        worst_pair = max(residuals_pairs, key=residuals_pairs.get)
        current_pairs.remove(worst_pair)
        return current_pairs, True

    # ---------------- Active Contour Refinement Loop ---------------- #
    # Dynamic "Grow-Trim-Grow" (Trimmed Iterative Closest Point using Dynamic Overlapping Subsets)
    # By interleaving trimming and expansion, removing a bad cluster corrects the shadow (rotation matrix).
    # This frequently un-warps the projection, suddenly revealing/aligning new valid clusters 
    # that were previously hidden from the shadow because of the outlier's distortion!
    
    # 1. Initial Growth
    mapped_pairs = contiguous_expansion_pass(mapped_pairs)
    
    visited_states = set()
    visited_states.add(frozenset(mapped_pairs))
    
    # 2. Active Polish Phase
    while len(mapped_pairs) >= 3:
        mapped_pairs, trimmed = trim_worst_outlier(mapped_pairs)
        if trimmed:
            # The shadow matrix (R) snapped back to biological truth!
            # Immediately try to grow into the newly corrected geometry space.
            mapped_pairs = contiguous_expansion_pass(mapped_pairs)
            
            # Danger: Infinite Oscillation Prevention
            # (e.g., Trim removes cluster X -> Shadow shifts -> Expand picks up cluster X -> Shadow shifts back -> Trim removes X)
            current_state = frozenset(mapped_pairs)
            if current_state in visited_states:
                break # We have entered a cyclic loop; the manifold has reached maximum stable equilibrium.
            visited_states.add(current_state)
        else:
            # Convergence: No outliers, and no more clusters physically fit on the frontier.
            break

    # Prevent scatter: Re-extract largest connected component treating pairs strictly as 1:1 atomic units
    if mapped_pairs:
        mapped_pairs = get_largest_connected_pair_component(mapped_pairs)
    
    if not mapped_pairs:
        # Fallback if mapping destroyed
        mapped_pairs = [(best_edge[0][0], best_edge[0][1])] if best_edge else [(np.where(valid_A)[0][0], np.where(valid_B)[0][0])]

    strong_A = [p[0] for p in mapped_pairs]
    strong_B = [p[1] for p in mapped_pairs]

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

