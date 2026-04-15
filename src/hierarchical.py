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

    # Calculate characteristic global spacing for tissue continuity
    from .core import estimate_characteristic_spacing
    char_gap_A = estimate_characteristic_spacing(sliceA, spatial_key=spatial_key) * 3.0
    char_gap_B = estimate_characteristic_spacing(sliceB, spatial_key=spatial_key) * 3.0
    
    # 2. Biological Tissue Adjacency based strictly on cellular continuous touching
    def build_structural_adjacency(coords, labels, valid_mask, char_gap):
        n_clusters = len(valid_mask)
        adj = np.zeros((n_clusters, n_clusters), dtype=np.float64)
        
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < 2:
            return adj
            
        centroids = np.zeros((n_clusters, 2))
        kdtries = {}
        
        for c_id in valid_idx:
            c_coords = coords[labels == c_id]
            centroids[c_id] = c_coords.mean(axis=0)
            kdtries[c_id] = cKDTree(c_coords)

        # Complete tissue graph: Edge exists exclusively if cell boundaries touch
        for i in range(len(valid_idx)):
            for j in range(i+1, len(valid_idx)):
                u, v = valid_idx[i], valid_idx[j]
                
                # Cellular physical gap analysis
                min_dists, _ = kdtries[u].query(coords[labels == v], k=1)
                min_gap = np.min(min_dists)
                
                # Two spatial clusters are continuous tissue if their minimal gap 
                # respects the characteristic bounding resolution of the manifold.
                # If so, the geodesic edge weight is the literal centroid distance.
                if min_gap <= char_gap:
                    dist_uv = np.linalg.norm(centroids[u] - centroids[v])
                    adj[u, v] = dist_uv
                    adj[v, u] = dist_uv
                
        return adj

    adj_A = build_structural_adjacency(coords_A, labels_A, valid_A, char_gap_A)
    adj_B = build_structural_adjacency(coords_B, labels_B, valid_B, char_gap_B)

    # Calculate global continuous tissue length geodesics with physics, NOT unweighted cluster partition hops
    geo_A = dijkstra(adj_A, directed=False, unweighted=False)
    geo_B = dijkstra(adj_B, directed=False, unweighted=False)
    
    # Scale to dimensionless biological structural scale (0.0 -> 1.0 manifold coordinates)
    max_A = np.max(geo_A[~np.isinf(geo_A)]) if np.any(~np.isinf(geo_A)) else 1.0
    max_B = np.max(geo_B[~np.isinf(geo_B)]) if np.any(~np.isinf(geo_B)) else 1.0
    geo_A = geo_A / (max_A + 1e-8)
    geo_B = geo_B / (max_B + 1e-8)

    # 3. Biological Mass Continuity (Use raw OT plan directly without mathematical cliff cuts)
    valid_masses = Pi_cluster.copy()
        
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
    
    # 5. Connect and Polish using Parameter-Free Topology Constraints
    
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

    def get_procrustes_transform(sA_list, sB_list):
        # 2D Rigid Registration (Procrustes) enforces exact size (scale=1.0) and shape (rotation only).
        # Unlike internal geodesics, this strictly prevents tissue from bending/warping.
        if len(sA_list) < 2:
            return np.eye(2), np.zeros(2), np.zeros(2), np.zeros(len(sA_list))
            
        ptsA = centroids_A[sA_list]
        ptsB = centroids_B[sB_list]
        
        # Center points via physical masses to prevent barycenter drift
        cA = np.average(ptsA, axis=0, weights=masses_A[sA_list])
        cB = np.average(ptsB, axis=0, weights=masses_B[sB_list])
        
        A_centered = ptsA - cA
        B_centered = ptsB - cB
        
        # Kabsch Algorithm for optimal biological orientation
        H = A_centered.T @ np.diag(masses_A[sA_list]) @ B_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Eliminate non-biological reflections (tissue cannot flip)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        # Predict physical position of A mapped onto B
        ptsA_aligned = A_centered @ R.T + cB
        
        # Absolute Euclidean displacement mismatch
        residuals = np.linalg.norm(ptsA_aligned - ptsB, axis=1)
        
        return R, cA, cB, residuals

    def anneal_suboptimal_pairs(current_pairs):
        # Topological "Sliding": Replaces mapped peripheral nodes with unmapped nodes 
        # strictly if they offer a computationally tighter 2D Rigid fit.
        pairs = list(current_pairs)
        improved = True
        
        while improved:
            improved = False
            sA_list = [p[0] for p in pairs]
            sB_list = [p[1] for p in pairs]
            
            if len(pairs) < 3: 
                break
                
            R, cA, cB, residuals = get_procrustes_transform(sA_list, sB_list)
            
            mapped_A = {p[0]: True for p in pairs}
            mapped_B = {p[1]: True for p in pairs}
            
            for i, (pA, pB) in enumerate(pairs):
                # Temporarily remove pair i to test alternatives against the fixed core
                rest_sA = [x for j, x in enumerate(sA_list) if j != i]
                rest_sB = [x for j, x in enumerate(sB_list) if j != i]
                
                # Transform vector from rest core to the candidate point
                pred_pB = (centroids_A[pA] - cA) @ R.T + cB
                pred_pA = (centroids_B[pB] - cB) @ R + cA
                
                best_new_pair = None
                best_err = residuals[i]
                
                # Can pA map to a better unmapped B?
                for alt_B in range(num_clusters_B):
                    if valid_B[alt_B] and alt_B not in mapped_B and valid_masses[pA, alt_B] > 0:
                        if adj_B[alt_B, pB] > 0: # Must physically sit nearby
                            alt_err = np.linalg.norm(pred_pB - centroids_B[alt_B])
                            if alt_err < best_err:
                                best_err = alt_err
                                best_new_pair = (pA, alt_B)
                                
                # Can pB map to a better unmapped A?
                for alt_A in range(num_clusters_A):
                    if valid_A[alt_A] and alt_A not in mapped_A and valid_masses[alt_A, pB] > 0:
                        if adj_A[alt_A, pA] > 0:
                            alt_err = np.linalg.norm(pred_pA - centroids_A[alt_A])
                            if alt_err < best_err:
                                best_err = alt_err
                                best_new_pair = (alt_A, pB)
                                
                if best_new_pair:
                    pairs[i] = best_new_pair
                    improved = True
                    break
        
        return pairs

    def contiguous_expansion_pass(current_pairs):
        # 1. 2D Euclidean Growth: Expand the active contour layer by layer 
        # strictly maintaining the rigid shape constraints of the core.
        pairs = list(current_pairs)
        improved = True
        
        while improved:
            improved = False
            sA_list = [p[0] for p in pairs]
            sB_list = [p[1] for p in pairs]
            
            mapped_A = {p[0]: True for p in pairs}
            mapped_B = {p[1]: True for p in pairs}
            
            R, cA, cB, _ = get_procrustes_transform(sA_list, sB_list)
            
            best_pair = None
            best_score = -1.0
            
            for uA in range(num_clusters_A):
                if not valid_A[uA] or uA in mapped_A: continue
                
                # Must be topologically adjacent to the growing footprint
                if not np.any(adj_A[uA, sA_list] > 0): continue
                
                pred_uB = (centroids_A[uA] - cA) @ R.T + cB
                
                for uB in range(num_clusters_B):
                    if not valid_B[uB] or uB in mapped_B: continue
                    if not np.any(adj_B[uB, sB_list] > 0): continue
                    
                    if valid_masses[uA, uB] == 0: continue
                    
                    err = np.linalg.norm(pred_uB - centroids_B[uB])
                    
                    # Parameter-Free Biological Ground: Limit deformation
                    # Two matched clusters are a biological size/shape anomaly if their rigidly 
                    # predicted overlap deviates by more than their intrinsic tissue cellular spread.
                    allowed_tolerance = intra_A[uA] + intra_B[uB] + (char_gap_A + char_gap_B) / 2.0
                    
                    if err <= allowed_tolerance:
                        score = valid_masses[uA, uB] / (err + 1e-3)
                        if score > best_score:
                            best_score = score
                            best_pair = (uA, uB)
                            
            if best_pair:
                pairs.append(best_pair)
                improved = True
                
        return pairs

    def trim_worst_outlier(current_pairs):
        if len(current_pairs) < 3:
            return current_pairs, False
            
        sA_list = [p[0] for p in current_pairs]
        sB_list = [p[1] for p in current_pairs]
        
        R, cA, cB, residuals = get_procrustes_transform(sA_list, sB_list)

        worst_idx = int(np.argmax(residuals))
        worst_err = residuals[worst_idx]
        worst_pair = current_pairs[worst_idx]
        pA, pB = worst_pair
        
        # Parameter-Free Biological Ground:
        # A rigidly aligned structure implies same size and same shape natively.
        # If the worst residual exceeds the biological cellular radii combo, it's a structural tear.
        allowed_tolerance = intra_A[pA] + intra_B[pB] + (char_gap_A + char_gap_B) / 2.0
        
        if worst_err > allowed_tolerance:
            current_pairs.pop(worst_idx)
            return current_pairs, True
            
        return current_pairs, False
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
        # Annealing Sliding Pass: Break sub-optimal peripheral locks 
        # and recruit geometrically tighter unmapped nodes closer to the barycenter
        mapped_pairs = anneal_suboptimal_pairs(mapped_pairs)

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

