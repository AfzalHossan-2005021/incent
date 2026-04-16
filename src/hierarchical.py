import logging
import numpy as np
import scipy.sparse as sp

from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance_matrix, Delaunay
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


def compute_pairwise_mutual_information_contribution(pi):
    """
    Per-pair mutual-information contribution under a size-preserving independence null.

    Each entry quantifies how much a cluster-pair in the coarse transport plan is
    enriched above the random coupling induced by the transport marginals.
    """
    total_mass = np.sum(pi)
    if total_mass <= 0:
        return np.zeros_like(pi, dtype=np.float64)

    P = pi / total_mass
    row_mass = P.sum(axis=1, keepdims=True)
    col_mass = P.sum(axis=0, keepdims=True)
    expected = row_mass @ col_mass

    contrib = np.zeros_like(P, dtype=np.float64)
    positive_mass = P > 0
    contrib[positive_mass] = P[positive_mass] * np.log(
        (P[positive_mass] + 1e-12) / (expected[positive_mass] + 1e-12)
    )
    return contrib


def select_initial_match_component(matches, match_adj, match_scores):
    """
    Select the strongest initial motif in the match graph.

    Preference order:
    1) triangle (three mutually adjacent one-to-one matched pairs)
    2) edge (two adjacent one-to-one matched pairs)
    3) singleton (single best matched pair)
    """
    num_matches = match_adj.shape[0]
    if num_matches == 0:
        return np.array([], dtype=int)

    match_scores = np.asarray(match_scores, dtype=np.float64)

    # Prefer the strongest triangle clique.
    best_triangle = None
    best_triangle_score = -np.inf
    for i in range(num_matches):
        for j in range(i + 1, num_matches):
            if not match_adj[i, j]:
                continue
            for k in range(j + 1, num_matches):
                if match_adj[i, k] and match_adj[j, k]:
                    tri_A = {matches[i][0], matches[j][0], matches[k][0]}
                    tri_B = {matches[i][1], matches[j][1], matches[k][1]}
                    if len(tri_A) < 3 or len(tri_B) < 3:
                        continue
                    score = match_scores[i] + match_scores[j] + match_scores[k]
                    if score > best_triangle_score:
                        best_triangle_score = score
                        best_triangle = np.array([i, j, k], dtype=int)

    if best_triangle is not None:
        return best_triangle

    # Fall back to the strongest supported edge.
    best_edge = None
    best_edge_score = -np.inf
    for i in range(num_matches):
        for j in range(i + 1, num_matches):
            if match_adj[i, j]:
                if matches[i][0] == matches[j][0] or matches[i][1] == matches[j][1]:
                    continue
                score = match_scores[i] + match_scores[j]
                if score > best_edge_score:
                    best_edge_score = score
                    best_edge = np.array([i, j], dtype=int)

    if best_edge is not None:
        return best_edge

    # Final fall back: single best matched pair.
    return np.array([int(np.argmax(match_scores))], dtype=int)


def fit_rigid_transform(src, tgt):
    """
    Fit a rigid transform in row-vector form: src @ R.T + t ~= tgt.
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)

    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("Rigid transform expects two arrays of shape (n, 2).")

    if src.shape[0] == 0:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)

    if src.shape[0] == 1:
        R = np.eye(2, dtype=np.float64)
        t = tgt_center - src_center
        return R, t

    src_centered = src - src_center
    tgt_centered = tgt - tgt_center

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_center - src_center @ R.T
    return R, t


def map_sliceB_to_sliceA_frame(centroids_A, centroids_B, matched_pairs):
    """
    Align slice B centroids into the current slice A frame using matched cluster pairs.
    """
    if len(matched_pairs) == 0:
        return centroids_B.copy(), np.zeros(2, dtype=np.float64)

    anchor_A = centroids_A[[a for a, _ in matched_pairs]]
    anchor_B = centroids_B[[b for _, b in matched_pairs]]
    R, t = fit_rigid_transform(anchor_B, anchor_A)
    aligned_B = centroids_B @ R.T + t
    origin = anchor_A.mean(axis=0)
    return aligned_B, origin


def angular_difference(theta_a, theta_b):
    """
    Smallest absolute angular difference between two angles.
    """
    return abs(np.arctan2(np.sin(theta_a - theta_b), np.cos(theta_a - theta_b)))


def pair_has_paired_frontier_support(pA, pB, matched_pairs, adj_A, adj_B):
    """
    A candidate extension must touch at least one already matched pair in both slices.
    """
    return any(adj_A[pA, a] and adj_B[pB, b] for a, b in matched_pairs)


def score_candidate_pair(pA, pB, centroids_A, aligned_centroids_B, origin, mi_contrib):
    """
    Parameter-free geometric ranking of a candidate pair in the shared frame.

    The primary score is the positional discrepancy in the common coordinate
    system; radial and angular discrepancies serve as interpretable tie-breakers.
    """
    rel_A = centroids_A[pA] - origin
    rel_B = aligned_centroids_B[pB] - origin

    radius_A = float(np.linalg.norm(rel_A))
    radius_B = float(np.linalg.norm(rel_B))
    radial_gap = abs(radius_A - radius_B)

    if radius_A > 1e-12 and radius_B > 1e-12:
        angle_A = float(np.arctan2(rel_A[1], rel_A[0]))
        angle_B = float(np.arctan2(rel_B[1], rel_B[0]))
        angle_gap = angular_difference(angle_A, angle_B)
    else:
        angle_gap = 0.0

    coord_gap = float(np.linalg.norm(rel_A - rel_B))
    return (coord_gap, radial_gap, angle_gap, -float(mi_contrib[pA, pB]))


def select_next_reciprocal_pair(frontier_A, frontier_B, matched_pairs, adj_A, adj_B, centroids_A, centroids_B, mi_contrib):
    """
    Select the next one-to-one frontier pair by reciprocal best match in the shared frame.
    """
    if len(frontier_A) == 0 or len(frontier_B) == 0:
        return None

    aligned_centroids_B, origin = map_sliceB_to_sliceA_frame(centroids_A, centroids_B, matched_pairs)

    best_for_A = {}
    best_rank_for_A = {}
    for pA in frontier_A:
        best_pair = None
        best_rank = None
        for pB in frontier_B:
            if mi_contrib[pA, pB] <= 0:
                continue
            if not pair_has_paired_frontier_support(pA, pB, matched_pairs, adj_A, adj_B):
                continue

            rank = score_candidate_pair(pA, pB, centroids_A, aligned_centroids_B, origin, mi_contrib)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_pair = pB

        if best_pair is not None:
            best_for_A[pA] = best_pair
            best_rank_for_A[pA] = best_rank

    best_for_B = {}
    for pB in frontier_B:
        best_pair = None
        best_rank = None
        for pA in frontier_A:
            if mi_contrib[pA, pB] <= 0:
                continue
            if not pair_has_paired_frontier_support(pA, pB, matched_pairs, adj_A, adj_B):
                continue

            rank = score_candidate_pair(pA, pB, centroids_A, aligned_centroids_B, origin, mi_contrib)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_pair = pA

        if best_pair is not None:
            best_for_B[pB] = best_pair

    reciprocal_pairs = []
    for pA, pB in best_for_A.items():
        if best_for_B.get(pB) == pA:
            reciprocal_pairs.append((best_rank_for_A[pA], pA, pB))

    if not reciprocal_pairs:
        return None

    reciprocal_pairs.sort(key=lambda x: x[0])
    _, pA, pB = reciprocal_pairs[0]
    return (pA, pB)


def extract_continuous_macro_section(sliceA, sliceB, labels_A, labels_B, Pi_cluster, spatial_key='spatial'):
    """
    Identifies a one-to-one, co-contiguous macro-section from the clustering alignment.
    """
    N, M = sliceA.shape[0], sliceB.shape[0]
    
    total_mass = np.sum(Pi_cluster)
    if total_mass == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), np.arange(N), np.arange(M), []
        
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

    # 3. Select cluster-pairs enriched above the size-preserving independence null
    mi_contrib = compute_pairwise_mutual_information_contribution(Pi_cluster)
    positive_flat_idx = np.flatnonzero(mi_contrib > 0)
    if positive_flat_idx.size > 0:
        selected_flat_idx = positive_flat_idx[np.argsort(mi_contrib.ravel()[positive_flat_idx])[::-1]]
    else:
        flat_pi = Pi_cluster.ravel()
        selected_flat_idx = np.argsort(flat_pi)[::-1]

    matches = []
    match_scores = []
    for idx in selected_flat_idx:
        u, v = np.unravel_index(idx, Pi_cluster.shape)
        if valid_A[u] and valid_B[v]:
            matches.append((u, v))
            match_scores.append(mi_contrib[u, v])
            
    num_matches = len(matches)
    if num_matches == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), np.arange(N), np.arange(M), []
        
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

    match_scores = np.asarray(match_scores, dtype=np.float64)
    largest_match_indices = select_initial_match_component(matches, match_adj, match_scores)
    matched_pairs = [matches[i] for i in largest_match_indices]
    matched_A = {a for a, _ in matched_pairs}
    matched_B = {b for _, b in matched_pairs}
    
    core_cells_A = np.where(np.isin(labels_A, list(matched_A)))[0]
    core_cells_B = np.where(np.isin(labels_B, list(matched_B)))[0]

    if len(core_cells_A) == 0 or len(core_cells_B) == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), np.arange(N), np.arange(M), matched_pairs

    initial_idx_A = core_cells_A.copy()
    initial_idx_B = core_cells_B.copy()

    motif_name = {1: "singleton", 2: "edge", 3: "triangle"}.get(len(matched_pairs), "component")
    print(
        f"[HOT] Topological Extension: starting from a {motif_name} and extending one-to-one "
        "until one slice frontier is exhausted or no reciprocal biologically supported pair remains."
    )

    # 5. Cluster-level one-to-one topological extension in the shared frame
    while True:
        matched_A_list = sorted(matched_A)
        matched_B_list = sorted(matched_B)

        neighbors_A = np.where(np.any(adj_A[matched_A_list, :], axis=0))[0]
        neighbors_B = np.where(np.any(adj_B[matched_B_list, :], axis=0))[0]

        frontier_A = [c for c in neighbors_A if valid_A[c] and c not in matched_A]
        frontier_B = [c for c in neighbors_B if valid_B[c] and c not in matched_B]

        if not frontier_A or not frontier_B:
            print("[HOT] Topological Extension stopped: one slice ran out of unmatched frontier clusters.")
            break

        next_pair = select_next_reciprocal_pair(
            frontier_A=frontier_A,
            frontier_B=frontier_B,
            matched_pairs=matched_pairs,
            adj_A=adj_A,
            adj_B=adj_B,
            centroids_A=centroids_A,
            centroids_B=centroids_B,
            mi_contrib=mi_contrib
        )

        if next_pair is None:
            print("[HOT] Topological Extension stopped: no reciprocal, topologically supported, MI-enriched pair remained.")
            break

        matched_pairs.append(next_pair)
        matched_A.add(next_pair[0])
        matched_B.add(next_pair[1])

    # Extract final cells based strictly on expanded cluster membership
    final_A = sorted(matched_A)
    final_B = sorted(matched_B)
    idx_A = np.where(np.isin(labels_A, final_A))[0]
    idx_B = np.where(np.isin(labels_B, final_B))[0]

    # Compute physical distances from the newly expanded core to all cells
    core_coords_A = coords_A[idx_A]
    core_coords_B = coords_B[idx_B]

    tree_A = cKDTree(core_coords_A)
    tree_B = cKDTree(core_coords_B)
    
    dist_A, _ = tree_A.query(coords_A)
    dist_B, _ = tree_B.query(coords_B)
    
    return idx_A, idx_B, dist_A, dist_B, initial_idx_A, initial_idx_B, matched_pairs

