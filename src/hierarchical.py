import logging
import numpy as np
import scipy.sparse as sp

from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance_matrix, Delaunay, QhullError
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
        
    C_A_norm = C_A / (np.max(C_A) + 1e-8)
    C_B_norm = C_B / (np.max(C_B) + 1e-8)
    M_norm = M_cluster / (np.max(M_cluster) + 1e-8)
    
    logging.info("Running Unbalanced FGW...")
    # reg_marginals controls how much marginal relaxation is allowed (lower = more mass can be dropped)
    pi_samp, pi_feat, log = fused_unbalanced_gromov_wasserstein(
        Cx=C_A_norm, Cy=C_B_norm, wx=p_A, wy=p_B, M=M_norm, alpha=alpha, reg_marginals=reg_m, log=True, max_iter=500
    )
    
    return pi_samp


def _encode_labels_by_unique_order(labels, expected_clusters):
    """
    Encodes arbitrary cluster labels into the same sorted unique order used by cluster feature extraction.
    """
    unique_labels = np.unique(labels)
    if unique_labels.size != expected_clusters:
        raise ValueError(
            f"Expected {expected_clusters} clusters from labels, found {unique_labels.size}."
        )

    encoded = np.searchsorted(unique_labels, labels)
    if not np.array_equal(unique_labels[encoded], labels):
        raise ValueError("Cluster labels could not be encoded into transport-matrix order.")

    return encoded


def _compute_cluster_centroids(coords, labels, num_clusters):
    centroids = np.zeros((num_clusters, coords.shape[1]), dtype=float)
    valid = np.zeros(num_clusters, dtype=bool)

    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        if np.any(mask):
            centroids[cluster_id] = coords[mask].mean(axis=0)
            valid[cluster_id] = True

    return centroids, valid


def _natural_spacing(points):
    if points.shape[0] < 2:
        return 1.0

    dists, _ = cKDTree(points).query(points, k=2)
    scale = float(np.median(dists[:, 1]))
    if scale > 0:
        return scale

    span = float(np.ptp(points, axis=0).max())
    return span if span > 0 else 1.0


def _get_max_edge_len(coords):
    if coords.shape[0] < 2:
        return float("inf")

    k = min(6, coords.shape[0])
    dists, _ = cKDTree(coords).query(coords, k=k)
    neighbor_dists = dists[:, -1] if dists.ndim > 1 else dists
    return float(np.median(neighbor_dists) * 5.0)


def _build_cluster_border_adjacency(coords, labels, num_clusters):
    """
    Builds a cluster adjacency graph from local cell neighborhoods.
    Uses Delaunay edges when possible and falls back to kNN for degenerate geometries.
    """
    adj = np.zeros((num_clusters, num_clusters), dtype=bool)
    np.fill_diagonal(adj, True)

    if coords.shape[0] < 2:
        return adj

    max_edge_len = _get_max_edge_len(coords)
    candidate_edges = set()

    if coords.shape[0] >= 3:
        try:
            tri = Delaunay(coords)
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        edge = tuple(sorted((int(simplex[i]), int(simplex[j]))))
                        candidate_edges.add(edge)
        except QhullError:
            pass

    if not candidate_edges:
        k = min(7, coords.shape[0])
        _, neighbors = cKDTree(coords).query(coords, k=k)
        neighbors = np.atleast_2d(neighbors)
        for i in range(coords.shape[0]):
            for j in neighbors[i, 1:]:
                if i == j:
                    continue
                candidate_edges.add(tuple(sorted((int(i), int(j)))))

    for i, j in candidate_edges:
        if np.linalg.norm(coords[i] - coords[j]) >= max_edge_len:
            continue

        label_i = labels[i]
        label_j = labels[j]
        if label_i == label_j:
            continue

        adj[label_i, label_j] = True
        adj[label_j, label_i] = True

    return adj


def extract_continuous_macro_section(sliceA, sliceB, labels_A, labels_B, Pi_cluster, spatial_key='spatial'):
    """
    Identifies a spatially continuous macro section using a one-to-one, symmetric cluster matching.

    Returns:
        idx_A, idx_B: selected cell indices in both slices
        dist_A, dist_B: distance-to-selected-region arrays for all cells
        matched_pairs: (K, 2) array of selected cluster matches using Pi_cluster row/column indices
    """
    N, M = sliceA.shape[0], sliceB.shape[0]

    empty_pairs = np.empty((0, 2), dtype=int)
    total_mass = float(np.sum(Pi_cluster))
    if total_mass <= 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), empty_pairs

    coords_A = np.asarray(sliceA.obsm[spatial_key], dtype=float)
    coords_B = np.asarray(sliceB.obsm[spatial_key], dtype=float)

    num_clusters_A, num_clusters_B = Pi_cluster.shape
    labels_A_idx = _encode_labels_by_unique_order(labels_A, num_clusters_A)
    labels_B_idx = _encode_labels_by_unique_order(labels_B, num_clusters_B)

    centroids_A, valid_A = _compute_cluster_centroids(coords_A, labels_A_idx, num_clusters_A)
    centroids_B, valid_B = _compute_cluster_centroids(coords_B, labels_B_idx, num_clusters_B)

    adj_A = _build_cluster_border_adjacency(coords_A, labels_A_idx, num_clusters_A)
    adj_B = _build_cluster_border_adjacency(coords_B, labels_B_idx, num_clusters_B)

    valid_idx_A = np.flatnonzero(valid_A)
    valid_idx_B = np.flatnonzero(valid_B)
    if valid_idx_A.size == 0 or valid_idx_B.size == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), empty_pairs

    eps = 1e-12
    row_mass = Pi_cluster.sum(axis=1, keepdims=True)
    col_mass = Pi_cluster.sum(axis=0, keepdims=True)
    mass_scale = float(np.max(Pi_cluster))
    if mass_scale <= eps:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), empty_pairs

    mass_norm = Pi_cluster / (mass_scale + eps)
    row_conf = np.divide(
        Pi_cluster,
        row_mass,
        out=np.zeros_like(Pi_cluster, dtype=float),
        where=row_mass > eps,
    )
    col_conf = np.divide(
        Pi_cluster,
        col_mass,
        out=np.zeros_like(Pi_cluster, dtype=float),
        where=col_mass > eps,
    )
    pair_score = mass_norm * np.sqrt(row_conf * col_conf)

    score_sub = pair_score[np.ix_(valid_idx_A, valid_idx_B)]
    row_ind, col_ind = linear_sum_assignment(-score_sub)
    assigned_pairs = np.column_stack((valid_idx_A[row_ind], valid_idx_B[col_ind]))
    assigned_scores = score_sub[row_ind, col_ind]
    assigned_masses = Pi_cluster[assigned_pairs[:, 0], assigned_pairs[:, 1]]

    positive_mask = assigned_masses > eps
    if np.any(positive_mask):
        assigned_pairs = assigned_pairs[positive_mask]
        assigned_scores = assigned_scores[positive_mask]
        assigned_masses = assigned_masses[positive_mask]
    elif assigned_pairs.size == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), empty_pairs

    num_pairs = assigned_pairs.shape[0]
    if num_pairs == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), empty_pairs

    match_adj = np.zeros((num_pairs, num_pairs), dtype=bool)
    np.fill_diagonal(match_adj, True)
    for i in range(num_pairs):
        u1, v1 = assigned_pairs[i]
        for j in range(i + 1, num_pairs):
            u2, v2 = assigned_pairs[j]
            if adj_A[u1, u2] and adj_B[v1, v2]:
                match_adj[i, j] = True
                match_adj[j, i] = True

    _, comp_labels = connected_components(match_adj, directed=False)
    selected_pair_ids = None
    best_signature = None
    for comp_id in np.unique(comp_labels):
        component_ids = np.flatnonzero(comp_labels == comp_id)
        signature = (
            float(np.sum(assigned_scores[component_ids])),
            float(np.sum(assigned_masses[component_ids])),
            int(component_ids.size),
        )
        if best_signature is None or signature > best_signature:
            best_signature = signature
            selected_pair_ids = component_ids.tolist()

    if not selected_pair_ids:
        best_pair_id = int(np.argmax(assigned_scores))
        selected_pair_ids = [best_pair_id]

    target_count = min(
        num_pairs,
        max(1, int(np.ceil(min(np.sum(valid_A), np.sum(valid_B)) / 3.0))),
    )
    scale_A = _natural_spacing(centroids_A[valid_A])
    scale_B = _natural_spacing(centroids_B[valid_B])
    selected_pair_set = set(selected_pair_ids)

    logging.info(
        "[HOT] Topological Extension: Expanding one-to-one cluster pairs until %d pairs are selected.",
        target_count,
    )

    while len(selected_pair_ids) < target_count:
        selected_A = assigned_pairs[selected_pair_ids, 0]
        selected_B = assigned_pairs[selected_pair_ids, 1]
        bary_A = centroids_A[selected_A].mean(axis=0)
        bary_B = centroids_B[selected_B].mean(axis=0)

        best_candidate_id = None
        best_candidate_signature = None
        for candidate_id in range(num_pairs):
            if candidate_id in selected_pair_set:
                continue

            u, v = assigned_pairs[candidate_id]
            is_contiguous = np.any(adj_A[selected_A, u] & adj_B[selected_B, v])
            if not is_contiguous:
                continue

            dist_A = np.linalg.norm(centroids_A[u] - bary_A) / scale_A
            dist_B = np.linalg.norm(centroids_B[v] - bary_B) / scale_B
            compactness = 0.5 * (dist_A + dist_B)
            candidate_signature = (
                float(assigned_scores[candidate_id] / (1.0 + compactness)),
                float(assigned_scores[candidate_id]),
                float(assigned_masses[candidate_id]),
            )
            if best_candidate_signature is None or candidate_signature > best_candidate_signature:
                best_candidate_signature = candidate_signature
                best_candidate_id = candidate_id

        if best_candidate_id is None:
            break

        selected_pair_ids.append(best_candidate_id)
        selected_pair_set.add(best_candidate_id)

    matched_pairs = assigned_pairs[np.array(selected_pair_ids, dtype=int)]
    strong_A = matched_pairs[:, 0]
    strong_B = matched_pairs[:, 1]

    idx_A = np.flatnonzero(np.isin(labels_A_idx, strong_A))
    idx_B = np.flatnonzero(np.isin(labels_B_idx, strong_B))
    if idx_A.size == 0 or idx_B.size == 0:
        return np.arange(N), np.arange(M), np.zeros(N), np.zeros(M), empty_pairs

    tree_A = cKDTree(coords_A[idx_A])
    tree_B = cKDTree(coords_B[idx_B])
    dist_A, _ = tree_A.query(coords_A)
    dist_B, _ = tree_B.query(coords_B)

    return idx_A, idx_B, dist_A, dist_B, matched_pairs
