import logging
import numpy as np
import scipy.sparse as sp

from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance_matrix, Delaunay
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata
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
        mu_struct: np.ndarray (C, T * 2) invariant structural descriptor made of
            per-cell-type abundance (m=0) and bilateral anisotropy magnitude
            (|m=2|) measured relative to the cluster centroid
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

    # Two invariant channels per cell type:
    # m=0 abundance and |m=2| bilateral anisotropy magnitude.
    mu_struct = np.zeros((n_clusters, n_types * 2))
    
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
            
            # --- Invariant intrinsic cluster context ---
            # Evaluates cell localization relative to the cluster centroid using
            # abundance and bilateral anisotropy magnitude only.
            rel_coords = coords[mask] - centroids[c_i]
            thetas = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
            
            local_feat = np.zeros((n_types, 2))
            c_types_idx = np.array(mapped_types)
            
            local_feat[:, 0] = counts

            ang = 2.0 * thetas
            real = np.bincount(c_types_idx, weights=np.cos(ang), minlength=n_types)
            imag = np.bincount(c_types_idx, weights=np.sin(ang), minlength=n_types)
            local_feat[:, 1] = np.sqrt(real ** 2 + imag ** 2)
            
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
        mu_struct_A: np.ndarray (C_A, K) invariant structural features for slice A
        mu_expr_B: np.ndarray (C_B, D) mean expression for slice B
        mu_struct_B: np.ndarray (C_B, K) invariant structural features for slice B
        beta: weight for structural distance (expression distance is 1 - beta)
        
    Returns:
        M_cluster: np.ndarray (C_A, C_B) cost matrix
    """

    if(beta > 1.0 or beta < 0.0):
        raise ValueError("Beta must be between 0 and 1.")
    
    # Cosine distance for continuous expression
    M_expr = cosine_distances(mu_expr_A, mu_expr_B)
            
    # Jensen-Shannon for nonnegative invariant structural descriptors
    M_struct = np.zeros((mu_struct_A.shape[0], mu_struct_B.shape[0]))
    for i in range(mu_struct_A.shape[0]):
        for j in range(mu_struct_B.shape[0]):
            # The descriptors are normalized nonnegative summaries over cell-type-specific structure.
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


def compute_pairwise_log_enrichment(pi):
    """
    Log-enrichment of each cluster-pair under a size-preserving independence null.

    The null model preserves the coarse transport marginals and therefore answers:
    "is this pair matched more strongly than expected from cluster sizes alone?"
    Positive values indicate enrichment above that null and serve as the
    parameter-free entry criterion for candidate anchor pairs.
    """
    total_mass = np.sum(pi)
    if total_mass <= 0:
        return np.full_like(pi, -np.inf, dtype=np.float64)

    P = pi / total_mass
    row_mass = P.sum(axis=1, keepdims=True)
    col_mass = P.sum(axis=0, keepdims=True)
    expected = row_mass @ col_mass

    log_enrichment = np.full_like(P, -np.inf, dtype=np.float64)
    positive_mass = P > 0
    log_enrichment[positive_mass] = np.log(
        (P[positive_mass] + 1e-12) / (expected[positive_mass] + 1e-12)
    )
    return log_enrichment


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
    log_enrichment = compute_pairwise_log_enrichment(pi)

    contrib = np.zeros_like(P, dtype=np.float64)
    positive_mass = np.isfinite(log_enrichment)
    contrib[positive_mass] = P[positive_mass] * log_enrichment[positive_mass]
    return contrib


def build_cluster_contact_graph(coords, labels, valid_mask):
    """
    Construct a parameter-light cluster contact graph from physical cluster geometry.

    The graph is deliberately based on verified spatial contact rather than raw
    centroid proximity. Candidate edges are proposed by the centroid Delaunay
    triangulation when possible and then retained only when the minimum
    inter-cluster gap is compatible with the intrinsic within-cluster spacing of
    the two clusters. This prevents long-range shortcuts between symmetric but
    non-overlapping tissue compartments.

    Returns
    -------
    adjacency:
        Boolean contact graph with self-loops on the diagonal.
    edge_lengths:
        Symmetric matrix containing centroid-to-centroid edge lengths for the
        retained contacts and zeros elsewhere.
    """
    n_clusters = len(valid_mask)
    adjacency = np.zeros((n_clusters, n_clusters), dtype=bool)
    edge_lengths = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    np.fill_diagonal(adjacency, True)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return adjacency, edge_lengths

    centroids = np.zeros((len(valid_idx), 2), dtype=np.float64)
    centroid_lookup = {}
    kdtrees = {}
    intra_dists = {}

    for i, c_id in enumerate(valid_idx):
        c_coords = coords[labels == c_id]
        centroids[i] = c_coords.mean(axis=0)
        centroid_lookup[c_id] = centroids[i]

        tree = cKDTree(c_coords)
        kdtrees[c_id] = tree

        if len(c_coords) > 1:
            d, _ = tree.query(c_coords, k=2)
            intra_dists[c_id] = float(np.percentile(d[:, 1], 99))
        else:
            intra_dists[c_id] = 0.0

    if len(valid_idx) >= 3:
        try:
            tri = Delaunay(centroids)
            candidate_edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        u = valid_idx[simplex[i]]
                        v = valid_idx[simplex[j]]
                        candidate_edges.add((min(u, v), max(u, v)))
        except Exception:
            candidate_edges = {
                (min(u, v), max(u, v))
                for i, u in enumerate(valid_idx)
                for v in valid_idx[i + 1:]
            }
    else:
        u, v = valid_idx[0], valid_idx[1]
        candidate_edges = {(min(u, v), max(u, v))}

    for u, v in candidate_edges:
        min_dists, _ = kdtrees[u].query(coords[labels == v], k=1)
        min_gap = float(np.min(min_dists))

        if min_gap <= (intra_dists[u] + intra_dists[v]):
            edge_len = float(np.linalg.norm(centroid_lookup[u] - centroid_lookup[v]))
            adjacency[u, v] = True
            adjacency[v, u] = True
            edge_lengths[u, v] = edge_len
            edge_lengths[v, u] = edge_len

    return adjacency, edge_lengths


def normalize_contact_graph(edge_lengths):
    """
    Normalize cluster-contact edge lengths by the intrinsic cluster spacing.

    The median retained edge length is used as the characteristic graph unit so
    that geodesic distances and rigid residuals become dimensionless and
    comparable across slices with different sampling density.
    """
    positive = edge_lengths[edge_lengths > 0]
    scale = float(np.median(positive)) if positive.size > 0 else 1.0
    return edge_lengths / max(scale, 1e-12), max(scale, 1e-12)


def compute_graph_geodesics(edge_lengths):
    """
    Compute all-pairs geodesic distances on the normalized cluster contact graph.

    Disconnected distances are mapped to a large finite penalty so they remain
    usable in downstream arithmetic without introducing special-case branches.
    """
    distances = dijkstra(sp.csr_matrix(edge_lengths), directed=False)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        distances = np.full_like(edge_lengths, 2.0, dtype=np.float64)
        np.fill_diagonal(distances, 0.0)
        return distances

    finite_nonzero = finite[finite > 0]
    fallback = float(np.max(finite_nonzero)) * 2.0 if finite_nonzero.size > 0 else 2.0
    distances[np.isinf(distances)] = fallback
    np.fill_diagonal(distances, 0.0)
    return distances


def compute_cluster_cell_type_histograms(adata, labels, n_clusters, label_key="cell_type_annot", all_types=None):
    """
    Compute within-cluster cell-type compositions for biologically grounded matching.

    These histograms act as coarse "microenvironment identities" that are much
    more stable than raw centroid geometry and help separate nearby symmetric
    compartments whose local molecular composition differs subtly.
    """
    cell_types = adata.obs[label_key].astype(str).to_numpy()
    if all_types is None:
        all_types = np.array(sorted(np.unique(cell_types)), dtype=str)
    else:
        all_types = np.array(all_types, dtype=str)

    type_to_idx = {ct: i for i, ct in enumerate(all_types)}
    hist = np.zeros((n_clusters, len(all_types)), dtype=np.float64)

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        mapped = [type_to_idx[x] for x in cell_types[mask] if x in type_to_idx]
        if not mapped:
            continue
        counts = np.bincount(mapped, minlength=len(all_types)).astype(np.float64)
        hist[cluster_id] = counts / max(counts.sum(), 1.0)

    return hist, all_types


def compute_cluster_context_features(cluster_hist, adjacency):
    """
    Build cluster context descriptors from own composition and adjacent composition.

    The descriptor concatenates the within-cluster composition with the mean
    composition of directly contacting neighboring clusters, yielding a local
    niche signature that is insensitive to rigid motion yet informative for
    symmetry resolution.
    """
    n_clusters = cluster_hist.shape[0]
    context = np.zeros_like(cluster_hist, dtype=np.float64)

    for i in range(n_clusters):
        neighbors = np.where(adjacency[i])[0]
        neighbors = neighbors[neighbors != i]

        aggregate = cluster_hist[i].copy()
        if neighbors.size > 0:
            aggregate += cluster_hist[neighbors].mean(axis=0)

        total = aggregate.sum()
        if total > 0:
            context[i] = aggregate / total

    features = np.concatenate([cluster_hist, context], axis=1)
    row_sums = features.sum(axis=1, keepdims=True)
    nz = row_sums[:, 0] > 0
    features[nz] /= row_sums[nz]
    return features


def fit_weighted_rigid_transform(source_points, target_points, weights=None):
    """
    Fit the best rigid transform mapping source points to target points.

    Reflection is allowed because cross-slice alignment must handle arbitrary
    flips. When only one matched pair is available, the transform reduces to a
    pure translation anchored at that pair.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.shape[0] == 0:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    if source_points.shape[0] == 1:
        return np.eye(2, dtype=np.float64), target_points[0] - source_points[0]

    if weights is None:
        weights = np.ones(source_points.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    weights = weights / max(weights.sum(), 1e-12)
    source_center = (weights[:, None] * source_points).sum(axis=0)
    target_center = (weights[:, None] * target_points).sum(axis=0)

    source_centered = source_points - source_center
    target_centered = target_points - target_center

    H = source_centered.T @ (weights[:, None] * target_centered)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = target_center - R @ source_center
    return R, t


def empirical_logit_evidence(values, larger_is_better=True):
    """
    Convert a score vector into centered, parameter-free evidence values.

    Scores are ranked empirically and mapped to log-odds. This places unrelated
    evidence channels on a common scale without introducing hand-tuned weights.
    Values above the empirical median receive positive evidence, values below it
    receive negative evidence, and tied/isolated cases remain near zero.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)

    working = values if larger_is_better else -values
    ranks = rankdata(working, method="average")
    p = ranks / (values.size + 1.0)
    return np.log(p) - np.log1p(-p)


def solve_frontier_assignment(frontier_A, frontier_B, candidate_pairs, candidate_scores):
    """
    Solve one-to-one matching on the current product-graph frontier.

    Each frontier node in slice A receives a private dummy option with zero
    evidence. A real pair is accepted only when its joint evidence is stronger
    than that neutral alternative, which yields an automatic stopping rule
    without prescribing a target region size.
    """
    if len(frontier_A) == 0 or len(frontier_B) == 0 or len(candidate_pairs) == 0:
        return []

    frontier_A = np.array(sorted(frontier_A), dtype=int)
    frontier_B = np.array(sorted(frontier_B), dtype=int)
    col_map = {v: i for i, v in enumerate(frontier_B)}

    BIG = 1e9
    cost = np.full((len(frontier_A), len(frontier_B) + len(frontier_A)), BIG, dtype=np.float64)
    np.fill_diagonal(cost[:, len(frontier_B):], 0.0)

    row_map = {u: i for i, u in enumerate(frontier_A)}
    for (u, v), score in zip(candidate_pairs, candidate_scores):
        if u not in row_map or v not in col_map:
            continue
        cost[row_map[u], col_map[v]] = -float(score)

    row_ind, col_ind = linear_sum_assignment(cost)

    selected = []
    for r, c in zip(row_ind, col_ind):
        if c < len(frontier_B) and cost[r, c] < 0:
            selected.append((int(frontier_A[r]), int(frontier_B[c])))
    return selected


def select_initial_match_component(match_adj, match_scores):
    """
    Select the strongest initial motif in the match graph.

    Preference order:
    1) triangle (three mutually adjacent matched pairs)
    2) edge (two adjacent matched pairs)
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
                score = match_scores[i] + match_scores[j]
                if score > best_edge_score:
                    best_edge_score = score
                    best_edge = np.array([i, j], dtype=int)

    if best_edge is not None:
        return best_edge

    # Final fall back: single best matched pair.
    return np.array([int(np.argmax(match_scores))], dtype=int)


def extract_continuous_macro_section(
    sliceA,
    sliceB,
    labels_A,
    labels_B,
    Pi_cluster,
    spatial_key='spatial',
    label_key='cell_type_annot'
):
    """
    Identify a compact, biologically consistent overlap region from the coarse alignment.

    The procedure intentionally avoids fixed region-size targets and barycentric
    growth heuristics. It proceeds in two stages:

    1. Seed selection:
       Candidate cluster-pairs are defined by enrichment above a
       size-preserving transport null and organized into a match graph. The
       initial anchor is the strongest triangle, then edge, then singleton.

    2. Coupled frontier expansion:
       Starting from the seed motif, the method grows both slices jointly on the
       product graph of admissible cluster-pairs. Only frontier pairs that are
       simultaneously supported by transport enrichment, local niche similarity,
       geodesic consistency relative to the already selected anchor, and rigid
       consistency under the current seed-derived transform are accepted.

    This design prevents the region from drifting into nearby symmetric
    compartments because expansion is never allowed independently on the two
    slices and every newly added pair must beat the neutral unmatched
    alternative in a one-to-one frontier assignment.

    Returns
    -------
    idx_A, idx_B:
        Cell indices belonging to the final coupled overlap region.
    dist_A, dist_B:
        Distance-to-core fields used to bias the final fine-scale alignment.
    initial_idx_A, initial_idx_B:
        Cell indices belonging to the initial seed motif before coupled growth.
    selected_pairs:
        Ordered list of matched cluster-pairs retained by the coupled expansion.
    """
    N, M = sliceA.shape[0], sliceB.shape[0]
    
    total_mass = np.sum(Pi_cluster)
    if total_mass == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.full(N, np.inf, dtype=np.float64),
            np.full(M, np.inf, dtype=np.float64),
            np.array([], dtype=int),
            np.array([], dtype=int),
            [],
        )
        
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

    # 2. Build cluster contact graphs and intrinsic geodesic coordinates
    adj_A, edge_A = build_cluster_contact_graph(coords_A, labels_A, valid_A)
    adj_B, edge_B = build_cluster_contact_graph(coords_B, labels_B, valid_B)

    edge_A_norm, edge_scale_A = normalize_contact_graph(edge_A)
    edge_B_norm, edge_scale_B = normalize_contact_graph(edge_B)
    geodesic_A = compute_graph_geodesics(edge_A_norm)
    geodesic_B = compute_graph_geodesics(edge_B_norm)

    # 3. Build biologically grounded cluster context descriptors
    all_types = np.array(sorted(
        set(sliceA.obs[label_key].astype(str)) |
        set(sliceB.obs[label_key].astype(str))
    ), dtype=str)
    cluster_hist_A, _ = compute_cluster_cell_type_histograms(
        sliceA, labels_A, num_clusters_A, label_key=label_key, all_types=all_types
    )
    cluster_hist_B, _ = compute_cluster_cell_type_histograms(
        sliceB, labels_B, num_clusters_B, label_key=label_key, all_types=all_types
    )
    context_feat_A = compute_cluster_context_features(cluster_hist_A, adj_A)
    context_feat_B = compute_cluster_context_features(cluster_hist_B, adj_B)

    # 4. Select cluster-pairs enriched above the size-preserving independence null
    log_enrichment = compute_pairwise_log_enrichment(Pi_cluster)
    mi_contrib = compute_pairwise_mutual_information_contribution(Pi_cluster)
    positive_flat_idx = np.flatnonzero(log_enrichment > 0)
    selected_flat_idx = positive_flat_idx[np.argsort(mi_contrib.ravel()[positive_flat_idx])[::-1]]

    matches = []
    match_scores = []
    transport_signal = []
    context_signal = []
    for idx in selected_flat_idx:
        u, v = np.unravel_index(idx, Pi_cluster.shape)
        if valid_A[u] and valid_B[v]:
            matches.append((u, v))
            match_scores.append(mi_contrib[u, v])
            transport_signal.append(log_enrichment[u, v])

            concat_A = context_feat_A[u]
            concat_B = context_feat_B[v]
            context_signal.append(-jensenshannon(concat_A, concat_B))
            
    num_matches = len(matches)
    if num_matches == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.full(N, np.inf, dtype=np.float64),
            np.full(M, np.inf, dtype=np.float64),
            np.array([], dtype=int),
            np.array([], dtype=int),
            [],
        )
        
    # Convert global biological evidence channels into centered log-odds.
    transport_evidence = empirical_logit_evidence(transport_signal, larger_is_better=True)
    context_evidence = empirical_logit_evidence(context_signal, larger_is_better=True)
    global_pair_evidence = {
        pair: float(te + ce)
        for pair, te, ce in zip(matches, transport_evidence, context_evidence)
    }

    # 5. Enforce Structural Similarity & Continuity via Match-Graph
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
    initial_match_indices = select_initial_match_component(match_adj, match_scores)
    
    seed_pairs = [matches[i] for i in initial_match_indices]
    strong_A = sorted({u for u, _ in seed_pairs})
    strong_B = sorted({v for _, v in seed_pairs})
    
    core_cells_A = np.where(np.isin(labels_A, strong_A))[0]
    core_cells_B = np.where(np.isin(labels_B, strong_B))[0]

    if len(core_cells_A) == 0 or len(core_cells_B) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.full(N, np.inf, dtype=np.float64),
            np.full(M, np.inf, dtype=np.float64),
            np.array([], dtype=int),
            np.array([], dtype=int),
            [],
        )

    initial_idx_A = core_cells_A.copy()
    initial_idx_B = core_cells_B.copy()

    # 6. Coupled frontier expansion on the product graph of admissible cluster pairs
    print("[HOT] Coupled frontier expansion: growing only one-to-one frontier matches that beat the unmatched null.")
    selected_pairs = list(seed_pairs)
    selected_A = set(strong_A)
    selected_B = set(strong_B)
    candidate_pair_set = set(matches)
    transform_scale = max(edge_scale_A, edge_scale_B, 1e-12)

    round_idx = 0
    while True:
        round_idx += 1
        frontier_A = {
            node
            for node in np.where(np.any(adj_A[list(selected_A), :], axis=0))[0]
            if valid_A[node] and node not in selected_A
        }
        frontier_B = {
            node
            for node in np.where(np.any(adj_B[list(selected_B), :], axis=0))[0]
            if valid_B[node] and node not in selected_B
        }

        if not frontier_A or not frontier_B:
            break

        seed_weights = np.array(
            [max(mi_contrib[u, v], 1e-12) for u, v in selected_pairs],
            dtype=np.float64
        )
        R_seed, t_seed = fit_weighted_rigid_transform(
            centroids_A[[u for u, _ in selected_pairs]],
            centroids_B[[v for _, v in selected_pairs]],
            weights=seed_weights
        )

        frontier_pairs = []
        support_counts = []
        topology_gaps = []
        attachment_gaps = []
        rigid_residuals = []

        selected_us = np.array([u for u, _ in selected_pairs], dtype=int)
        selected_vs = np.array([v for _, v in selected_pairs], dtype=int)

        for u in sorted(frontier_A):
            for v in sorted(frontier_B):
                if (u, v) not in candidate_pair_set:
                    continue

                support_pairs = [
                    (su, sv)
                    for su, sv in selected_pairs
                    if adj_A[u, su] and adj_B[v, sv]
                ]
                if not support_pairs:
                    continue

                topology_gap = float(np.median(np.abs(
                    geodesic_A[u, selected_us] - geodesic_B[v, selected_vs]
                )))

                support_us = np.array([su for su, _ in support_pairs], dtype=int)
                support_vs = np.array([sv for _, sv in support_pairs], dtype=int)
                attachment_gap = float(np.median(np.abs(
                    edge_A_norm[u, support_us] - edge_B_norm[v, support_vs]
                )))

                rigid_prediction = centroids_A[u] @ R_seed.T + t_seed
                rigid_residual = float(
                    np.linalg.norm(centroids_B[v] - rigid_prediction) / transform_scale
                )

                frontier_pairs.append((u, v))
                support_counts.append(len(support_pairs))
                topology_gaps.append(topology_gap)
                attachment_gaps.append(attachment_gap)
                rigid_residuals.append(rigid_residual)

        if not frontier_pairs:
            break

        support_evidence = empirical_logit_evidence(support_counts, larger_is_better=True)
        topology_evidence = empirical_logit_evidence(topology_gaps, larger_is_better=False)
        attachment_evidence = empirical_logit_evidence(attachment_gaps, larger_is_better=False)
        rigid_evidence = empirical_logit_evidence(rigid_residuals, larger_is_better=False)

        frontier_scores = []
        for pair, se, te, ae, re in zip(
            frontier_pairs,
            support_evidence,
            topology_evidence,
            attachment_evidence,
            rigid_evidence
        ):
            frontier_scores.append(global_pair_evidence[pair] + float(se + te + ae + re))

        accepted_pairs = solve_frontier_assignment(frontier_A, frontier_B, frontier_pairs, frontier_scores)
        if not accepted_pairs:
            break

        for u, v in accepted_pairs:
            selected_pairs.append((u, v))
            selected_A.add(u)
            selected_B.add(v)

        print(f"[HOT] Expansion round {round_idx}: accepted {len(accepted_pairs)} matched frontier pair(s).")

    if round_idx == 1 and len(selected_pairs) == len(seed_pairs):
        print("[HOT] Expansion terminated at the seed: no frontier pair exceeded the unmatched null.")

    strong_A = sorted(selected_A)
    strong_B = sorted(selected_B)

    # Extract final cells based strictly on the coupled expansion result
    idx_A = np.where(np.isin(labels_A, strong_A))[0]
    idx_B = np.where(np.isin(labels_B, strong_B))[0]

    # Compute physical distances from the newly expanded core to all cells
    core_coords_A = coords_A[idx_A]
    core_coords_B = coords_B[idx_B]

    tree_A = cKDTree(core_coords_A)
    tree_B = cKDTree(core_coords_B)
    
    dist_A, _ = tree_A.query(coords_A)
    dist_B, _ = tree_B.query(coords_B)
    
    return idx_A, idx_B, dist_A, dist_B, initial_idx_A, initial_idx_B, selected_pairs

