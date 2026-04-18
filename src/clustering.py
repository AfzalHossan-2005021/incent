import numpy as np
import scipy.sparse as sp

from anndata import AnnData
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.cluster import AgglomerativeClustering


def _dense_matrix(X) -> np.ndarray:
    """
    Materialize an expression matrix as a dense floating-point array.
    """
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _standardize_block(X: np.ndarray) -> np.ndarray:
    """
    Standardize one feature block and equalize its total contribution.

    Each block is z-scored feature-wise and then divided by the square root of
    its dimensionality. This keeps expression, neighborhood, and annotation
    blocks on comparable scales without introducing user-facing weights.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[1] == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float64)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std
    X /= np.sqrt(float(X.shape[1]))
    return X


def _row_normalize(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    """
    Row-normalize a sparse adjacency matrix.
    """
    adjacency = adjacency.tocsr()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    inv = np.zeros_like(degrees)
    inv[degrees > 0] = 1.0 / degrees[degrees > 0]
    return sp.diags(inv) @ adjacency


def _one_hot(labels: np.ndarray) -> np.ndarray:
    """
    Deterministic one-hot encoding with sorted label order.
    """
    labels = np.asarray(labels).astype(str)
    unique = np.array(sorted(np.unique(labels)), dtype=str)
    lookup = {label: idx for idx, label in enumerate(unique)}
    encoded = np.zeros((labels.shape[0], unique.shape[0]), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), [lookup[label] for label in labels]] = 1.0
    return encoded


def build_spatial_graph(coords: np.ndarray) -> sp.csr_matrix:
    """
    Build a deterministic local cell-contact graph using the Gabriel graph.

    We start from the Delaunay triangulation and keep only Gabriel edges, whose
    closed diameter disc is empty. This yields a parameter-free, locality-aware
    graph that is less prone than raw Delaunay edges to creating long shortcuts
    between nearby but biologically distinct structures.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_cells = coords.shape[0]
    if n_cells <= 1:
        return sp.csr_matrix((n_cells, n_cells), dtype=np.float64)

    try:
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        candidate_edges = []
        for i in range(n_cells):
            for j in indices[indptr[i]:indptr[i + 1]]:
                if i < j:
                    candidate_edges.append((int(i), int(j)))
    except QhullError:
        tree = cKDTree(coords)
        dists, nbrs = tree.query(coords, k=min(2, n_cells))
        candidate_edges = []
        for i in range(n_cells):
            if nbrs.ndim == 1:
                continue
            j = int(nbrs[i, 1])
            if i != j:
                candidate_edges.append((min(i, j), max(i, j)))
        candidate_edges = sorted(set(candidate_edges))

    if not candidate_edges:
        return sp.csr_matrix((n_cells, n_cells), dtype=np.float64)

    tree = cKDTree(coords)
    kept_edges = []
    tol = 1e-12
    for i, j in candidate_edges:
        midpoint = 0.5 * (coords[i] + coords[j])
        radius = 0.5 * float(np.linalg.norm(coords[i] - coords[j]))
        nearest_dist = float(tree.query(midpoint, k=1)[0])
        if nearest_dist + tol >= radius:
            kept_edges.append((i, j))

    if not kept_edges:
        kept_edges = candidate_edges

    row = np.array([i for i, j in kept_edges] + [j for i, j in kept_edges], dtype=int)
    col = np.array([j for i, j in kept_edges] + [i for i, j in kept_edges], dtype=int)
    data = np.ones(row.shape[0], dtype=np.float64)
    adjacency = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells), dtype=np.float64)
    adjacency = adjacency.tocsr()
    adjacency.sum_duplicates()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    return adjacency


def build_biology_oriented_features(
    adata: AnnData,
    adjacency: sp.csr_matrix,
    label_key: str = "cell_type_annot",
) -> np.ndarray:
    """
    Build deterministic biology-aware features for mesoregion clustering.

    The feature space follows the paper-level design goal of combining a cell's
    own molecular identity with multiscale spatial context. We therefore use
    deterministic blocks that capture:

    1. own expression / latent profile
    2. first-order neighborhood expression
    3. second-order neighborhood expression
    4. local expression-boundary strength
    5. neighborhood cell-type context, when annotations are available
    6. local annotation-boundary strength and neighborhood diversity

    This is still parameter-free from the user's point of view, but it is much
    closer to the neighborhood-aware ideas that perform well in modern spatial
    domain methods.
    """
    if "X_pca" in adata.obsm:
        expr = _dense_matrix(adata.obsm["X_pca"])
    else:
        expr = _dense_matrix(adata.X)

    transition = _row_normalize(adjacency)
    transition2 = transition @ transition if adjacency.nnz > 0 else transition
    nbr_expr = transition @ expr if adjacency.nnz > 0 else np.zeros_like(expr)
    nbr2_expr = transition2 @ expr if adjacency.nnz > 0 else np.zeros_like(expr)
    expr_boundary_1 = np.linalg.norm(expr - nbr_expr, axis=1, keepdims=True)
    expr_boundary_2 = np.linalg.norm(nbr_expr - nbr2_expr, axis=1, keepdims=True)

    feature_blocks = [
        _standardize_block(expr),
        _standardize_block(nbr_expr),
        _standardize_block(nbr2_expr),
        _standardize_block(expr_boundary_1),
        _standardize_block(expr_boundary_2),
    ]

    if label_key in adata.obs:
        own_type = _one_hot(adata.obs[label_key].to_numpy())
        nbr_type = transition @ own_type if adjacency.nnz > 0 else np.zeros_like(own_type)
        nbr2_type = transition2 @ own_type if adjacency.nnz > 0 else np.zeros_like(own_type)
        type_boundary_1 = np.linalg.norm(own_type - nbr_type, axis=1, keepdims=True)
        type_boundary_2 = np.linalg.norm(nbr_type - nbr2_type, axis=1, keepdims=True)
        type_entropy_1 = -np.sum(nbr_type * np.log(np.clip(nbr_type, 1e-12, 1.0)), axis=1, keepdims=True)
        type_entropy_2 = -np.sum(nbr2_type * np.log(np.clip(nbr2_type, 1e-12, 1.0)), axis=1, keepdims=True)
        feature_blocks.extend([
            _standardize_block(own_type),
            _standardize_block(nbr_type),
            _standardize_block(nbr2_type),
            _standardize_block(type_boundary_1),
            _standardize_block(type_boundary_2),
            _standardize_block(type_entropy_1),
            _standardize_block(type_entropy_2),
        ])

    return np.concatenate(feature_blocks, axis=1)


def _ward_wss_progression(features: np.ndarray, children: np.ndarray):
    """
    Track total within-cluster sum of squares across the Ward dendrogram.

    This lets us evaluate every possible contiguous partition without trying
    many user-tuned resolutions.
    """
    features = np.asarray(features, dtype=np.float64)
    children = np.asarray(children, dtype=int)
    n_samples, n_features = features.shape
    total_nodes = 2 * n_samples - 1

    cluster_size = np.zeros(total_nodes, dtype=np.int64)
    cluster_size[:n_samples] = 1

    cluster_sum = np.zeros((total_nodes, n_features), dtype=np.float64)
    cluster_sum[:n_samples] = features

    cluster_sumsq = np.zeros(total_nodes, dtype=np.float64)
    cluster_sumsq[:n_samples] = np.einsum("ij,ij->i", features, features)

    cluster_sse = np.zeros(total_nodes, dtype=np.float64)
    wss_by_k = np.full(n_samples + 1, np.nan, dtype=np.float64)
    merge_delta_by_k = np.zeros(n_samples + 1, dtype=np.float64)
    entropy_evenness_by_k = np.zeros(n_samples + 1, dtype=np.float64)
    wss_by_k[n_samples] = 0.0
    entropy_current = np.log(float(n_samples))
    entropy_evenness_by_k[n_samples] = 1.0

    def entropy_term(size: int) -> float:
        p = float(size) / float(n_samples)
        return -p * np.log(max(p, 1e-300))

    running_wss = 0.0
    for step, (left, right) in enumerate(children):
        node = n_samples + step
        cluster_size[node] = cluster_size[left] + cluster_size[right]
        cluster_sum[node] = cluster_sum[left] + cluster_sum[right]
        cluster_sumsq[node] = cluster_sumsq[left] + cluster_sumsq[right]

        mean_norm_sq = np.dot(cluster_sum[node], cluster_sum[node]) / float(cluster_size[node])
        cluster_sse[node] = max(cluster_sumsq[node] - mean_norm_sq, 0.0)
        delta = max(cluster_sse[node] - cluster_sse[left] - cluster_sse[right], 0.0)
        running_wss += delta

        k = n_samples - step - 1
        wss_by_k[k] = running_wss
        merge_delta_by_k[k] = delta
        entropy_current += (
            entropy_term(cluster_size[node])
            - entropy_term(cluster_size[left])
            - entropy_term(cluster_size[right])
        )
        if k >= 2:
            entropy_evenness_by_k[k] = float(entropy_current / np.log(float(k)))
        else:
            entropy_evenness_by_k[k] = 0.0

    total_sum = np.sum(features, axis=0)
    total_sumsq = float(np.sum(features * features))
    total_sse = max(total_sumsq - np.dot(total_sum, total_sum) / float(n_samples), 0.0)
    return wss_by_k, merge_delta_by_k, entropy_evenness_by_k, total_sse


def select_cluster_count_from_ward_tree(features: np.ndarray, children: np.ndarray) -> tuple[int, dict[str, float | int | None]]:
    """
    Choose the mesoregion count automatically from the full Ward dendrogram.

    We maximize a balance-aware and persistence-aware Calinski-Harabasz
    criterion over all contiguous partitions represented in the dendrogram.
    Plain Calinski-Harabasz often prefers a pathological split with one tiny
    outlier cluster and one giant remainder cluster. To avoid that failure mode
    deterministically, we:

    1. multiply the CH score by the normalized entropy of the cluster-size
       distribution, discouraging degenerate cuts
    2. multiply again by a normalized merge-persistence term, rewarding cuts
       that sit just before strong Ward merges and are therefore more stable

    This stays parameter-free while better matching the multiscale, stable
    mesoregion story that is easier to defend in a paper.
    """
    n_samples = int(features.shape[0])
    if n_samples <= 3:
        return 1, {
            "cluster_count_mode": "small_component_single_cluster",
            "selected_cluster_count": 1,
            "selected_ch_score": None,
        }

    wss_by_k, merge_delta_by_k, entropy_evenness_by_k, total_sse = _ward_wss_progression(features, children)
    positive_merge_deltas = merge_delta_by_k[merge_delta_by_k > 0]
    if positive_merge_deltas.size > 0:
        max_log_merge = float(np.max(np.log1p(positive_merge_deltas)))
    else:
        max_log_merge = 0.0
    candidate_scores = []

    for k in range(2, n_samples):
        wss = wss_by_k[k]
        if not np.isfinite(wss) or wss <= 1e-12:
            continue

        between = max(total_sse - wss, 0.0)
        within_mean = wss / float(max(n_samples - k, 1))
        if within_mean <= 0:
            continue

        ch_score = (between / float(max(k - 1, 1))) / within_mean
        if not np.isfinite(ch_score):
            continue
        evenness = float(np.clip(entropy_evenness_by_k[k], 0.0, 1.0))
        balanced_ch_score = float(ch_score * evenness)
        if max_log_merge > 0.0:
            persistence = float(np.log1p(merge_delta_by_k[k]) / max_log_merge)
        else:
            persistence = 1.0
        persistence = float(np.clip(persistence, 0.0, 1.0))
        stable_score = float(balanced_ch_score * persistence)

        candidate_scores.append((
            stable_score,
            balanced_ch_score,
            float(ch_score),
            evenness,
            persistence,
            float(merge_delta_by_k[k]),
            -int(k),
            int(k),
        ))

    if not candidate_scores:
        return 1, {
            "cluster_count_mode": "ward_tree_no_valid_cut",
            "selected_cluster_count": 1,
            "selected_ch_score": None,
        }

    candidate_scores.sort(reverse=True)
    best_stable_score, best_balanced_ch, best_ch, best_evenness, best_persistence, best_delta, _, best_k = candidate_scores[0]
    return best_k, {
        "cluster_count_mode": "ward_tree_stable_balanced_calinski_harabasz",
        "selected_cluster_count": int(best_k),
        "selected_ch_score": float(best_ch),
        "selected_balanced_ch_score": float(best_balanced_ch),
        "selected_size_evenness": float(best_evenness),
        "selected_merge_persistence": float(best_persistence),
        "selected_stable_score": float(best_stable_score),
        "selected_merge_delta": float(best_delta),
        "candidate_cut_count": int(len(candidate_scores)),
    }


def labels_from_ward_tree(children: np.ndarray, n_samples: int, n_clusters: int) -> np.ndarray:
    """
    Cut a Ward dendrogram deterministically without rerunning clustering.
    """
    n_samples = int(n_samples)
    n_clusters = int(max(1, min(n_clusters, n_samples)))
    if n_clusters == 1:
        return np.zeros(n_samples, dtype=int)
    if n_clusters == n_samples:
        return np.arange(n_samples, dtype=int)

    parent = np.arange(n_samples, dtype=int)
    rank = np.zeros(n_samples, dtype=np.int8)
    node_rep = np.arange(2 * n_samples - 1, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return ra
        if rank[ra] < rank[rb]:
            parent[ra] = rb
            return rb
        if rank[ra] > rank[rb]:
            parent[rb] = ra
            return ra
        parent[rb] = ra
        rank[ra] += 1
        return ra

    merges_to_apply = n_samples - n_clusters
    for step in range(merges_to_apply):
        left, right = map(int, children[step])
        merged_rep = union(node_rep[left], node_rep[right])
        node_rep[n_samples + step] = merged_rep

    roots = np.array([find(i) for i in range(n_samples)], dtype=int)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(int)


def _cluster_connected_component(features: np.ndarray, connectivity: sp.csr_matrix) -> tuple[np.ndarray, dict[str, float | int | None]]:
    """
    Cluster one connected tissue component.
    """
    n_cells = int(features.shape[0])
    if n_cells <= 3:
        return np.zeros(n_cells, dtype=int), {
            "component_size": n_cells,
            "selected_cluster_count": 1,
            "cluster_count_mode": "small_component_single_cluster",
        }

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.0,
        linkage="ward",
        connectivity=connectivity,
        compute_full_tree=True,
    )
    model.fit(features)

    n_clusters, diagnostics = select_cluster_count_from_ward_tree(features, model.children_)
    labels = labels_from_ward_tree(model.children_, n_cells, n_clusters)
    diagnostics = dict(diagnostics)
    diagnostics["component_size"] = n_cells
    diagnostics["ward_merge_count"] = int(model.children_.shape[0])
    diagnostics["feature_dim"] = int(features.shape[1])
    return labels, diagnostics


def cluster_cells_spatial(adata: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """
    Deterministic, biology-aware mesoregion clustering for spatial alignment.

    The previous implementation clustered cells using Ward linkage on
    coordinates alone with a manually chosen cluster count. That approach was
    deterministic but overly geometric, which is undesirable when nearby
    symmetric structures should remain separated by their molecular and
    microenvironmental context.

    The current method keeps determinism and removes user-facing clustering
    parameters:

    1. build a parameter-free Gabriel cell-contact graph
    2. construct neighborhood-augmented biology-aware features
    3. run spatially constrained Ward agglomeration on each connected component
    4. choose the cut automatically by maximizing a stable, balance-aware
       Calinski-Harabasz score over the full Ward dendrogram
    """

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = coords.shape[0]
    if n_cells == 0:
        return np.zeros(0, dtype=int)
    if n_cells == 1:
        return np.zeros(1, dtype=int)

    adjacency = build_spatial_graph(coords)
    if adjacency.nnz == 0:
        return np.zeros(n_cells, dtype=int)

    features = build_biology_oriented_features(adata, adjacency)

    n_components, component_ids = connected_components(adjacency, directed=False, return_labels=True)
    labels = np.full(n_cells, -1, dtype=int)
    next_label = 0
    component_diagnostics = []

    for component_id in range(int(n_components)):
        idx = np.where(component_ids == component_id)[0]
        if idx.size == 0:
            continue

        component_labels, diagnostics = _cluster_connected_component(
            # Each connected tissue component is clustered independently so
            # disconnected islands cannot be merged into one mesoregion.
            features[idx],
            adjacency[idx][:, idx].tocsr(),
        )
        labels[idx] = component_labels + next_label
        next_label += int(component_labels.max()) + 1
        diagnostics = dict(diagnostics)
        diagnostics["component_id"] = int(component_id)
        component_diagnostics.append(diagnostics)

    adata.uns["incent_clustering"] = {
        "graph_mode": "gabriel",
        "feature_mode": "multiscale_biology_context",
        "n_components": int(n_components),
        "n_clusters": int(len(np.unique(labels))),
        "component_diagnostics": component_diagnostics,
        "cluster_sizes": np.bincount(labels, minlength=int(labels.max()) + 1).astype(int).tolist(),
    }

    return labels
