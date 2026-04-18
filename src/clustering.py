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

    The feature space is inspired by the core idea behind neighborhood-aware
    spatial clustering methods: a cell should be represented both by its own
    molecular state and by the local tissue context around it. We therefore use
    four parameter-free blocks whenever annotations are available:

    1. own expression / latent profile
    2. mean neighborhood expression / latent profile
    3. own cell-type identity
    4. neighborhood cell-type composition

    We also append simple boundary-strength summaries to make close but
    biologically distinct structures easier to separate when they touch or lie
    nearby in space.
    """
    if "X_pca" in adata.obsm:
        expr = _dense_matrix(adata.obsm["X_pca"])
    else:
        expr = _dense_matrix(adata.X)

    transition = _row_normalize(adjacency)
    nbr_expr = transition @ expr if adjacency.nnz > 0 else np.zeros_like(expr)
    expr_boundary = np.linalg.norm(expr - nbr_expr, axis=1, keepdims=True)

    feature_blocks = [
        _standardize_block(expr),
        _standardize_block(nbr_expr),
        _standardize_block(expr_boundary),
    ]

    if label_key in adata.obs:
        own_type = _one_hot(adata.obs[label_key].to_numpy())
        nbr_type = transition @ own_type if adjacency.nnz > 0 else np.zeros_like(own_type)
        type_boundary = np.linalg.norm(own_type - nbr_type, axis=1, keepdims=True)
        feature_blocks.extend([
            _standardize_block(own_type),
            _standardize_block(nbr_type),
            _standardize_block(type_boundary),
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
    wss_by_k[n_samples] = 0.0

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

    total_sum = np.sum(features, axis=0)
    total_sumsq = float(np.sum(features * features))
    total_sse = max(total_sumsq - np.dot(total_sum, total_sum) / float(n_samples), 0.0)
    return wss_by_k, merge_delta_by_k, total_sse


def select_cluster_count_from_ward_tree(features: np.ndarray, children: np.ndarray) -> tuple[int, dict[str, float | int | None]]:
    """
    Choose the mesoregion count automatically from the full Ward dendrogram.

    We maximize the Calinski-Harabasz criterion over all contiguous partitions
    represented in the dendrogram. This is deterministic and parameter-free
    from the user's point of view, while still favoring partitions that are
    both compact and well separated in the biology-aware feature space.
    """
    n_samples = int(features.shape[0])
    if n_samples <= 3:
        return 1, {
            "cluster_count_mode": "small_component_single_cluster",
            "selected_cluster_count": 1,
            "selected_ch_score": None,
        }

    wss_by_k, merge_delta_by_k, total_sse = _ward_wss_progression(features, children)
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

        candidate_scores.append((
            float(ch_score),
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
    _, best_delta, _, best_k = candidate_scores[0]
    best_ch = candidate_scores[0][0]
    return best_k, {
        "cluster_count_mode": "ward_tree_calinski_harabasz",
        "selected_cluster_count": int(best_k),
        "selected_ch_score": float(best_ch),
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
    4. choose the cut automatically by maximizing the Calinski-Harabasz score
       over the full Ward dendrogram
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

    for component_id in range(int(n_components)):
        idx = np.where(component_ids == component_id)[0]
        if idx.size == 0:
            continue

        component_labels, _ = _cluster_connected_component(
            features[idx],
            adjacency[idx][:, idx].tocsr(),
        )
        labels[idx] = component_labels + next_label
        next_label += int(component_labels.max()) + 1

    return labels
