import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import fcluster
from anndata import AnnData


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_pruned_delaunay(coords: np.ndarray):
    """
    Build a Delaunay spatial graph and remove pathologically long edges.

    Long edges arise wherever tissue is sparse or where the convex hull of the
    point cloud forces the triangulation to bridge large gaps. They cause Ward
    to treat distant cells as adjacent, producing non-contiguous or misshapen
    clusters. The Hampel identifier (median ± 3·MAD) is a robust outlier filter
    that does not require a free scale parameter: it uses only the distribution
    of the graph's own edge lengths.
    """
    n_cells = coords.shape[0]
    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices

    edge_i, edge_j = [], []
    for i in range(n_cells):
        for j in indices[indptr[i]:indptr[i + 1]]:
            if i < j:
                edge_i.append(i)
                edge_j.append(j)

    if not edge_i:
        return [], np.array([], dtype=np.float64)

    edge_i = np.array(edge_i, dtype=np.int32)
    edge_j = np.array(edge_j, dtype=np.int32)

    lengths = np.linalg.norm(coords[edge_i] - coords[edge_j], axis=1)
    median_l = np.median(lengths)
    mad_l = np.median(np.abs(lengths - median_l))
    # Hampel identifier: keep edges within median + 3·MAD
    # Equivalent to ~3σ for Gaussian but robust to heavy-tailed distributions
    keep = lengths <= (median_l + 3.0 * mad_l)

    edge_i = edge_i[keep]
    edge_j = edge_j[keep]
    edges = list(zip(edge_i.tolist(), edge_j.tolist()))
    return edges, lengths[keep]


def _build_banksy_features(
    expr: np.ndarray,
    edges: list,
    n_cells: int,
    lambda_nbr: float = 0.2,
) -> np.ndarray:
    """
    Compute BANKSY-augmented feature vectors for each cell.

    Each cell's feature vector concatenates its own (L2-normalized) PCA
    embedding with the L2-normalized mean embedding of its spatial neighbors.
    The weight λ = 0.2 is fixed following Singhal et al. (Nature Genetics 2024)
    and encodes the principle that cell identity (80%) should dominate over
    spatial context (20%), while still allowing local microenvironment to
    influence cluster assignment.

    No free parameters are introduced: λ is a fixed algorithmic constant, not
    a tuneable hyperparameter.
    """
    if not edges:
        return normalize(expr, norm='l2')

    edge_i = np.array([e[0] for e in edges], dtype=np.int32)
    edge_j = np.array([e[1] for e in edges], dtype=np.int32)

    # Build symmetric sparse adjacency
    rows = np.concatenate([edge_i, edge_j])
    cols = np.concatenate([edge_j, edge_i])
    adj = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(n_cells, n_cells)
    ).tocsr()

    # Row-normalized neighbor mean (each cell gets the average of its neighbors)
    row_sums = np.asarray(adj.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    nbr_expr = D_inv @ adj @ expr           # (N, D)

    # L2-normalize each block independently before concatenation
    expr_norm = normalize(expr, norm='l2')
    nbr_norm  = normalize(nbr_expr, norm='l2')

    F = np.concatenate(
        [(1.0 - lambda_nbr) * expr_norm,
          lambda_nbr        * nbr_norm],
        axis=1
    )
    return F


def _build_connectivity_matrix(edges: list, n_cells: int) -> sp.csr_matrix:
    """
    Build a symmetric binary sparse adjacency matrix from an edge list.
    """
    if not edges:
        return sp.eye(n_cells, format='csr', dtype=np.float64)

    edge_i = np.array([e[0] for e in edges], dtype=np.int32)
    edge_j = np.array([e[1] for e in edges], dtype=np.int32)
    rows = np.concatenate([edge_i, edge_j])
    cols = np.concatenate([edge_j, edge_i])
    data = np.ones(len(rows), dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()


def _sklearn_tree_to_scipy_linkage(agg: AgglomerativeClustering, n_cells: int) -> np.ndarray:
    """
    Convert sklearn AgglomerativeClustering output to a scipy-compatible linkage matrix.

    scipy's fcluster requires a (n-1, 4) linkage matrix: [left, right, distance, count].
    sklearn stores merges in children_ and distances_ but not counts; we recompute
    counts by traversing the merge tree in order.
    """
    children = agg.children_       # (n-1, 2): merge tree
    distances = agg.distances_     # (n-1,  ): merge distances in merge order

    # Recompute subtree sizes
    n_leaves = n_cells
    counts = np.zeros(len(children), dtype=int)
    subtree_size = np.ones(n_leaves + len(children), dtype=int)
    for i, (left, right) in enumerate(children):
        counts[i] = subtree_size[left] + subtree_size[right]
        subtree_size[n_leaves + i] = counts[i]

    return np.column_stack([children, distances, counts]).astype(float)


def _select_k_calinski_harabasz(
    linkage_matrix: np.ndarray,
    F: np.ndarray,
    n_cells: int,
) -> int:
    """
    Choose the number of clusters that maximizes the Calinski-Harabász index.

    CH(k) = [B(k)/(k-1)] / [W(k)/(n-k)], where B and W are between- and
    within-cluster variance in biological feature space F. The criterion peaks
    at the partition where biological coherence within clusters is highest
    relative to the total variance — i.e. the natural biological scale.

    The search range [k_min, k_max] is derived from n_cells:
      k_min = max(3, n_cells // 500)  → at least 3 clusters, ≥30 cells each
      k_max = max(k_min+2, n_cells // 30) → at most n_cells/30 clusters
    This range is a biologically motivated bracket, not a magic number:
    mesoregions smaller than ~30 cells are too granular to be meaningful
    for the coarse FGW; mesoregions larger than ~500 cells lose local specificity.
    """
    k_min = max(3, n_cells // 200)
    k_max = max(k_min + 2, n_cells // 50)
    k_max = min(k_max, n_cells - 1)

    best_k, best_ch = k_min, -1.0
    for k in range(k_min, k_max + 1):
        labels_k = fcluster(linkage_matrix, k, criterion='maxclust') - 1
        n_unique = len(np.unique(labels_k))
        if n_unique < 2:
            continue
        ch = calinski_harabasz_score(F, labels_k)
        if ch > best_ch:
            best_ch = ch
            best_k = k

    return best_k


def _enforce_contiguity(labels: np.ndarray, edges: list, n_cells: int) -> np.ndarray:
    """
    Split geometrically disconnected communities into separate clusters.

    Ward with a connected Delaunay graph should never produce disconnected
    communities, but edge pruning can fragment the graph. This pass detects any
    community whose induced subgraph has more than one connected component and
    assigns each fragment a distinct label. It is a safety net, not a crutch.
    """
    new_labels = labels.copy()
    next_label = int(labels.max()) + 1

    # Build a cell-level adjacency lookup
    adj = {i: [] for i in range(n_cells)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    for community_id in np.unique(labels):
        member_mask = labels == community_id
        member_idx  = np.where(member_mask)[0]
        if member_idx.size <= 1:
            continue

        idx_map = {v: pos for pos, v in enumerate(member_idx)}
        n_sub = member_idx.size

        sub_rows, sub_cols = [], []
        for glob_i in member_idx:
            for glob_j in adj.get(glob_i, []):
                if glob_j in idx_map:
                    sub_rows.append(idx_map[glob_i])
                    sub_cols.append(idx_map[glob_j])

        if not sub_rows:
            continue

        sub_adj = sp.coo_matrix(
            (np.ones(len(sub_rows)), (sub_rows, sub_cols)),
            shape=(n_sub, n_sub)
        ).tocsr()
        n_comp, comp_labels = connected_components(sub_adj, directed=False)

        if n_comp > 1:
            comp_sizes  = np.bincount(comp_labels)
            largest_comp = int(np.argmax(comp_sizes))
            for comp_id in range(n_comp):
                if comp_id == largest_comp:
                    continue
                orphan_global = member_idx[comp_labels == comp_id]
                new_labels[orphan_global] = next_label
                next_label += 1

    return new_labels


def _absorb_singletons(labels: np.ndarray, edges: list, n_cells: int) -> np.ndarray:
    """
    Merge size-1 clusters into the cluster of their most-connected neighbor.

    Singletons occur when edge pruning isolates a cell or when CH selects a k
    that leaves one-cell partitions. Each singleton is absorbed into the
    neighbor cluster that shares the most edges with it (ties broken by first
    neighbor encountered). Repeated until no singletons remain.
    """
    new_labels = labels.copy()
    adj = {i: [] for i in range(n_cells)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    changed = True
    while changed:
        changed = False
        label_sizes = np.bincount(new_labels, minlength=int(new_labels.max()) + 1)
        for i in range(n_cells):
            if label_sizes[new_labels[i]] != 1:
                continue
            # Count edges to each neighbor cluster
            neighbor_counts: dict[int, int] = {}
            for j in adj.get(i, []):
                lbl_j = new_labels[j]
                if lbl_j != new_labels[i]:
                    neighbor_counts[lbl_j] = neighbor_counts.get(lbl_j, 0) + 1
            if not neighbor_counts:
                continue
            best_lbl = max(neighbor_counts, key=neighbor_counts.__getitem__)
            old_lbl   = new_labels[i]
            new_labels[i] = best_lbl
            label_sizes[old_lbl]  -= 1
            label_sizes[best_lbl] += 1
            changed = True

    return new_labels


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def cluster_cells(
    adata: AnnData,
    spatial_key: str = 'spatial',
    feature_key: str = 'X_pca'
) -> np.ndarray:
    """
    Cluster cells into biologically grounded, spatially contiguous mesoregions.

    This is a drop-in replacement for the original geometry-only Ward clustering.
    It differs in three fundamental ways:

    1. Feature space: BANKSY-augmented PCA (intrinsic identity + neighborhood
       context) instead of raw spatial coordinates.

    2. Graph quality: Delaunay edges longer than median + 3·MAD are pruned
       before clustering, eliminating false adjacencies across tissue gaps.

    3. Cluster count: automatically determined by the Calinski-Harabász
       criterion evaluated in biological feature space over a data-adaptive
       range. No magic number.

    The result is a partition where:
    - Every cluster is spatially contiguous (guaranteed by Ward + Delaunay)
    - Every cluster has a coherent transcriptional profile
    - The number of clusters reflects the tissue's own biological structure

    Args:
        adata       : AnnData object with spatial coordinates and expression.
        spatial_key : Key in adata.obsm for (N, 2) coordinates.
        feature_key : Key in adata.obsm for PCA or latent features.
                      Use 'X' to fall back to the raw count matrix.
        label_key   : Key in adata.obs for cell-type annotations (optional).
        resolution  : Ignored. Kept for call-site compatibility only.

    Returns:
        labels : (N,) integer array with cluster indices 0 … C-1.
    """
    coords  = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = coords.shape[0]

    if n_cells <= 3:
        return np.zeros(n_cells, dtype=int)

    # ── Expression features ──────────────────────────────────────────────────
    if feature_key == 'X':
        expr = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    else:
        expr = np.asarray(adata.obsm[feature_key], dtype=np.float64)
    expr = expr.astype(np.float64)

    # ── Stage 1: Pruned Delaunay graph ───────────────────────────────────────
    edges, _ = _build_pruned_delaunay(coords)
    print(f"  [BioSpatial-Ward] Pruned Delaunay: {len(edges)} edges retained")

    if not edges:
        return np.zeros(n_cells, dtype=int)

    # ── Stage 2: BANKSY-augmented biological features ────────────────────────
    F = _build_banksy_features(expr, edges, n_cells, lambda_nbr=0.2)

    # ── Stage 3: Full Ward dendrogram (spatially constrained) ────────────────
    connectivity = _build_connectivity_matrix(edges, n_cells)
    agg = AgglomerativeClustering(
        n_clusters=None,
        linkage='ward',
        connectivity=connectivity,
        distance_threshold=0,       # forces full dendrogram computation
        compute_distances=True,
    )
    agg.fit(F)

    # ── Stage 4: Calinski-Harabász automatic k selection ────────────────────
    linkage_matrix = _sklearn_tree_to_scipy_linkage(agg, n_cells)
    k_star = _select_k_calinski_harabasz(linkage_matrix, F, n_cells)
    print(f"  [BioSpatial-Ward] Automatic k* = {k_star} "
          f"(search range [{max(3, n_cells//200)}, {max(5, n_cells//50)}])")

    labels = fcluster(linkage_matrix, k_star, criterion='maxclust') - 1

    # ── Stage 5: Contiguity enforcement + singleton cleanup ──────────────────
    labels = _enforce_contiguity(labels, edges, n_cells)
    labels = _absorb_singletons(labels, edges, n_cells)

    # Reindex to 0 … C-1
    _, labels = np.unique(labels, return_inverse=True)
    print(f"  [BioSpatial-Ward] Final cluster count: {len(np.unique(labels))}")
    return labels.astype(int)