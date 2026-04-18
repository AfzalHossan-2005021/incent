import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import connected_components
from anndata import AnnData


def build_bio_spatial_graph(
    coords: np.ndarray,
    expr: np.ndarray,
    cell_types: np.ndarray | None = None,
) -> tuple[list, np.ndarray]:
    """
    Build a Delaunay spatial contiguity graph with biologically-weighted edges.

    Edge weight = cosine similarity in expression space, optionally boosted
    by a cell-type agreement term that scales with vocabulary size (no free
    parameter). The weight lives in [0, 1] and represents how "biologically
    compatible" two adjacent cells are.

    Args:
        coords  : (N, 2) spatial coordinates.
        expr    : (N, D) PCA or latent expression features.
        cell_types: (N,) string cell-type labels, or None.

    Returns:
        edges   : list of (i, j) undirected edge pairs.
        weights : (|E|,) float array of biological affinities in [0, 1].
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

    # Expression affinity: 1 - cosine_distance, clipped to [0, 1]
    # Compute row-by-row to avoid O(N²) memory cost
    expr_norm = expr / (np.linalg.norm(expr, axis=1, keepdims=True) + 1e-12)
    dot_products = np.einsum('ij,ij->i', expr_norm[edge_i], expr_norm[edge_j])
    expr_sim = np.clip((dot_products + 1.0) / 2.0, 0.0, 1.0)  # map [-1,1]→[0,1]

    weights = expr_sim.copy()

    # Cell-type agreement boost: parameter-free, scales with vocabulary size
    if cell_types is not None:
        K = max(len(np.unique(cell_types)), 1)
        type_bonus = 1.0 / K          # bonus diminishes with diversity
        same_type = (cell_types[edge_i] == cell_types[edge_j])
        weights[same_type] = np.minimum(1.0, weights[same_type] + type_bonus)

    edges = list(zip(edge_i.tolist(), edge_j.tolist()))
    return edges, weights


def cluster_cells_biological(
    adata: AnnData,
    spatial_key: str = 'spatial',
    feature_key: str = 'X_pca',
    label_key: str = 'cell_type_annot',
) -> np.ndarray:
    """
    Cluster cells into biologically grounded, spatially contiguous mesoregions.

    Uses a Delaunay spatial contiguity graph with expression-weighted edges and
    the Constant Potts Model (CPM) Leiden algorithm. The CPM resolution is
    derived entirely from the data — it equals the mean biological edge weight,
    meaning communities form only when cells are more similar to each other than
    the average pair of spatially adjacent cells in the tissue.

    No magic numbers. No free resolution parameter. The granularity is
    determined by the tissue's own transcriptional structure.

    Args:
        adata       : AnnData object.
        spatial_key : Key in adata.obsm storing (N, 2) coordinates.
        feature_key : Key in adata.obsm storing expression features (e.g. X_pca).
        label_key   : Key in adata.obs storing cell-type annotations (optional).

    Returns:
        labels : (N,) integer array with cluster indices 0..C-1.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError(
            "BioSpatial-CPM clustering requires igraph and leidenalg. "
            "Install via: pip install igraph leidenalg"
        )

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = coords.shape[0]

    if n_cells <= 2:
        return np.zeros(n_cells, dtype=int)

    # ── Expression features ──────────────────────────────────────────────────
    if feature_key == 'X':
        import scipy.sparse as sp
        expr = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    else:
        expr = np.asarray(adata.obsm[feature_key], dtype=np.float64)

    # ── Cell types (optional) ────────────────────────────────────────────────
    cell_types = None
    if label_key is not None and label_key in adata.obs:
        cell_types = adata.obs[label_key].astype(str).to_numpy()

    # ── Build biologically-weighted spatial graph ────────────────────────────
    edges, weights = build_bio_spatial_graph(coords, expr, cell_types)

    if not edges:
        return np.zeros(n_cells, dtype=int)

    # ── Data-adaptive CPM resolution: mean biological edge weight ────────────
    # This is the natural null model: a community is stable only when its
    # average internal cohesion exceeds the tissue-wide average adjacency.
    gamma = float(np.mean(weights))

    # ── Build igraph and run Leiden (CPM) ───────────────────────────────────
    g = ig.Graph(n=n_cells, edges=edges, directed=False)
    g.es['weight'] = weights.tolist()

    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights='weight',
        resolution_parameter=gamma,
        seed=0,          # deterministic
    )
    labels = np.array(partition.membership, dtype=int)

    # ── Post-process 1: enforce strict spatial contiguity ────────────────────
    labels = _enforce_contiguity(labels, edges, n_cells)

    # ── Post-process 2: absorb singletons into best neighbor ────────────────
    labels = _absorb_singletons(labels, edges, weights, n_cells)

    # Reindex to 0..C-1
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(int)


def _enforce_contiguity(
    labels: np.ndarray,
    edges: list,
    n_cells: int,
) -> np.ndarray:
    """
    Split any geometrically disconnected community into separate communities.

    Leiden on a spatial graph can occasionally assign a type-coherent but
    geometrically fragmented set of cells to the same community. This pass
    detects disconnected induced subgraphs and assigns them distinct labels.
    """
    new_labels = labels.copy()
    next_label = int(labels.max()) + 1

    for community_id in np.unique(labels):
        member_mask = labels == community_id
        member_idx = np.where(member_mask)[0]
        if member_idx.size <= 1:
            continue

        # Build induced subgraph adjacency
        idx_map = {v: i for i, v in enumerate(member_idx)}
        sub_edges = []
        for (i, j) in edges:
            if i in idx_map and j in idx_map:
                sub_edges.append((idx_map[i], idx_map[j]))

        if not sub_edges:
            continue

        n_sub = member_idx.size
        rows = [e[0] for e in sub_edges] + [e[1] for e in sub_edges]
        cols = [e[1] for e in sub_edges] + [e[0] for e in sub_edges]
        adj = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_sub, n_sub)
        ).tocsr()
        n_comp, comp_labels = connected_components(adj, directed=False)

        if n_comp > 1:
            # Keep the largest component with original label; split the rest
            comp_sizes = np.bincount(comp_labels)
            largest = int(np.argmax(comp_sizes))
            for comp_id in range(n_comp):
                if comp_id == largest:
                    continue
                orphan_global = member_idx[comp_labels == comp_id]
                new_labels[orphan_global] = next_label
                next_label += 1

    return new_labels


def _absorb_singletons(
    labels: np.ndarray,
    edges: list,
    weights: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """
    Merge size-1 communities into the most transcriptionally similar neighbor.

    Singletons can appear when a cell's expression profile is too distinct to
    exceed the CPM threshold with any neighbor. Rather than reporting them as
    isolated clusters (which would produce trivial uninformative mesoregions),
    they are absorbed into the neighbor community with the highest edge weight.
    """
    new_labels = labels.copy()
    # Build adjacency with weights: {cell_i: [(cell_j, weight), ...]}
    adj = {i: [] for i in range(n_cells)}
    for (i, j), w in zip(edges, weights):
        adj[i].append((j, w))
        adj[j].append((i, w))

    label_sizes = np.bincount(labels)

    changed = True
    while changed:
        changed = False
        for i in range(n_cells):
            if label_sizes[new_labels[i]] == 1:
                # Find best neighbor by edge weight
                best_j, best_w = -1, -1.0
                for j, w in adj[i]:
                    if new_labels[j] != new_labels[i] and w > best_w:
                        best_j, best_w = j, w
                if best_j >= 0:
                    old_lbl = new_labels[i]
                    new_labels[i] = new_labels[best_j]
                    label_sizes[old_lbl] -= 1
                    label_sizes[new_labels[i]] += 1
                    changed = True

    return new_labels