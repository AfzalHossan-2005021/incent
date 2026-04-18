"""
clustering.py — Deterministic, parameter-free mesoregion clustering for spatial alignment.

Every algorithmic choice in this module satisfies three design invariants:

  (I) DETERMINISM  — identical floating-point results for identical inputs,
                     regardless of OS scheduling or thread count.
  (II) ZERO FREE PARAMETERS — the user exposes no tunable knobs. All internal
                     quantities (graph topology, diffusion hops, AGF orders,
                     cluster count) are uniquely determined by the data.
  (III) PRINCIPLED DEFAULTS — every structural choice (Gabriel graph, dyadic
                     diffusion, Fourier harmonic orders, Hartigan criterion) is
                     grounded in an explicit theoretical argument and published
                     reference, not empirical convenience.

Design overview
---------------
1. Gabriel graph    — parameter-free spatial contact graph; uniquely determined
                      by cell positions (Gabriel 1969; Jaromczyk & Toussaint 1992).
2. Multi-hop diffusion features at dyadic hops {1, 2, 4, 8} — covers local to
                      meso-scale context without a radius or bandwidth parameter;
                      dyadic spacing is the standard spectral diffusion basis
                      (Coifman & Lafon 2006, Applied & Computational Harmonic Analysis).
3. Rotation-invariant Angular Gradient Features (AGF) at harmonic orders {1, 2}
                      — computed from Gabriel graph edges; invariant to rigid
                      rotation and reflection by design; follows the BANKSY AGF
                      kernel (Singhal et al. 2024, Nat Genet 56:431) but is made
                      parameter-free by using the Gabriel graph instead of k-NN.
4. Spatially constrained Ward agglomeration — deterministic, produces compact
                      contiguous regions, no initialization sensitivity.
5. Hartigan (1975) criterion as primary admissibility filter — identifies the
                      statistically supported cluster count range directly from
                      the Ward dendrogram without any user-supplied threshold;
                      the threshold 10 follows from the asymptotic F-statistic
                      derivation in Hartigan & Wong (1979, Applied Statistics).
6. Balance-aware Calinski–Harabasz score with merge-persistence weighting —
                      selects the most stable and well-balanced cut within the
                      Hartigan-admissible range.
"""

import numpy as np
import scipy.sparse as sp

from anndata import AnnData
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def _dense_matrix(X) -> np.ndarray:
    """Materialize an expression matrix as a dense floating-point array."""
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _standardize_block(X: np.ndarray) -> np.ndarray:
    """
    Z-score each feature column, then divide by sqrt(d) to equalize the total
    L2-norm contribution of every block regardless of its dimensionality.

    This keeps expression (high-d), AGF (high-d), boundary (1-d), and density
    (1-d) blocks on comparable scales without introducing user-facing weights.
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
    """Row-normalize a sparse adjacency matrix to form the random-walk transition matrix."""
    adjacency = adjacency.tocsr()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    inv = np.zeros_like(degrees)
    inv[degrees > 0] = 1.0 / degrees[degrees > 0]
    return sp.diags(inv) @ adjacency


def _one_hot(labels: np.ndarray) -> np.ndarray:
    """
    Deterministic one-hot encoding with lexicographically sorted label order.

    The sorted order ensures the encoding is identical across runs and
    platforms, which is necessary for determinism of the feature matrix.
    """
    labels = np.asarray(labels).astype(str)
    unique = np.array(sorted(np.unique(labels)), dtype=str)
    lookup = {label: idx for idx, label in enumerate(unique)}
    encoded = np.zeros((labels.shape[0], unique.shape[0]), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), [lookup[label] for label in labels]] = 1.0
    return encoded


# ---------------------------------------------------------------------------
# Gabriel graph
# ---------------------------------------------------------------------------

def build_spatial_graph(coords: np.ndarray) -> sp.csr_matrix:
    """
    Build a deterministic, parameter-free spatial contact graph using the Gabriel graph.

    Construction proceeds in two steps:
      1. Compute the Delaunay triangulation to enumerate candidate edges.
      2. Retain only Gabriel edges: edge (i, j) is kept iff no other cell
         lies strictly inside the closed disc whose diameter is the segment ij
         (Gabriel 1969). The Gabriel graph is a parameter-free subgraph of the
         Delaunay triangulation and a supergraph of the Euclidean MST.

    Rationale for the Gabriel graph over k-NN:
      - k-NN introduces a free parameter k whose optimal value is tissue-density-
        dependent and cannot be set without data-adaptive logic.
      - The Gabriel graph is uniquely determined by cell positions: no k, no
        radius, no bandwidth. Jaromczyk & Toussaint (1992) establish it as the
        canonical parameter-free proximity graph for planar point sets.
      - Gabriel edges are shorter and more local than raw Delaunay edges,
        reducing the risk of spurious long-range connections between distinct
        anatomical structures.
      - Edge weights are set to 1 / distance, giving closer cells stronger
        influence in the subsequent diffusion operations.

    Fallback: for degenerate configurations (co-linear points) where Delaunay
    fails, a nearest-neighbor graph is used with the same Gabriel filter.
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

    edge_lengths = np.array(
        [np.linalg.norm(coords[i] - coords[j]) for i, j in kept_edges],
        dtype=np.float64,
    )
    edge_weights = 1.0 / np.maximum(edge_lengths, 1e-12)
    row = np.array([i for i, j in kept_edges] + [j for i, j in kept_edges], dtype=int)
    col = np.array([j for i, j in kept_edges] + [i for i, j in kept_edges], dtype=int)
    data = np.concatenate([edge_weights, edge_weights]).astype(np.float64)
    adjacency = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells), dtype=np.float64)
    adjacency = adjacency.tocsr()
    adjacency.sum_duplicates()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    return adjacency


# ---------------------------------------------------------------------------
# Angular Gradient Features
# ---------------------------------------------------------------------------

def compute_gabriel_agf(
    coords: np.ndarray,
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    orders: tuple = (1, 2),
) -> np.ndarray:
    """
    Rotation- and reflection-invariant Angular Gradient Features (AGF) computed
    from Gabriel graph edges.

    For cell i and harmonic order m, the AGF is:

        AGF_m[i, d] = |Σ_{j ∈ N_G(i)}  w_ij · exp(i·m·θ_ij) · f_j[d]|
                      ────────────────────────────────────────────────────
                                    Σ_{j ∈ N_G(i)}  w_ij

    where N_G(i) are the Gabriel neighbors, w_ij = 1/dist(i,j) are the
    Gabriel edge weights, θ_ij is the angle from i to j, f_j[d] is the d-th
    feature of cell j, and |·| is the element-wise complex modulus.

    Taking the modulus discards the phase, making AGF_m invariant to arbitrary
    rigid rotation and reflection of the tissue section (Singhal et al. 2024).
    This invariance is necessary for cross-slice alignment where the relative
    orientation of source and target sections is unknown.

    Orders {1, 2} are chosen because:
      - m=0 is already captured by the standard neighborhood mean (nbr_expr).
      - m=1 encodes local expression polarity (directional gradient magnitude).
      - m=2 encodes bilateral / opposite-half anisotropy.
      - m ≥ 3 contributions average to near zero over isotropic tissue density
        and add no discriminative signal in practice (Singhal et al. 2024).
    These are structural choices from the standard Fourier basis, not tuned
    hyperparameters.

    The Gabriel graph (rather than k-NN) is used as the neighborhood definition
    so that no k parameter is required. The neighborhood is completely determined
    by cell positions.

    Implementation uses sparse matrix–dense matrix products (M_cos @ F and
    M_sin @ F) for efficiency and strict IEEE 754 determinism. Edges are sorted
    lexicographically before matrix construction to guarantee bit-identical
    results regardless of upstream edge ordering.

    Parameters
    ----------
    coords:    (n_cells, 2) array of spatial coordinates.
    adjacency: (n_cells, n_cells) symmetric sparse Gabriel adjacency matrix.
    features:  (n_cells, d) feature matrix (e.g. PCA or expression).
    orders:    Fourier harmonic orders to compute. Default: (1, 2).

    Returns
    -------
    agf: (n_cells, d * len(orders)) array of rotation-invariant AGF features.
    """
    coords = np.asarray(coords, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    n_cells, n_features = features.shape

    if adjacency.nnz == 0 or n_cells <= 1:
        return np.zeros((n_cells, n_features * len(orders)), dtype=np.float64)

    # Convert to COO and sort edges (row-major) for strict determinism.
    coo = adjacency.tocoo()
    sort_idx = np.lexsort((coo.col, coo.row))
    edge_i = coo.row[sort_idx].astype(np.int64)
    edge_j = coo.col[sort_idx].astype(np.int64)
    edge_w = coo.data[sort_idx].astype(np.float64)

    # Edge direction angles. np.arctan2 is IEEE 754 compliant and deterministic.
    dx = coords[edge_j, 0] - coords[edge_i, 0]
    dy = coords[edge_j, 1] - coords[edge_i, 1]
    theta = np.arctan2(dy, dx)

    # Per-node weight sums for normalization.
    weight_sum = np.zeros(n_cells, dtype=np.float64)
    np.add.at(weight_sum, edge_i, edge_w)
    weight_sum_safe = np.maximum(weight_sum, 1e-12)

    blocks = []
    for m in orders:
        # Build weighted cosine and sine sparse matrices in a single pass.
        # M_cos[i, j] = w_ij * cos(m * theta_ij)
        # M_sin[i, j] = w_ij * sin(m * theta_ij)
        cos_w = edge_w * np.cos(m * theta)
        sin_w = edge_w * np.sin(m * theta)

        M_cos = sp.csr_matrix(
            (cos_w, (edge_i, edge_j)),
            shape=(n_cells, n_cells),
            dtype=np.float64,
        )
        M_sin = sp.csr_matrix(
            (sin_w, (edge_i, edge_j)),
            shape=(n_cells, n_cells),
            dtype=np.float64,
        )

        # (n_cells, d) real and imaginary parts.
        real_part = (M_cos @ features) / weight_sum_safe[:, None]
        imag_part = (M_sin @ features) / weight_sum_safe[:, None]

        # Element-wise complex modulus — rotation invariant.
        magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
        blocks.append(magnitude)

    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# Multi-scale diffusion
# ---------------------------------------------------------------------------

def _compute_multihop_diffusion(
    transition: sp.csr_matrix,
    features: np.ndarray,
) -> dict:
    """
    Compute diffused feature vectors at dyadic hop counts {1, 2, 4, 8}.

    At hop h, cell i receives:
        f^(h)[i] = Σ_j  T^h[i, j] · f[j]

    i.e. the h-step random-walk weighted average of all cell features. This
    captures an increasingly global view of molecular context as h grows.

    Rationale for the dyadic sequence {1, 2, 4, 8}:
      - The dyadic (geometric doubling) progression is the minimal set that
        covers the frequency spectrum of signals on graphs with O(log h_max)
        steps. It is the graph analogue of the octave spacing in the continuous
        wavelet transform (Coifman & Lafon 2006).
      - Bilateral symmetric structures (e.g. left/right hippocampus) that are
        locally indistinguishable at h=1,2 can be separated at h=4,8 because
        the diffusion front has propagated far enough to reach tissue boundaries
        or morphologically asymmetric compartments that break the symmetry.
      - No bandwidth, radius, or number-of-hops parameter is exposed to the
        user: the sequence is fixed and principled.

    Implementation applies the transition matrix T iteratively as sparse @
    dense products. This avoids materialising dense T^h matrices and keeps
    memory complexity O(nnz + n*d). The total computational cost is 8 sparse
    matrix–vector products (1+1+2+4 for hops 1,2,4,8 respectively).

    Parameters
    ----------
    transition: (n, n) row-normalized Gabriel adjacency (random-walk matrix T).
    features:   (n, d) dense feature matrix.

    Returns
    -------
    dict mapping hop count -> (n, d) diffused feature array.
    """
    if transition.nnz == 0:
        z = np.zeros_like(features, dtype=np.float64)
        return {1: z, 2: z, 4: z, 8: z}

    f1 = transition @ features                              # T^1
    f2 = transition @ f1                                    # T^2
    f4 = transition @ (transition @ f2)                     # T^4
    f8 = transition @ (transition @ (transition @ (transition @ f4)))  # T^8

    return {1: f1, 2: f2, 4: f4, 8: f8}


# ---------------------------------------------------------------------------
# Biology-oriented feature matrix
# ---------------------------------------------------------------------------

def build_biology_oriented_features(
    adata: AnnData,
    adjacency: sp.csr_matrix,
    label_key: str = "cell_type_annot",
) -> np.ndarray:
    """
    Build a deterministic, parameter-free, rotation-invariant feature matrix
    for mesoregion clustering.

    Feature blocks (all standardized independently via _standardize_block):

    Expression blocks (d = PCA or raw expression dimension):
      [A] own expression / latent profile
      [B] T^1-diffused expression  (1-hop neighborhood mean)
      [C] T^2-diffused expression  (2-hop neighborhood mean)
      [D] T^4-diffused expression  (4-hop; new)
      [E] T^8-diffused expression  (8-hop; new — key for symmetric separation)
      [F] AGF order 1 of expression (new; rotation-invariant polarity)
      [G] AGF order 2 of expression (new; rotation-invariant anisotropy)

    Boundary / discontinuity signals (1-d each):
      [H] ||expr - T^1_expr||   (expression boundary at 1-hop scale)
      [I] ||T^1 - T^2_expr||   (expression boundary at 2-hop scale)
      [J] ||T^2 - T^4_expr||   (expression boundary at 4-hop scale; new)
      [K] ||T^4 - T^8_expr||   (expression boundary at 8-hop scale; new)

    Graph density context:
      [L] local degree
      [M] T^1-diffused degree
      [N] T^2-diffused degree
      [O] |degree - T^1_degree| (density boundary)
      [P] |T^1_deg - T^2_deg|   (density boundary at 2-hop)

    Cell-type blocks (C = number of cell types; only if label_key present):
      [Q] T^1 cell-type histogram (local composition)
      [R] T^2 cell-type histogram (2-hop composition)
      [S] T^4 cell-type histogram (4-hop composition; new)
      [T] AGF order 1 of cell type (new; rotation-invariant type gradient)
      [U] AGF order 2 of cell type (new)
      [V] ||own_type - T^1_type||   (type boundary 1-hop)
      [W] ||T^1 - T^2_type||        (type boundary 2-hop)
      [X] ||T^2 - T^4_type||        (type boundary 4-hop; new)
      [Y] -Σ T^1_type · log(T^1_type)  (Shannon entropy of local composition)
      [Z] -Σ T^2_type · log(T^2_type)  (entropy of 2-hop composition)

    Design rationale:
      - The Gabriel graph defines all neighborhoods without any k or radius.
      - Dyadic diffusion hops {1,2,4,8} cover local to meso-scale context
        without a scale parameter (Coifman & Lafon 2006).
      - AGF orders {1,2} are the standard angular Fourier basis for 2D spatial
        signals; rotation invariance comes from discarding the complex phase
        (Singhal et al. 2024, Nat Genet).
      - All block weights are equalised by dividing each standardised block by
        sqrt(d), so no block dominates due to its dimension alone.
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    if "X_pca" in adata.obsm:
        expr = _dense_matrix(adata.obsm["X_pca"])
    else:
        expr = _dense_matrix(adata.X)

    transition = _row_normalize(adjacency)

    # --- Expression: multi-hop diffusion ---
    expr_hops = _compute_multihop_diffusion(transition, expr)
    nbr1_expr = expr_hops[1]
    nbr2_expr = expr_hops[2]
    nbr4_expr = expr_hops[4]
    nbr8_expr = expr_hops[8]

    # --- Expression: rotation-invariant AGF ---
    if adjacency.nnz > 0:
        agf_expr = compute_gabriel_agf(coords, adjacency, expr, orders=(1, 2))
    else:
        agf_expr = np.zeros((expr.shape[0], expr.shape[1] * 2), dtype=np.float64)

    # --- Expression boundary signals ---
    expr_boundary_1 = np.linalg.norm(expr       - nbr1_expr, axis=1, keepdims=True)
    expr_boundary_2 = np.linalg.norm(nbr1_expr  - nbr2_expr, axis=1, keepdims=True)
    expr_boundary_4 = np.linalg.norm(nbr2_expr  - nbr4_expr, axis=1, keepdims=True)
    expr_boundary_8 = np.linalg.norm(nbr4_expr  - nbr8_expr, axis=1, keepdims=True)

    # --- Graph density context ---
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)[:, None]
    if adjacency.nnz > 0:
        nbr_degree  = transition @ degree
        nbr2_degree = transition @ nbr_degree
    else:
        nbr_degree  = np.zeros_like(degree)
        nbr2_degree = np.zeros_like(degree)
    density_boundary_1 = np.abs(degree     - nbr_degree)
    density_boundary_2 = np.abs(nbr_degree - nbr2_degree)

    feature_blocks = [
        _standardize_block(expr),         # [A]
        _standardize_block(nbr1_expr),    # [B]
        _standardize_block(nbr2_expr),    # [C]
        _standardize_block(nbr4_expr),    # [D]
        _standardize_block(nbr8_expr),    # [E]
        _standardize_block(agf_expr),     # [F,G]
        _standardize_block(expr_boundary_1),   # [H]
        _standardize_block(expr_boundary_2),   # [I]
        _standardize_block(expr_boundary_4),   # [J]
        _standardize_block(expr_boundary_8),   # [K]
        _standardize_block(degree),            # [L]
        _standardize_block(nbr_degree),        # [M]
        _standardize_block(nbr2_degree),       # [N]
        _standardize_block(density_boundary_1),# [O]
        _standardize_block(density_boundary_2),# [P]
    ]

    if label_key in adata.obs:
        own_type = _one_hot(adata.obs[label_key].to_numpy())
        type_hops = _compute_multihop_diffusion(transition, own_type)
        nbr1_type = type_hops[1]
        nbr2_type = type_hops[2]
        nbr4_type = type_hops[4]

        # AGF on cell-type one-hot: captures direction of local type-composition
        # gradient in a rotation-invariant way.
        if adjacency.nnz > 0:
            agf_type = compute_gabriel_agf(coords, adjacency, own_type, orders=(1, 2))
        else:
            agf_type = np.zeros(
                (own_type.shape[0], own_type.shape[1] * 2), dtype=np.float64
            )

        type_boundary_1 = np.linalg.norm(own_type  - nbr1_type, axis=1, keepdims=True)
        type_boundary_2 = np.linalg.norm(nbr1_type - nbr2_type, axis=1, keepdims=True)
        type_boundary_4 = np.linalg.norm(nbr2_type - nbr4_type, axis=1, keepdims=True)

        # Shannon entropy of local and 2-hop cell-type distributions.
        # High entropy = diverse niche; low entropy = homogeneous niche.
        type_entropy_1 = -np.sum(
            nbr1_type * np.log(np.clip(nbr1_type, 1e-12, 1.0)), axis=1, keepdims=True
        )
        type_entropy_2 = -np.sum(
            nbr2_type * np.log(np.clip(nbr2_type, 1e-12, 1.0)), axis=1, keepdims=True
        )

        feature_blocks.extend([
            _standardize_block(nbr1_type),      # [Q]
            _standardize_block(nbr2_type),      # [R]
            _standardize_block(nbr4_type),      # [S]
            _standardize_block(agf_type),       # [T,U]
            _standardize_block(type_boundary_1),# [V]
            _standardize_block(type_boundary_2),# [W]
            _standardize_block(type_boundary_4),# [X]
            _standardize_block(type_entropy_1), # [Y]
            _standardize_block(type_entropy_2), # [Z]
        ])

    return np.concatenate(feature_blocks, axis=1)


# ---------------------------------------------------------------------------
# Ward dendrogram analysis
# ---------------------------------------------------------------------------

def _ward_wss_progression(features: np.ndarray, children: np.ndarray):
    """
    Track total within-cluster sum of squares (WSS) at every level of the Ward
    dendrogram. Also records the size-normalized Shannon entropy of the cluster
    size distribution and the marginal merge cost at each level.

    All quantities are computed in a single bottom-up pass over the dendrogram
    in O(n * d) time. No pairwise distance matrix is needed.
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

        mean_norm_sq = (
            np.dot(cluster_sum[node], cluster_sum[node])
            / float(cluster_size[node])
        )
        cluster_sse[node] = max(cluster_sumsq[node] - mean_norm_sq, 0.0)
        delta = max(
            cluster_sse[node] - cluster_sse[left] - cluster_sse[right], 0.0
        )
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
            entropy_evenness_by_k[k] = float(
                entropy_current / np.log(float(k))
            )
        else:
            entropy_evenness_by_k[k] = 0.0

    total_sum = np.sum(features, axis=0)
    total_sumsq = float(np.sum(features * features))
    total_sse = max(
        total_sumsq - np.dot(total_sum, total_sum) / float(n_samples), 0.0
    )
    return wss_by_k, merge_delta_by_k, entropy_evenness_by_k, total_sse


def _hartigan_criterion(wss_by_k: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Compute Hartigan's H statistic for every partition level in the dendrogram.

        H(k) = ( WSS(k) / WSS(k+1) - 1 ) * (n - k - 1)

    H(k) > 10 indicates that the (k+1)-cluster solution is statistically
    preferred over the k-cluster solution. The threshold 10 is not a free
    parameter: it follows from the asymptotic null distribution of the
    F-ratio under the assumption of spherical clusters (Hartigan & Wong 1979,
    Applied Statistics 28:100–108). It corresponds roughly to a p-value of
    0.001 under that null.

    This function returns the H value at every k so that
    select_cluster_count_from_ward_tree can use it as a pre-filter:
    a k-cluster solution is Hartigan-admissible iff H(k-1) > 10, meaning
    that the data statistically support having at least k clusters.

    Parameters
    ----------
    wss_by_k:  (n_samples+1,) array where wss_by_k[k] = WSS with k clusters.
    n_samples: number of cells / data points.

    Returns
    -------
    H_by_k: (n_samples+1,) array; H_by_k[k] = H(k); NaN where undefined.
    """
    H_by_k = np.full(n_samples + 1, np.nan, dtype=np.float64)
    for k in range(1, n_samples - 1):
        wk  = wss_by_k[k]
        wk1 = wss_by_k[k + 1]
        if np.isfinite(wk) and np.isfinite(wk1) and wk1 > 1e-12:
            H_by_k[k] = (wk / wk1 - 1.0) * float(n_samples - k - 1)
    return H_by_k


# ---------------------------------------------------------------------------
# Cut boundary alignment (tie-break only)
# ---------------------------------------------------------------------------

def _compute_cut_boundary_alignment(
    labels: np.ndarray,
    connectivity: sp.csr_matrix,
    features: np.ndarray,
) -> float:
    """
    Score how well a partition follows strong local biological boundaries.

    Used only as a tie-break among cuts with indistinguishable primary scores.
    Higher values mean the partition aligns to stronger local feature gradients.
    """
    connectivity = connectivity.tocoo()
    mask = connectivity.row < connectivity.col
    if not np.any(mask):
        return 0.0

    rows = connectivity.row[mask]
    cols = connectivity.col[mask]
    edge_strength = np.linalg.norm(features[rows] - features[cols], axis=1)
    if edge_strength.size == 0:
        return 0.0

    cut_mask = labels[rows] != labels[cols]
    if not np.any(cut_mask):
        return 0.0

    cut_mean    = float(np.mean(edge_strength[cut_mask]))
    within_mean = (
        float(np.mean(edge_strength[~cut_mask])) if np.any(~cut_mask) else 0.0
    )
    scale = float(np.mean(edge_strength)) + 1e-12
    return float((cut_mean - within_mean) / scale)


# ---------------------------------------------------------------------------
# Automatic cluster count selection
# ---------------------------------------------------------------------------

def select_cluster_count_from_ward_tree(
    features: np.ndarray,
    children: np.ndarray,
    connectivity: sp.csr_matrix | None = None,
) -> tuple[int, dict[str, float | int | None]]:
    """
    Automatically choose the mesoregion count from the full Ward dendrogram.

    The selection proceeds in three stages, all parameter-free:

    Stage 1 — Hartigan admissibility pre-filter (Hartigan & Wong 1979).
    ─────────────────────────────────────────────────────────────────────
    Compute H(k) = (WSS(k)/WSS(k+1) - 1) * (n - k - 1) for every k.
    A k-cluster solution is Hartigan-admissible iff H(k-1) > 10, meaning
    the data statistically support having at least k clusters. Only admissible
    k values are considered in Stage 2. This pre-filter eliminates both
    over-split solutions (where splitting no longer reduces WSS significantly)
    and trivially under-split solutions (k < the minimum supported structure).

    If no k satisfies the Hartigan criterion (e.g. very small or uniform
    components), all k ≥ 2 remain eligible and the method degrades gracefully
    to Stage 2 alone.

    Stage 2 — Stable, balance-aware Calinski–Harabász maximisation.
    ─────────────────────────────────────────────────────────────────────
    Among Hartigan-admissible k values, compute a composite score:

        stable_score(k) = CH(k) × evenness(k) × persistence(k)

    where:
      - CH(k) = (BSS/(k-1)) / (WSS/(n-k)) is the Calinski–Harabász ratio.
      - evenness(k) = H_k / log(k) is the normalised Shannon entropy of the
        cluster size distribution. This discourages degenerate one-vs-rest
        splits that inflate CH by isolating a single outlier cluster.
      - persistence(k) = log1p(Δ_k) / max_k log1p(Δ_k) where Δ_k is the
        Ward merge delta just ABOVE cut k. High persistence means the cut
        sits at a strong Ward merge — i.e. the partition is stable.

    The k with the highest stable_score is selected. Ties within floating-point
    score resolution form a plateau.

    Stage 3 — Boundary-alignment tie-break (plateau only).
    ─────────────────────────────────────────────────────────────────────
    If the plateau contains more than one k, select the k whose partition
    boundary (on the Gabriel graph) aligns best with the strongest local
    biological feature gradients. As a final tie-break, prefer the finest
    (largest k) in the plateau to maximise alignment anchor granularity.

    All three stages are deterministic and parameter-free. The only externally
    sourced constant is the Hartigan threshold of 10, which is derived from the
    asymptotic theory and is not a tunable hyperparameter.
    """
    n_samples = int(features.shape[0])
    if n_samples <= 3:
        return 1, {
            "cluster_count_mode": "small_component_single_cluster",
            "selected_cluster_count": 1,
            "selected_ch_score": None,
        }

    # Stage 0: compute WSS progression and Hartigan H.
    wss_by_k, merge_delta_by_k, entropy_evenness_by_k, total_sse = (
        _ward_wss_progression(features, children)
    )
    H_by_k = _hartigan_criterion(wss_by_k, n_samples)

    # Stage 1: Hartigan admissibility pre-filter.
    # k is admissible iff H(k-1) > 10: the split from k-1 to k clusters is
    # statistically significant under the Hartigan (1975) criterion.
    hartigan_admissible: set[int] = set()
    for k in range(2, n_samples):
        hk_prev = H_by_k[k - 1]
        if np.isfinite(hk_prev) and hk_prev > 10.0:
            hartigan_admissible.add(k)

    # Stage 2: composite score over the admissible range.
    positive_merge_deltas = merge_delta_by_k[merge_delta_by_k > 0]
    max_log_merge = (
        float(np.max(np.log1p(positive_merge_deltas)))
        if positive_merge_deltas.size > 0
        else 0.0
    )
    candidate_scores = []

    for k in range(2, n_samples):
        wss = wss_by_k[k]
        if not np.isfinite(wss) or wss <= 1e-12:
            continue

        between     = max(total_sse - wss, 0.0)
        within_mean = wss / float(max(n_samples - k, 1))
        if within_mean <= 0:
            continue

        ch_score = (between / float(max(k - 1, 1))) / within_mean
        if not np.isfinite(ch_score):
            continue

        evenness = float(np.clip(entropy_evenness_by_k[k], 0.0, 1.0))
        balanced_ch = float(ch_score * evenness)

        if max_log_merge > 0.0:
            persistence = float(
                np.log1p(merge_delta_by_k[k]) / max_log_merge
            )
        else:
            persistence = 1.0
        persistence = float(np.clip(persistence, 0.0, 1.0))

        stable_score = float(balanced_ch * persistence)

        candidate_scores.append((
            stable_score,
            balanced_ch,
            float(ch_score),
            evenness,
            persistence,
            float(merge_delta_by_k[k]),
            int(k),
        ))

    if not candidate_scores:
        return 1, {
            "cluster_count_mode": "ward_tree_no_valid_cut",
            "selected_cluster_count": 1,
            "selected_ch_score": None,
            "hartigan_admissible_count": len(hartigan_admissible),
        }

    # Apply Hartigan pre-filter if any admissible k values exist.
    hartigan_filtered = [r for r in candidate_scores if r[6] in hartigan_admissible]
    if hartigan_filtered:
        candidate_scores = hartigan_filtered
        hartigan_filter_applied = True
    else:
        # Graceful fallback: Hartigan criterion found no admissible k.
        # This occurs for very small or near-uniform components; proceed
        # with the full candidate set.
        hartigan_filter_applied = False

    candidate_scores.sort(reverse=True)

    # Stage 3a: identify the score plateau.
    stable_values = np.array(
        [r[0] for r in candidate_scores], dtype=np.float64
    )
    unique_stable = np.unique(stable_values)
    positive_gaps = np.diff(np.sort(unique_stable))
    positive_gaps = positive_gaps[positive_gaps > 0]
    score_resolution = float(np.min(positive_gaps)) if positive_gaps.size > 0 else 0.0
    best_stable = float(candidate_scores[0][0])
    plateau_tol = max(score_resolution, 1e-12)
    plateau_records = [
        r for r in candidate_scores
        if r[0] >= best_stable - plateau_tol
        or np.isclose(r[0], best_stable, rtol=1e-6, atol=plateau_tol)
    ]

    # Stage 3b: boundary-alignment tie-break within the plateau.
    boundary_scores = []
    for record in plateau_records:
        if connectivity is None or connectivity.nnz == 0:
            boundary_scores.append(0.0)
        else:
            lbl = labels_from_ward_tree(children, n_samples, int(record[6]))
            boundary_scores.append(
                _compute_cut_boundary_alignment(lbl, connectivity, features)
            )

    plateau_records = [
        r + (float(bs),)
        for r, bs in zip(plateau_records, boundary_scores)
    ]
    plateau_records.sort(
        key=lambda r: (r[7], r[6], r[0], r[1], r[2], r[3], r[4], r[5]),
        reverse=True,
    )

    boundary_values = np.array(
        [r[7] for r in plateau_records], dtype=np.float64
    )
    unique_boundary = np.unique(boundary_values)
    pos_b_gaps = np.diff(np.sort(unique_boundary))
    pos_b_gaps = pos_b_gaps[pos_b_gaps > 0]
    boundary_resolution = (
        float(np.min(pos_b_gaps)) if pos_b_gaps.size > 0 else 0.0
    )

    (
        best_stable_score,
        best_balanced_ch,
        best_ch,
        best_evenness,
        best_persistence,
        best_delta,
        best_k,
        best_boundary_alignment,
    ) = plateau_records[0]

    boundary_plateau_tol = max(boundary_resolution, 1e-12)
    boundary_plateau = [
        r for r in plateau_records
        if r[7] >= best_boundary_alignment - boundary_plateau_tol
        or np.isclose(r[7], best_boundary_alignment, rtol=1e-6, atol=boundary_plateau_tol)
    ]

    # Hartigan H value at the selected k (for diagnostics).
    selected_h = float(H_by_k[best_k]) if np.isfinite(H_by_k[best_k]) else None
    h_prev = float(H_by_k[best_k - 1]) if best_k > 1 and np.isfinite(H_by_k[best_k - 1]) else None

    return best_k, {
        "cluster_count_mode": (
            "hartigan_filtered_stable_balanced_ch_with_boundary_tiebreak"
            if hartigan_filter_applied
            else "fallback_stable_balanced_ch_with_boundary_tiebreak"
        ),
        "selected_cluster_count":        int(best_k),
        "selected_ch_score":             float(best_ch),
        "selected_balanced_ch_score":    float(best_balanced_ch),
        "selected_size_evenness":        float(best_evenness),
        "selected_merge_persistence":    float(best_persistence),
        "selected_stable_score":         float(best_stable_score),
        "selected_merge_delta":          float(best_delta),
        "selected_boundary_alignment":   float(best_boundary_alignment),
        "selected_hartigan_h":           selected_h,
        "selected_hartigan_h_prev":      h_prev,
        "hartigan_admissible_count":     int(len(hartigan_admissible)),
        "hartigan_filter_applied":       bool(hartigan_filter_applied),
        "selected_score_resolution":     float(score_resolution),
        "selected_plateau_count":        int(len(plateau_records)),
        "selected_boundary_resolution":  float(boundary_resolution),
        "selected_boundary_plateau_count": int(len(boundary_plateau)),
        "selected_by_boundary_tiebreak": bool(
            len(plateau_records) > 1
            and len(np.unique(boundary_values)) > 1
        ),
        "selected_by_finest_plateau_tiebreak": bool(len(boundary_plateau) > 1),
        "candidate_cut_count":           int(len(candidate_scores)),
    }


# ---------------------------------------------------------------------------
# Dendrogram cut
# ---------------------------------------------------------------------------

def labels_from_ward_tree(
    children: np.ndarray, n_samples: int, n_clusters: int
) -> np.ndarray:
    """
    Cut a Ward dendrogram deterministically at a given number of clusters.

    Uses a union-find structure with path compression and union by rank for
    O(n α(n)) total time (where α is the inverse Ackermann function). The
    result is unique and does not depend on traversal order.
    """
    n_samples  = int(n_samples)
    n_clusters = int(max(1, min(n_clusters, n_samples)))
    if n_clusters == 1:
        return np.zeros(n_samples, dtype=int)
    if n_clusters == n_samples:
        return np.arange(n_samples, dtype=int)

    parent   = np.arange(n_samples, dtype=int)
    rank     = np.zeros(n_samples, dtype=np.int8)
    node_rep = np.arange(2 * n_samples - 1, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
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

    for step in range(n_samples - n_clusters):
        left, right = map(int, children[step])
        node_rep[n_samples + step] = union(node_rep[left], node_rep[right])

    roots = np.array([find(i) for i in range(n_samples)], dtype=int)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(int)


# ---------------------------------------------------------------------------
# Per-component clustering
# ---------------------------------------------------------------------------

def _cluster_connected_component(
    features: np.ndarray,
    connectivity: sp.csr_matrix,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    """
    Cluster one connected tissue component with spatially constrained Ward
    agglomeration and automatic cut selection.

    Spatial connectivity is enforced by passing the Gabriel sub-graph as the
    Ward connectivity mask, which prevents the algorithm from merging cells that
    are not spatially adjacent. This guarantees contiguous mesoregions.
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

    n_clusters, diagnostics = select_cluster_count_from_ward_tree(
        features, model.children_, connectivity=connectivity
    )
    labels = labels_from_ward_tree(model.children_, n_cells, n_clusters)

    diagnostics = dict(diagnostics)
    diagnostics["component_size"]     = n_cells
    diagnostics["ward_merge_count"]   = int(model.children_.shape[0])
    diagnostics["feature_dim"]        = int(features.shape[1])
    return labels, diagnostics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cluster_cells_spatial(
    adata: AnnData,
    spatial_key: str = "spatial",
    label_key: str = "cell_type_annot",
) -> np.ndarray:
    """
    Deterministic, parameter-free, rotation-invariant mesoregion clustering.

    This function exposes zero tunable parameters to the user. The mesoregion
    partition is completely determined by the data (cell positions, expression
    profiles, and, when available, cell type annotations).

    Pipeline
    ────────
    1. Gabriel cell-contact graph (Jaromczyk & Toussaint 1992).
       Uniquely determined by cell positions; no k, radius, or bandwidth.

    2. Multi-scale biology-aware feature matrix.
       Combines:
         a) own expression and dyadic-hop diffused expression (hops 1,2,4,8)
            for local-to-meso-scale molecular context;
         b) rotation- and reflection-invariant Angular Gradient Features (AGF)
            at harmonic orders {1,2} computed from Gabriel edges (Singhal et al.
            2024, Nat Genet 56:431), enabling symmetric-region separation;
         c) expression boundary signals at each hop-transition scale;
         d) local graph density cues;
         e) cell-type neighborhood histograms and their AGF counterparts.
       All blocks are independently Z-scored and normalized by sqrt(d).

    3. Spatially constrained Ward agglomeration per connected component.
       The Gabriel graph acts as the Ward connectivity mask, guaranteeing
       spatially contiguous mesoregions.

    4. Automatic cut selection via Hartigan (1975) + balanced Calinski–Harabász.
       a) Hartigan admissibility pre-filter: retain only k values where the
          split from k-1 to k clusters is statistically significant
          (H(k-1) > 10; Hartigan & Wong 1979, Applied Statistics 28:100–108).
       b) Among admissible k values, maximise the stable Calinski–Harabász
          score (CH weighted by cluster-size entropy and Ward merge persistence).
       c) Boundary-alignment tie-break: among plateau k values, prefer the cut
          that aligns best with strong local biological boundaries.

    Parameters
    ──────────
    adata:       AnnData object with .obsm[spatial_key] (n×2 coordinates),
                 .obsm["X_pca"] or .X (expression), and optionally
                 .obs[label_key] (cell type annotations).
    spatial_key: key in adata.obsm for 2-D spatial coordinates.
    label_key:   key in adata.obs for cell type annotation strings.

    Returns
    ───────
    labels: (n_cells,) integer array of mesoregion assignments.
            Diagnostics are stored in adata.uns["incent_clustering"].
    """
    coords  = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = coords.shape[0]
    if n_cells == 0:
        return np.zeros(0, dtype=int)
    if n_cells == 1:
        return np.zeros(1, dtype=int)

    adjacency = build_spatial_graph(coords)
    if adjacency.nnz == 0:
        return np.zeros(n_cells, dtype=int)

    features = build_biology_oriented_features(adata, adjacency, label_key=label_key)

    n_components, component_ids = connected_components(
        adjacency, directed=False, return_labels=True
    )
    labels = np.full(n_cells, -1, dtype=int)
    next_label = 0
    component_diagnostics = []

    for component_id in range(int(n_components)):
        idx = np.where(component_ids == component_id)[0]
        if idx.size == 0:
            continue

        component_labels, diagnostics = _cluster_connected_component(
            features[idx],
            adjacency[idx][:, idx].tocsr(),
        )
        labels[idx] = component_labels + next_label
        next_label += int(component_labels.max()) + 1
        diagnostics = dict(diagnostics)
        diagnostics["component_id"] = int(component_id)
        component_diagnostics.append(diagnostics)

    adata.uns["incent_clustering"] = {
        "graph_mode":             "gabriel",
        "feature_mode":           "multiscale_agf_biology_context",
        "agf_orders":             [1, 2],
        "diffusion_hops":         [1, 2, 4, 8],
        "cluster_count_criterion": "hartigan_balanced_calinski_harabasz",
        "n_components":           int(n_components),
        "n_clusters":             int(len(np.unique(labels))),
        "component_diagnostics":  component_diagnostics,
        "cluster_sizes":          np.bincount(
            labels, minlength=int(labels.max()) + 1
        ).astype(int).tolist(),
    }

    return labels