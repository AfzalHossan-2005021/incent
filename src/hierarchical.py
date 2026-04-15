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


import logging
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist

# ──────────────────────────────────────────────────────────────────────────────
# Re-used private helpers from hierarchical_old.py (C5, C7, C9, C10)
# ──────────────────────────────────────────────────────────────────────────────
def _build_geometric_adjacency(coords: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    n_clusters = len(valid_mask)
    adj = np.zeros((n_clusters, n_clusters), dtype=bool)
    np.fill_diagonal(adj, True)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return adj

    cluster_coords = {}
    cluster_trees = {}
    cluster_spacing = {}

    for c in valid_idx:
        c_coords = coords[labels == c]
        cluster_coords[c] = c_coords
        cluster_trees[c] = cKDTree(c_coords)
        if len(c_coords) > 1:
            k_eff = min(2, len(c_coords))
            d, _ = cluster_trees[c].query(c_coords, k=k_eff)
            cluster_spacing[c] = float(np.percentile(d[:, -1], 99))
        else:
            cluster_spacing[c] = 0.0

    centroids_valid = np.array([cluster_coords[c].mean(axis=0) for c in valid_idx])
    candidate_pairs = set()

    if len(valid_idx) >= 3:
        try:
            tri = Delaunay(centroids_valid)
            simplices = tri.simplices
            for ci, cj in [(0, 1), (0, 2), (1, 2)]:
                u_arr = valid_idx[simplices[:, ci]]
                v_arr = valid_idx[simplices[:, cj]]
                for u, v in zip(u_arr.tolist(), v_arr.tolist()):
                    if u != v:
                        candidate_pairs.add((min(u, v), max(u, v)))
        except Exception:
            for i in range(len(valid_idx)):
                for j in range(i + 1, len(valid_idx)):
                    candidate_pairs.add((valid_idx[i], valid_idx[j]))
    elif len(valid_idx) == 2:
        candidate_pairs.add((min(valid_idx[0], valid_idx[1]), max(valid_idx[0], valid_idx[1])))

    for u, v in candidate_pairs:
        gaps, _ = cluster_trees[u].query(cluster_coords[v], k=1)
        if float(np.min(gaps)) <= cluster_spacing[u] + cluster_spacing[v]:
            adj[u, v] = True
            adj[v, u] = True

    return adj

def _median_intercentroid_spacing(centroids: np.ndarray, valid_mask: np.ndarray) -> float:
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return 1.0

    vc = centroids[valid_idx]
    try:
        tri = Delaunay(vc)
        simplices = tri.simplices
        lengths = []
        for ci, cj in [(0, 1), (0, 2), (1, 2)]:
            diff = vc[simplices[:, ci]] - vc[simplices[:, cj]]
            lengths.append(np.linalg.norm(diff, axis=1))
        all_len = np.concatenate(lengths)
        return float(np.median(all_len)) if all_len.size else 1.0
    except Exception:
        dists = pdist(vc)
        return float(np.median(dists)) if dists.size else 1.0

def _cell_level_contiguous_subset(coords_selected, labels_selected, core_cluster_ids, edge_scale_factor=3.0):
    n = len(coords_selected)
    if n < 4:
        return np.ones(n, dtype=bool)

    try:
        tri = Delaunay(coords_selected)
        simplices = tri.simplices

        all_i, all_j, all_len = [], [], []
        for ci, cj in [(0, 1), (0, 2), (1, 2)]:
            i_arr = simplices[:, ci]
            j_arr = simplices[:, cj]
            lengths = np.linalg.norm(coords_selected[i_arr] - coords_selected[j_arr], axis=1)
            all_i.append(i_arr)
            all_j.append(j_arr)
            all_len.append(lengths)

        all_i = np.concatenate(all_i)
        all_j = np.concatenate(all_j)
        all_len = np.concatenate(all_len)

        median_len = float(np.median(all_len))
        keep = all_len <= edge_scale_factor * median_len

        rows = np.concatenate([all_i[keep], all_j[keep]])
        cols = np.concatenate([all_j[keep], all_i[keep]])

        adj = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
        n_comp, comp_lbl = connected_components(adj, directed=False)

        is_core = np.isin(labels_selected, list(core_cluster_ids))
        if not is_core.any():
            best = int(np.bincount(comp_lbl).argmax())
        else:
            core_per_comp = np.bincount(comp_lbl, weights=is_core.astype(float), minlength=n_comp)
            best = int(np.argmax(core_per_comp))

        return comp_lbl == best

    except Exception as exc:
        logging.debug(f"Cell-level BFS failed ({exc}). Returning full selection.")
        return np.ones(n, dtype=bool)

def _mass_weighted_barycenter(cluster_ids, centroids, masses):
    ids = list(cluster_ids)
    w = masses[ids]
    w_sum = w.sum()
    if w_sum == 0:
        return centroids[ids].mean(axis=0)
    return np.average(centroids[ids], axis=0, weights=w)

# ──────────────────────────────────────────────────────────────────────────────
# New private helpers merged from hierarchical.py (rigid grow-trim — biological geometry)
# ──────────────────────────────────────────────────────────────────────────────
def _get_largest_connected_pair_component(pairs, adj_A, adj_B):
    if not pairs:
        return []
    n_pairs = len(pairs)
    adj_pairs = np.zeros((n_pairs, n_pairs), dtype=bool)
    for i, (uA, vB) in enumerate(pairs):
        for j, (vA, wB) in enumerate(pairs):
            if i != j and (adj_A[uA, vA] or adj_B[vB, wB]):
                adj_pairs[i, j] = True
    n_comp, comp_lbl = connected_components(adj_pairs, directed=False)
    largest = np.bincount(comp_lbl).argmax()
    return [pairs[i] for i in np.where(comp_lbl == largest)[0]]

def _contiguous_expansion_pass(current_pairs, centroids_A, centroids_B, masses_A, masses_B,
                               valid_masses, adj_A, adj_B, num_clusters_A, num_clusters_B):
    pairs = list(current_pairs)
    while True:
        sA = {p[0] for p in pairs}
        sB = {p[1] for p in pairs}
        if len(sA) >= np.sum(valid_masses.sum(axis=1) > 0) or len(sB) >= np.sum(valid_masses.sum(axis=0) > 0):
            break

        frontier_A = {i for i in range(num_clusters_A) if i not in sA and np.any(adj_A[i, list(sA)])}
        frontier_B = {j for j in range(num_clusters_B) if j not in sB and np.any(adj_B[j, list(sB)])}
        if not frontier_A or not frontier_B:
            break

        sA_list = list(sA)
        sB_list = list(sB)
        bary_A = np.average(centroids_A[sA_list], axis=0, weights=masses_A[sA_list])
        bary_B = np.average(centroids_B[sB_list], axis=0, weights=masses_B[sB_list])

        # Rigid alignment on current core (biological geometric consistency)
        if len(sA_list) >= 3 and len(sB_list) >= 3:
            P_c = centroids_A[sA_list] - bary_A
            Q_c = centroids_B[sB_list] - bary_B
            W_cross = valid_masses[np.ix_(sA_list, sB_list)]
            try:
                U, _, Vt = np.linalg.svd(P_c.T @ W_cross @ Q_c)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = U @ Vt
            except:
                R = np.eye(2)
        else:
            R = np.eye(2)

        best_pair = None
        best_physical_match = float('inf')
        for pA in frontier_A:
            for pB in frontier_B:
                if valid_masses[pA, pB] > 0:
                    pA_proj = (centroids_A[pA] - bary_A) @ R + bary_B
                    shadow_dist = np.linalg.norm(pA_proj - centroids_B[pB])
                    # Biological check: projection must be closer than any current neighbor
                    min_dist_to_core = min((np.linalg.norm(centroids_B[pB] - centroids_B[c]) for c in sB_list if adj_B[pB, c]), default=float('inf'))
                    if shadow_dist < min_dist_to_core and shadow_dist < best_physical_match:
                        best_physical_match = shadow_dist
                        best_pair = (pA, pB)
        if best_pair is not None:
            pairs.append(best_pair)
        else:
            break
    return pairs

def _trim_worst_outlier(current_pairs, centroids_A, centroids_B, masses_A, masses_B, valid_masses, adj_A, adj_B):
    if len(current_pairs) < 3:
        return current_pairs, False
    sA_list = [p[0] for p in current_pairs]
    sB_list = [p[1] for p in current_pairs]
    bary_A = np.average(centroids_A[sA_list], axis=0, weights=masses_A[sA_list])
    bary_B = np.average(centroids_B[sB_list], axis=0, weights=masses_B[sB_list])

    # Compute final rigid transform
    try:
        P_c = centroids_A[sA_list] - bary_A
        Q_c = centroids_B[sB_list] - bary_B
        W_cross = valid_masses[np.ix_(sA_list, sB_list)]
        U, _, Vt = np.linalg.svd(P_c.T @ W_cross @ Q_c)
        R_final = U @ Vt
        if np.linalg.det(R_final) < 0:
            Vt[-1, :] *= -1
            R_final = U @ Vt
    except:
        R_final = np.eye(2)

    residuals = {}
    for (pA, pB) in current_pairs:
        # Projection errors vs local geometry (biological outlier test)
        neighbors_A = [c for c in sA_list if c != pA and adj_A[pA, c]]
        min_dist_A = min((np.linalg.norm(centroids_A[pA] - centroids_A[n]) for n in neighbors_A), default=float('inf'))
        res_A = np.linalg.norm((centroids_A[pA] - bary_A) @ R_final + bary_B - centroids_B[pB])

        neighbors_B = [c for c in sB_list if c != pB and adj_B[pB, c]]
        min_dist_B = min((np.linalg.norm(centroids_B[pB] - centroids_B[n]) for n in neighbors_B), default=float('inf'))
        res_B = np.linalg.norm((centroids_B[pB] - bary_B) @ R_final.T + bary_A - centroids_A[pA])

        worst_err = max(res_A - min_dist_A, res_B - min_dist_B)
        if worst_err > 0:
            residuals[(pA, pB)] = max(res_A, res_B)

    if not residuals:
        return current_pairs, False

    worst_pair = max(residuals, key=lambda pair: residuals[pair])
    current_pairs.remove(worst_pair)
    return current_pairs, True

# =============================================================================
# Private helpers — all prefixed with underscore; not part of public API
# =============================================================================
 
def _build_geometric_adjacency(
    coords: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Parameter-free cluster adjacency via KD-tree geometric verification (C5).
 
    Two clusters are adjacent iff their minimum inter-cell gap ≤ the sum of
    their respective 99th-percentile intra-cluster 1-NN distances.  This is
    robust to non-convex tissue (folded cortex, branched structures, holes)
    where cell-level Delaunay triangulation creates phantom long-range edges
    across concavities that span physically separated regions.
 
    Centroid-level Delaunay pre-filters pairs to O(C) macroscopic candidates,
    keeping total cost O(C log C) rather than O(C^2).
    """
    n_clusters = len(valid_mask)
    adj        = np.zeros((n_clusters, n_clusters), dtype=bool)
    np.fill_diagonal(adj, True)
 
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return adj
 
    cluster_coords  = {}
    cluster_trees   = {}
    cluster_spacing = {}
 
    for c in valid_idx:
        cc                 = coords[labels == c]
        cluster_coords[c]  = cc
        cluster_trees[c]   = cKDTree(cc)
        if len(cc) > 1:
            d, _               = cluster_trees[c].query(cc, k=min(2, len(cc)))
            cluster_spacing[c] = float(np.percentile(d[:, -1], 99))
        else:
            cluster_spacing[c] = 0.0
 
    # Centroid Delaunay → candidate edge pairs (vectorised — C10)
    vc_arr          = np.array([cluster_coords[c].mean(axis=0) for c in valid_idx])
    candidate_pairs: set = set()
 
    if len(valid_idx) >= 3:
        try:
            tri = Delaunay(vc_arr)
            s   = tri.simplices
            for ci, cj in [(0, 1), (0, 2), (1, 2)]:
                for u, v in zip(valid_idx[s[:, ci]].tolist(),
                                valid_idx[s[:, cj]].tolist()):
                    if u != v:
                        candidate_pairs.add((min(u, v), max(u, v)))
        except Exception:
            for i in range(len(valid_idx)):
                for j in range(i + 1, len(valid_idx)):
                    candidate_pairs.add((valid_idx[i], valid_idx[j]))
    elif len(valid_idx) == 2:
        candidate_pairs.add((min(valid_idx[0], valid_idx[1]),
                             max(valid_idx[0], valid_idx[1])))
 
    # KD-tree minimum gap verification (C5 — non-convex safe)
    for u, v in candidate_pairs:
        gaps, _ = cluster_trees[u].query(cluster_coords[v], k=1)
        if float(np.min(gaps)) <= cluster_spacing[u] + cluster_spacing[v]:
            adj[u, v] = True
            adj[v, u] = True
 
    return adj
 
 
def _median_intercentroid_spacing(
    centroids: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """
    Median Delaunay edge length among valid cluster centroids (C9).
    Normalising barycenter distances by this scale makes them dimensionless
    and equitable across slices with different cluster granularities: a slice
    with 4x more clusters has 2x smaller characteristic spacing, so raw
    Euclidean distances would be dominated by that slice's geometry.
    """
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return 1.0
    vc = centroids[valid_idx]
    try:
        tri = Delaunay(vc)
        s   = tri.simplices
        lengths = []
        for ci, cj in [(0, 1), (0, 2), (1, 2)]:
            diff = vc[s[:, ci]] - vc[s[:, cj]]
            lengths.append(np.linalg.norm(diff, axis=1))
        all_len = np.concatenate(lengths)
        return float(np.median(all_len)) if all_len.size else 1.0
    except Exception:
        dists = pdist(vc)
        return float(np.median(dists)) if dists.size else 1.0
 
 
def _cell_level_contiguous_subset(
    coords_selected: np.ndarray,
    labels_selected: np.ndarray,
    core_cluster_ids: set,
    edge_scale_factor: float = 3.0,
) -> np.ndarray:
    """
    Return the largest spatially contiguous component of the selected cells
    that contains the seed-core clusters (N10, C7, C10).
 
    Steps:
      1. Delaunay triangulation on selected cells (vectorised — C10).
      2. Filter edges longer than edge_scale_factor × median edge length.
         Removes the same class of phantom long-range connections that
         global Delaunay creates across non-convex tissue, applied here at
         the cell level (same principle as _build_geometric_adjacency at C5).
      3. scipy.sparse connected_components — not dense boolean (C10).
      4. Keep the component containing the most seed-core cluster cells (C7).
    """
    n = len(coords_selected)
    if n < 4:
        return np.ones(n, dtype=bool)
 
    try:
        tri = Delaunay(coords_selected)
        s   = tri.simplices
 
        all_i, all_j, all_len = [], [], []
        for ci, cj in [(0, 1), (0, 2), (1, 2)]:
            i_arr   = s[:, ci]; j_arr = s[:, cj]
            lengths = np.linalg.norm(
                coords_selected[i_arr] - coords_selected[j_arr], axis=1
            )
            all_i.append(i_arr); all_j.append(j_arr); all_len.append(lengths)
 
        all_i   = np.concatenate(all_i)
        all_j   = np.concatenate(all_j)
        all_len = np.concatenate(all_len)
 
        keep = all_len <= edge_scale_factor * float(np.median(all_len))
        rows = np.concatenate([all_i[keep], all_j[keep]])
        cols = np.concatenate([all_j[keep], all_i[keep]])
 
        adj_sp = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n, n),
        )
        n_comp, comp_lbl = connected_components(adj_sp, directed=False)
 
        is_core = np.isin(labels_selected, list(core_cluster_ids))
        if not is_core.any():
            best = int(np.bincount(comp_lbl).argmax())
        else:
            core_per = np.bincount(comp_lbl,
                                   weights=is_core.astype(float),
                                   minlength=n_comp)
            best = int(np.argmax(core_per))
 
        return comp_lbl == best
 
    except Exception as exc:
        logging.debug(
            f"Cell-level BFS failed ({exc}). Returning full selection."
        )
        return np.ones(n, dtype=bool)
 
 
def _mass_weighted_barycenter(
    cluster_ids,
    centroids: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Mass-weighted spatial centroid of a cluster set (C3).
    Updated after every hop so the expansion anchor tracks the true
    geometric centre of the growing region, preventing drift.
    """
    ids   = list(cluster_ids)
    w     = masses[ids]
    w_sum = w.sum()
    if w_sum == 0:
        return centroids[ids].mean(axis=0)
    return np.average(centroids[ids], axis=0, weights=w)
 
 
def _compute_lift_matrix(Pi_cluster: np.ndarray) -> np.ndarray:
    """
    Per-entry lift over the independence null model (N6, N9).
 
    lift[u, v] = Pi[u, v] / (marg_A[u] * marg_B[v] / total_mass)
 
    Interpretation: a lift of k means cluster pair (u, v) carries k times
    more joint mass than expected if cluster assignments were independent —
    the standard enrichment metric used in genomics co-occurrence analysis.
 
    Size bias removed (N9): large clusters with high marginals do not
    automatically score high; only specific co-transport elevates lift.
 
    Safe division: where the null expectation is zero (cluster completely
    absent from the OT plan), lift is set to zero.
    """
    total_mass = float(Pi_cluster.sum())
    if total_mass == 0:
        return np.zeros_like(Pi_cluster)
    marg_A        = Pi_cluster.sum(axis=1)
    marg_B        = Pi_cluster.sum(axis=0)
    null_expected = np.outer(marg_A, marg_B) / total_mass
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(null_expected > 0,
                        Pi_cluster / null_expected,
                        0.0)
    return lift
 
 
def _seed_from_best_edge(
    lift_matrix: np.ndarray,
    adj_A: np.ndarray,
    adj_B: np.ndarray,
    valid_A: np.ndarray,
    valid_B: np.ndarray,
    lift_threshold: float,
) -> list:
    """
    Seed the initial matched pair set from the highest-confidence adjacent
    edge pair, using fully vectorised NumPy (N1) and lift-normalised scores (N9).
 
    Complexity: O(E_A × E_B) ≈ O(C^2) — replacing the O(C^4) Python loop.
    For C=150 clusters this is 150^2 = 22,500 operations vs 150^4 = 506M.
 
    A matched edge is a (edgeA, edgeB, orientation) triple:
      edgeA = (uA, vA)  with adj_A[uA, vA] = True
      edgeB = (uB, vB)  with adj_B[uB, vB] = True
    Orientation 1: uA→uB, vA→vB    Score = lift[uA,uB] * lift[vA,vB]
    Orientation 2: uA→vB, vA→uB    Score = lift[uA,vB] * lift[vA,uB]
 
    Seeding from an edge (two co-adjacent pairs) rather than a single pair
    immediately establishes a spatial direction vector, preventing the
    unstable rotation estimate that caused the degenerate-SVD failure (N5).
 
    Returns an empty list if no pair exceeds lift_threshold — the main
    function returns a full-slice fallback in that case.
    """
    valid_adj_A = np.triu(adj_A, k=1) & valid_A[:, None] & valid_A[None, :]
    valid_adj_B = np.triu(adj_B, k=1) & valid_B[:, None] & valid_B[None, :]
    ea_i, ea_j  = np.where(valid_adj_A)
    eb_i, eb_j  = np.where(valid_adj_B)
 
    if ea_i.size == 0 or eb_i.size == 0:
        # No adjacent edge pair exists — single-cluster fallback
        masked = lift_matrix * (valid_A[:, None] * valid_B[None, :]).astype(float)
        flat   = np.argmax(masked)
        uA, uB = np.unravel_index(flat, lift_matrix.shape)
        return [(int(uA), int(uB))] if float(lift_matrix[uA, uB]) >= lift_threshold else []
 
    # N1: All (eA, eB) scores computed in two vectorised outer products
    scores_fwd = (lift_matrix[ea_i[:, None], eb_i[None, :]] *
                  lift_matrix[ea_j[:, None], eb_j[None, :]])
    scores_rev = (lift_matrix[ea_i[:, None], eb_j[None, :]] *
                  lift_matrix[ea_j[:, None], eb_i[None, :]])
    scores     = np.maximum(scores_fwd, scores_rev)
 
    if scores.max() == 0:
        return []
 
    best_ea, best_eb = np.unravel_index(np.argmax(scores), scores.shape)
    uA, vA           = int(ea_i[best_ea]), int(ea_j[best_ea])
 
    if scores_fwd[best_ea, best_eb] >= scores_rev[best_ea, best_eb]:
        uB, vB = int(eb_i[best_eb]), int(eb_j[best_eb])
    else:
        uB, vB = int(eb_j[best_eb]), int(eb_i[best_eb])
 
    # Only include pairs that individually meet the lift threshold
    seed = []
    if float(lift_matrix[uA, uB]) >= lift_threshold:
        seed.append((uA, uB))
    if float(lift_matrix[vA, vB]) >= lift_threshold:
        seed.append((vA, vB))
 
    if not seed:
        # Best edge does not clear threshold; try single best-lift pair
        masked = lift_matrix * (valid_A[:, None] * valid_B[None, :]).astype(float)
        flat   = np.argmax(masked)
        bA, bB = np.unravel_index(flat, lift_matrix.shape)
        if float(lift_matrix[bA, bB]) >= lift_threshold:
            seed = [(int(bA), int(bB))]
 
    return seed
 
 
# =============================================================================
# Main function
# =============================================================================
 
def extract_continuous_macro_section(
    sliceA,
    sliceB,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    Pi_cluster: np.ndarray,
    lift_threshold: float = 5.0,
    spatial_key: str = "spatial",
    marginal_mass_tol: float = 0.05,
    min_cluster_frac: float = 0.10,
    enforce_cell_contiguity: bool = True,
):
    """
    Identify the best co-contiguous macro-section from the coarse cluster
    transport plan and return cell indices, OT-grounded weights, and a
    quality report.
 
    Parameters
    ----------
    sliceA, sliceB : AnnData
    labels_A       : (N,) integer cluster labels for slice A cells
    labels_B       : (M,) integer cluster labels for slice B cells
    Pi_cluster     : (C_A, C_B) coarse OT plan from partial FGW
 
    lift_threshold : float, default 5.0
        Minimum lift over the independence null for a cluster pair to be
        admitted as a seed.  Lift k = k-fold enrichment above chance.
        Null expectation per entry: total_mass / (C_A * C_B).
        5x matches the standard enrichment threshold used in genomics
        co-occurrence analyses.  Growth floor is always 1x (above independence).
 
    marginal_mass_tol : float, default 0.05
        Adaptive stopping: expansion halts when the best available frontier
        pair has lift < marginal_mass_tol * mean_lift_of_current_set (C3).
        Captures diminishing biological signal at the region boundary without
        fixing a maximum number of expansion hops.
 
    min_cluster_frac : float, default 0.10
        Minimum fraction of cells that must be selected before the result is
        flagged as degenerate (N11).  A UserWarning is emitted; the calling
        code should inspect quality['is_degenerate'] and optionally fall back.
 
    enforce_cell_contiguity : bool, default True
        Run BFS on the selected cells' Delaunay graph to guarantee spatial
        contiguity at the cell level, not just at the cluster level (N10).
        Disabling is useful only for diagnostic purposes.
 
    Returns
    -------
    idx_A     : (n_A,) int
    idx_B     : (n_B,) int
    weights_A : (n_A,) float  Pi-evidence per-cell weights summing to 1 (C6)
    weights_B : (n_B,) float  analogous for slice B
    quality   : dict with keys:
                  mass_coverage  — fraction of Pi in selection
                  mean_lift      — mean lift of selected cluster pairs
                  n_clusters_A/B — selected cluster counts
                  n_cells_A/B    — selected cell counts
                  is_degenerate  — bool flag
                  dist_A_all     — (N,) distance to selected boundary (vis only)
                  dist_B_all     — (M,) analogous
    """
    N          = sliceA.shape[0]
    M          = sliceB.shape[0]
    coords_A   = np.asarray(sliceA.obsm[spatial_key], dtype=np.float64)
    coords_B   = np.asarray(sliceB.obsm[spatial_key], dtype=np.float64)
    num_cls_A, num_cls_B = Pi_cluster.shape
    total_mass = float(Pi_cluster.sum())
 
    # ── Canonical fallback ─────────────────────────────────────────────────
    def _fallback(reason: str):
        logging.warning(
            f"[HOT] extract_continuous_macro_section: {reason}. "
            "Returning all cells with uniform weights."
        )
        q = dict(
            mass_coverage  = 1.0,
            mean_lift      = float(_compute_lift_matrix(Pi_cluster).mean()),
            n_clusters_A   = num_cls_A,
            n_clusters_B   = num_cls_B,
            n_cells_A      = N,
            n_cells_B      = M,
            is_degenerate  = True,
            dist_A_all     = np.zeros(N),
            dist_B_all     = np.zeros(M),
        )
        return np.arange(N), np.arange(M), np.ones(N)/N, np.ones(M)/M, q
 
    if total_mass == 0:
        return _fallback("Pi_cluster has zero total mass")
 
    # ── Step 1: Centroids, validity flags, per-cluster cell-mass fractions ─
    centroids_A = np.zeros((num_cls_A, 2))
    centroids_B = np.zeros((num_cls_B, 2))
    valid_A     = np.zeros(num_cls_A, dtype=bool)
    valid_B     = np.zeros(num_cls_B, dtype=bool)
    masses_A    = np.zeros(num_cls_A)   # fraction of A cells per cluster
    masses_B    = np.zeros(num_cls_B)
 
    for i in range(num_cls_A):
        mask = labels_A == i
        if mask.any():
            centroids_A[i] = coords_A[mask].mean(axis=0)
            valid_A[i]     = True
            masses_A[i]    = mask.sum() / N
 
    for i in range(num_cls_B):
        mask = labels_B == i
        if mask.any():
            centroids_B[i] = coords_B[mask].mean(axis=0)
            valid_B[i]     = True
            masses_B[i]    = mask.sum() / M
 
    # ── Step 2: C5 — Parameter-free geometric cluster adjacency ───────────
    adj_A = _build_geometric_adjacency(coords_A, labels_A, valid_A)
    adj_B = _build_geometric_adjacency(coords_B, labels_B, valid_B)
 
    # ── Step 3: N6, N9 — Lift matrix ──────────────────────────────────────
    lift_matrix = _compute_lift_matrix(Pi_cluster)
 
    # ── Step 4: C9 — Scale factors for dimensionless barycenter distances ─
    scale_A = max(_median_intercentroid_spacing(centroids_A, valid_A), 1e-8)
    scale_B = max(_median_intercentroid_spacing(centroids_B, valid_B), 1e-8)
 
    # ── Step 5: N1, N9, N6 — Vectorised edge-based seeding ────────────────
    seed_pairs = _seed_from_best_edge(
        lift_matrix, adj_A, adj_B, valid_A, valid_B, lift_threshold
    )
    if not seed_pairs:
        return _fallback(
            f"No cluster pairs exceed lift_threshold={lift_threshold}. "
            "Try reducing lift_threshold or increasing Leiden resolution"
        )
 
    # Seed pairs from _seed_from_best_edge are co-adjacent by construction:
    # (uA, vA) is a Delaunay edge in adj_A and (uB, vB) in adj_B, so
    # adj_A[uA,vA]=True AND adj_B[uB,vB]=True — the AND co-contiguity
    # requirement (N2) is satisfied for the initial seed without a separate
    # component step.
    mapped_pairs = list(seed_pairs)
 
    # Core cluster IDs used for cell-level BFS seeding (N10)
    core_cluster_ids_A = {p[0] for p in mapped_pairs}
    core_cluster_ids_B = {p[1] for p in mapped_pairs}
 
    # ── Step 6: Adaptive expansion ─────────────────────────────────────────
    #
    # Design rationale (N3, N4, N5, N7, N8):
    # ----------------------------------------
    # The prior "Grow-Trim-Grow" (GTC) loop used SVD to fit a rigid rotation R
    # at every expansion step, then projected each candidate cluster through R
    # to obtain a "shadow" position in the target slice coordinate frame.
    # Candidates were accepted if their shadow distance was below the local
    # Voronoi radius, and poorly fitting clusters were trimmed to correct R.
    #
    # This approach has four fundamental flaws:
    #   (a) Serial tissue sections are NOT related by rigid body transforms.
    #       Differential cutting angle, anisotropic tissue compression during
    #       staining, and registration artefacts all produce non-rigid fields.
    #   (b) Rank-1 SVD under collinear early-stage centroids (< 4 non-collinear
    #       matched pairs) produces a numerically arbitrary rotation direction —
    #       a silent failure numpy SVD never raises an exception for.
    #   (c) W_cross = Pi[ix(sA, sB)] included all cross-combinations of
    #       currently matched clusters, not just confirmed matched pairs.
    #   (d) The trim step used different scale normalisation for detection
    #       (subtracted Voronoi radius) vs. ranking (raw maximum residual).
    #
    # Replacement: scale-normalised barycenter distance with lift-based
    # admission (N6) and dynamic mass-weighted barycenter update (C3).
    # This is parameter-free with respect to tissue geometry, requires no
    # rigid-body assumption, and has a clear null-model interpretation.
 
    max_iter = max(int(valid_A.sum()), int(valid_B.sum()))
 
    for _ in range(max_iter):
        strong_A = {p[0] for p in mapped_pairs}
        strong_B = {p[1] for p in mapped_pairs}
 
        if (len(strong_A) >= int(valid_A.sum()) and
                len(strong_B) >= int(valid_B.sum())):
            break
 
        # C3: Dynamic mass-weighted barycenter — updated after every hop
        bary_A = _mass_weighted_barycenter(strong_A, centroids_A, masses_A)
        bary_B = _mass_weighted_barycenter(strong_B, centroids_B, masses_B)
 
        # Topological frontier: clusters adjacent to the current strong set
        sa_arr = np.array(list(strong_A), dtype=int)
        sb_arr = np.array(list(strong_B), dtype=int)
        cand_A = [c for c in np.where(np.any(adj_A[sa_arr, :], axis=0))[0]
                  if c not in strong_A and valid_A[c]]
        cand_B = [c for c in np.where(np.any(adj_B[sb_arr, :], axis=0))[0]
                  if c not in strong_B and valid_B[c]]
 
        if not cand_A or not cand_B:
            break
 
        # C3: Mean lift of current set for adaptive stopping threshold
        mean_lift = float(np.mean([lift_matrix[p[0], p[1]]
                                   for p in mapped_pairs]))
 
        best_pair  = None
        best_score = float("inf")
        best_lift  = 0.0
 
        for pA in cand_A:
            # C9: Divide by scale → dimensionless → equitable across slices
            norm_dA = np.linalg.norm(centroids_A[pA] - bary_A) / scale_A
            for pB in cand_B:
                lft = float(lift_matrix[pA, pB])
                if lft < 1.0:           # N6: reject sub-null pairs
                    continue
                norm_dB  = np.linalg.norm(centroids_B[pB] - bary_B) / scale_B
                score    = norm_dA + norm_dB  # dimensionless combined distance
                # C11: explicit tie-breaking by lift (descending) ensures
                # fully deterministic output regardless of dict/set ordering
                if score < best_score or (score == best_score
                                          and lft > best_lift):
                    best_score = score
                    best_lift  = lft
                    best_pair  = (pA, pB)
 
        if best_pair is None:
            break
 
        # C3: Adaptive stopping — marginal lift below tolerance
        if best_lift < marginal_mass_tol * mean_lift:
            break
 
        # N2: When appending a new pair (pA, pB) to mapped_pairs, both
        # pA is in cand_A (adjacent to ≥1 cluster in strong_A via adj_A)
        # and pB is in cand_B (adjacent to ≥1 cluster in strong_B via adj_B).
        # The AND condition is thus satisfied structurally by the frontier
        # construction — no explicit post-hoc check is required.
        mapped_pairs.append(best_pair)
 
    # ── Step 7: Cell index extraction ─────────────────────────────────────
    strong_A_final = list({p[0] for p in mapped_pairs})
    strong_B_final = list({p[1] for p in mapped_pairs})
 
    idx_A = np.where(np.isin(labels_A, strong_A_final))[0]
    idx_B = np.where(np.isin(labels_B, strong_B_final))[0]
 
    if idx_A.size == 0 or idx_B.size == 0:
        return _fallback("Expanded cluster set contains no cells")
 
    # ── Step 8: N10, C7 — Cell-level spatial contiguity via BFS ───────────
    if enforce_cell_contiguity:
        mask_A = _cell_level_contiguous_subset(
            coords_A[idx_A], labels_A[idx_A], core_cluster_ids_A
        )
        mask_B = _cell_level_contiguous_subset(
            coords_B[idx_B], labels_B[idx_B], core_cluster_ids_B
        )
        idx_A = idx_A[mask_A]
        idx_B = idx_B[mask_B]
 
    if idx_A.size == 0 or idx_B.size == 0:
        return _fallback("Cell-level BFS produced an empty selection")
 
    # ── Step 9: C6 — Pi-confidence per-cell weights for G_init ───────────
    # weight(cell i in cluster c_A) = sum of Pi[c_A, c_B] over all c_B in
    # the selected set = total OT evidence from the coarse step for that
    # cluster.  This is grounded in the OT solution and makes no assumption
    # about spatial decay from a geometric boundary.
    strong_A_set  = set(strong_A_final)
    strong_B_set  = set(strong_B_final)
    strong_A_list = sorted(strong_A_set)
    strong_B_list = sorted(strong_B_set)
 
    cluster_wt_A = np.zeros(num_cls_A)
    cluster_wt_B = np.zeros(num_cls_B)
    for c in strong_A_list:
        cluster_wt_A[c] = Pi_cluster[c, strong_B_list].sum()
    for c in strong_B_list:
        cluster_wt_B[c] = Pi_cluster[strong_A_list, c].sum()
 
    weights_A = np.maximum(cluster_wt_A[labels_A[idx_A]], 0.0)
    weights_B = np.maximum(cluster_wt_B[labels_B[idx_B]], 0.0)
    wA_sum = weights_A.sum()
    wB_sum = weights_B.sum()
    weights_A = (weights_A / wA_sum if wA_sum > 0
                 else np.ones(idx_A.size) / idx_A.size)
    weights_B = (weights_B / wB_sum if wB_sum > 0
                 else np.ones(idx_B.size) / idx_B.size)
 
    # ── Step 10: N11, C8 — Quality report ─────────────────────────────────
    selected_pi   = sum(float(Pi_cluster[p[0], p[1]]) for p in mapped_pairs)
    mean_lift_sel = float(np.mean([lift_matrix[p[0], p[1]]
                                   for p in mapped_pairs]))
    mass_coverage = selected_pi / total_mass
 
    is_degenerate = (
        idx_A.size < min_cluster_frac * N
        or idx_B.size < min_cluster_frac * M
        or len(strong_A_final) <= 1
    )
    if is_degenerate:
        warnings.warn(
            f"[HOT] Degenerate macro-section detected: "
            f"{idx_A.size}/{N} A-cells  ({len(strong_A_final)} cluster(s)), "
            f"{idx_B.size}/{M} B-cells  ({len(strong_B_final)} cluster(s)). "
            "Consider reducing lift_threshold or increasing Leiden resolution.",
            UserWarning,
            stacklevel=2,
        )
 
    tree_A        = cKDTree(coords_A[idx_A])
    tree_B        = cKDTree(coords_B[idx_B])
    dist_A_all, _ = tree_A.query(coords_A)
    dist_B_all, _ = tree_B.query(coords_B)
 
    quality = dict(
        mass_coverage  = float(mass_coverage),
        mean_lift      = float(mean_lift_sel),
        n_clusters_A   = len(strong_A_final),
        n_clusters_B   = len(strong_B_final),
        n_cells_A      = int(idx_A.size),
        n_cells_B      = int(idx_B.size),
        is_degenerate  = bool(is_degenerate),
        dist_A_all     = dist_A_all,
        dist_B_all     = dist_B_all,
    )
 
    return idx_A, idx_B, weights_A, weights_B, quality