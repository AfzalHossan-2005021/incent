import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering


def build_spatial_graph(coords: np.ndarray):
    """
    Builds a spatial connectivity graph (adjacency matrix/edge list).
    
    Args:
        coords: Spatial coordinates array shape (N, 2).
        
    Returns:
        edges: list of tuples (i, j).
        weights: corresponding edge weights (e.g. 1/distance, or 1.0).
    """
    n_cells = coords.shape[0]
    edges = []
    
    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices
    for i in range(n_cells):
        for j in indices[indptr[i]:indptr[i+1]]:
            if i < j:  # Avoid duplicates
                edges.append((i, j))
        
    return edges


def cluster_cells_spatial(
    adata: AnnData, 
    spatial_key: str = 'spatial', 
    feature_key: str = 'X_pca',
    resolution: float = 1.0,
    max_cells_per_subcluster: int = 200
) -> np.ndarray:
    """
    Two-phase clustering:
    1. Global deterministic biological clustering using neighborhood-smoothed features.
    2. Spatial subdivision of large biological clusters into contiguous supercells.
    
    This guarantees 100% deterministic, biologically-aware, and mathematically fixed spatial regions.
    
    Args:
        adata: AnnData object.
        spatial_key: Key in adata.obsm storing spatial coordinates.
        feature_key: Key in adata.obsm or adata.obs for biological features (e.g. 'X_pca', 'cell_type').
        resolution: Resolution parameter (scales number of clusters).
        max_cells_per_subcluster: Target threshold size to trigger spatial subdivision.
        
    Returns:
        Cluster labels from 0 to C-1.
    """
    coords = adata.obsm[spatial_key]
    n_cells = coords.shape[0]
    
    # 1. Build Global Spatial Graph & Connectivity
    edges = build_spatial_graph(coords)
    
    if not edges:
        return np.zeros(n_cells, dtype=int)
        
    row = np.array([e[0] for e in edges] + [e[1] for e in edges])
    col = np.array([e[1] for e in edges] + [e[0] for e in edges])
    data = np.ones(len(row))
    connectivity = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
    
    # 2. Extract and Smooth Biological Features (Neighborhood Aware)
    if feature_key in adata.obsm:
        X_base = np.asarray(adata.obsm[feature_key])
    elif feature_key in adata.obs:
        # e.g., discrete cell types
        import pandas as pd
        X_base = pd.get_dummies(adata.obs[feature_key]).values
    else:
        # Fallback to PCA, then generic X, then spatial spatial coords
        if 'X_pca' in adata.obsm:
            X_base = np.asarray(adata.obsm['X_pca'])
        elif hasattr(adata, 'X') and adata.X is not None:
            X_base = np.asarray(adata.X.todense() if sp.issparse(adata.X) else adata.X)
        else:
            X_base = coords

    # Smoothing: X_niche = (I + GraphNorm) * X_base
    I = sp.eye(n_cells)
    adj_with_self = connectivity + I
    degree = np.array(adj_with_self.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / degree)
    X_niche = D_inv.dot(adj_with_self).dot(X_base)
    
    # 3. Global Biological Niches Clustering
    # We want clusters of size approx `target_size`. 
    target_size = 200.0 / resolution
    # Let's not overestimate the initial biological clusters too much; we just want to separate major biological domains.
    n_global_clusters = max(2, int((n_cells / target_size) / 2))
    
    global_agg = AgglomerativeClustering(
        n_clusters=n_global_clusters,
        linkage='ward'  # Pure biological clustering, no geographic constraints yet
    )
    global_labels = global_agg.fit_predict(X_niche)
    
    # 4. Spatially-Constrained Sub-Clustering
    final_labels = np.zeros(n_cells, dtype=int)
    current_cluster_id = 0
    
    from scipy.sparse.csgraph import connected_components
    
    for g_id in np.unique(global_labels):
        idx = np.where(global_labels == g_id)[0]
        sub_conn = connectivity[idx, :][:, idx]
        
        # Explicitly divide into physically contiguous graph components
        n_comps, comp_labels = connected_components(sub_conn, directed=False)
        
        for comp_id in range(n_comps):
            comp_idx = np.where(comp_labels == comp_id)[0]
            global_comp_idx = idx[comp_idx]
            n_comp_local = len(comp_idx)
            
            # Determine the mathematically ideal number of sub-clusters to maintain target supercell size
            num_subclusters = max(1, int(np.round(n_comp_local / target_size)))
            
            # If the physically contiguous component is sufficiently small, keep it intact
            if num_subclusters == 1:
                final_labels[global_comp_idx] = current_cluster_id
                current_cluster_id += 1
                continue
                
            # If large, break it down geometrically into `num_subclusters`
            comp_conn = sub_conn[comp_idx, :][:, comp_idx]
            
            local_agg = AgglomerativeClustering(
                n_clusters=num_subclusters,
                linkage='ward',
                connectivity=comp_conn
            )
            
            local_labels = local_agg.fit_predict(coords[global_comp_idx])
            
            # Remap localized indices to the global supercell array
            for l_id in np.unique(local_labels):
                final_labels[global_comp_idx[local_labels == l_id]] = current_cluster_id
                current_cluster_id += 1

    return final_labels

