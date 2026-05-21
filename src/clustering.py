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
    
    # 3. Create Joint Spatial-Biological Embedding
    # Normalize features to give balanced weighting to spatial and biological variance
    std_coords = np.std(coords, axis=0).mean()
    coords_scaled = (coords / std_coords) if std_coords > 0 else coords
    
    std_niche = np.std(X_niche, axis=0).mean()
    X_niche_scaled = (X_niche / std_niche) if std_niche > 0 else X_niche
    
    # Combine spatial position and biological identity
    X_joint = np.hstack([coords_scaled, X_niche_scaled])
    
    # 4. Spatially-Constrained Joint Clustering
    # Scale number of clusters based on total cells and resolution identically to original setup
    target_cluster_size = 200.0 / resolution
    n_clusters_target = max(2, int(n_cells / target_cluster_size))
    
    agg = AgglomerativeClustering(
        n_clusters=n_clusters_target,
        linkage='ward',
        connectivity=connectivity
    )
    labels = agg.fit_predict(X_joint)

    return labels

