import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, Delaunay
import igraph as ig
import leidenalg
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData


def build_spatial_graph(coords: np.ndarray, method: str = 'knn', k: int = 6, radius: float = None):
    """
    Builds a spatial connectivity graph (adjacency matrix/edge list).
    
    Args:
        coords: Spatial coordinates array shape (N, 2).
        method: 'knn', 'radius', or 'delaunay'.
        k: Number of nearest neighbors (for 'knn').
        radius: Distance threshold (for 'radius').
        
    Returns:
        edges: list of tuples (i, j).
        weights: corresponding edge weights (e.g. 1/distance, or 1.0).
    """
    n_cells = coords.shape[0]
    edges = []
    weights = []
    
    if method == 'knn':
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        seen = set()
        for i in range(n_cells):
            for j_idx in range(1, k+1):
                j = indices[i, j_idx]
                dist = distances[i, j_idx]
                u, v = min(i, j), max(i, j)
                if (u, v) not in seen:
                    seen.add((u, v))
                    edges.append((u, v))
                    weights.append(1.0 / (dist + 1e-5))
    elif method == 'radius':
        if radius is None:
            raise ValueError("Radius must be specified for 'radius' method.")
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=radius)
        for i, j in pairs:
            dist = np.linalg.norm(coords[i] - coords[j])
            edges.append((i, j))
            weights.append(1.0 / (dist + 1e-5))
    elif method == 'delaunay':
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        for i in range(n_cells):
            for j in indices[indptr[i]:indptr[i+1]]:
                if i < j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    edges.append((i, j))
                    weights.append(1.0 / (dist + 1e-5))
    else:
        raise ValueError(f"Unknown spatial graph method {method}.")
        
    return edges, weights


def cluster_cells_spatial(adata: AnnData, spatial_key: str = 'spatial', resolution: float = 1.0, 
                          method: str = 'knn', k: int = 10, radius: float = None) -> np.ndarray:
    """
    Clustering cells into contiguous supercells (mesoregions) using a spatial graph and Leiden.
    
    We focus purely on the spatial graph structure here initially.
    
    Args:
        adata: AnnData object.
        spatial_key: Key in adata.obsm storing spatial coordinates.
        resolution: Resolution parameter for Leiden. Higher = more/smaller clusters.
        method: Structure of spatial graph. 'knn', 'radius', or 'delaunay'.
        k: nearest neighbors.
        
    Returns:
        Cluster labels from 0 to C-1.
    """
    coords = adata.obsm[spatial_key]
    n_cells = coords.shape[0]
    
    # 1. Build Spatial Graph
    edges, weights = build_spatial_graph(coords, method=method, k=k, radius=radius)
    
    # 2. Convert to igraph
    g = ig.Graph(n=n_cells, edges=edges, directed=False)
    g.es['weight'] = weights
    
    # 3. Leiden Community Detection
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution
    )

    labels = np.array(partition.membership)
    return labels

