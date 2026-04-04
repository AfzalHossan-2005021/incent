import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def test_dbscan():
    coords = np.vstack([
        np.random.normal(0, 1, (100, 2)),
        np.random.normal(10, 1, (100, 2))  # 2nd portion far away
    ])
    
    nn = NearestNeighbors(n_neighbors=6).fit(coords)
    dists, _ = nn.kneighbors(coords)
    eps = np.median(dists[:, 1:]) * 5.0
    
    labels = DBSCAN(eps=eps, min_samples=10).fit_predict(coords)
    print(np.unique(labels, return_counts=True))

test_dbscan()
