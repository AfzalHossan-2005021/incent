import numpy as np

def build_cluster_adj(centroids):
    from scipy.spatial import Delaunay
    from collections import defaultdict
    if len(centroids) < 4:
        return {i: set(range(len(centroids))) - {i} for i in range(len(centroids))}
    try:
        tri = Delaunay(centroids)
        adj = defaultdict(set)
        indptr, indices = tri.vertex_neighbor_vertices
        for i in range(len(centroids)):
            adj[i] = set(indices[indptr[i]:indptr[i+1]])
        return adj
    except Exception:
        # fallback to fully connected if Delaunay fails (e.g. collinear points)
        return {i: set(range(len(centroids))) - {i} for i in range(len(centroids))}

def region_grow(Pi, centroids_A, centroids_B):
    adj_A = build_cluster_adj(centroids_A)
    adj_B = build_cluster_adj(centroids_B)
    
    # Strongest seed
    seed_A, seed_B = np.unravel_index(np.argmax(Pi), Pi.shape)
    
    matched_A = {seed_A}
    matched_B = {seed_B}
    queue = [(seed_A, seed_B)]
    
    thresh = np.max(Pi) * 0.15 # Minimum matching confidence relative to max
    
    while len(queue) > 0:
        curr_A, curr_B = queue.pop(0)
        
        # Grow to neighbors
        for n_A in adj_A[curr_A]:
            if n_A in matched_A: continue
                
            for n_B in adj_B[curr_B]:
                if n_B in matched_B: continue
                
                # Check cross-match confidence
                if Pi[n_A, n_B] >= thresh:
                    matched_A.add(n_A)
                    matched_B.add(n_B)
                    queue.append((n_A, n_B))
                    
    return list(matched_A), list(matched_B)

ca, cb = np.random.rand(10, 2), np.random.rand(10, 2)
Pi = np.eye(10) * 0.1
grow = region_grow(Pi, ca, cb)
print("Grown:", grow)
