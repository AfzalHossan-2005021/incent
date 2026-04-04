import numpy as np
from sklearn.neighbors import NearestNeighbors

def mock_slice(N, C):
    class Obj:
        def __init__(self):
            self.shape = (N, 0)
            # cluster centroids in line 
            cents = np.linspace(0, 10, C)
            self.obsm = {'spatial': np.random.randn(N, 2)}
    return Obj()

def test_extract():
    sliceA = mock_slice(10000, 20)
    sliceB = mock_slice(14000, 24)
    labels_A = np.random.randint(0, 20, 10000)
    labels_B = np.random.randint(0, 24, 14000)
    
    Pi_cluster = np.zeros((20, 24))
    for i in range(10):
        Pi_cluster[i, i] = 100.0  # Top 10 match perfectly
    for i in range(20):
        for j in range(24):
            Pi_cluster[i, j] += np.random.rand() * 0.1
            
    # Copy snippet here
    N, M = sliceA.shape[0], sliceB.shape[0]
    total_mass = np.sum(Pi_cluster)
    num_clusters_A = Pi_cluster.shape[0]
    num_clusters_B = Pi_cluster.shape[1]
    
    centroids_A = np.zeros((num_clusters_A, 2))
    sizes_A = np.zeros(num_clusters_A)
    for c in range(num_clusters_A):
        idx = np.where(labels_A == c)[0]
        if len(idx) > 0:
            centroids_A[c] = sliceA.obsm['spatial'][idx].mean(axis=0)
            sizes_A[c] = len(idx)

    centroids_B = np.zeros((num_clusters_B, 2))
    sizes_B = np.zeros(num_clusters_B)
    for c in range(num_clusters_B):
        idx = np.where(labels_B == c)[0]
        if len(idx) > 0:
            centroids_B[c] = sliceB.obsm['spatial'][idx].mean(axis=0)
            sizes_B[c] = len(idx)

    nn_A = NearestNeighbors(n_neighbors=min(6, num_clusters_A)).fit(centroids_A)
    adj_A = nn_A.kneighbors_graph(mode='connectivity').toarray()
    
    nn_B = NearestNeighbors(n_neighbors=min(6, num_clusters_B)).fit(centroids_B)
    adj_B = nn_B.kneighbors_graph(mode='connectivity').toarray()

    idx_max = np.argmax(Pi_cluster)
    seed_A, seed_B = np.unravel_index(idx_max, Pi_cluster.shape)
    
    core_A = {seed_A}
    core_B = {seed_B}
    
    sum_cells_A = sizes_A[seed_A]
    sum_cells_B = sizes_B[seed_B]
    
    target_coverage = 0.75 
    ratio_target = N / float(M)
    min_score = np.max(Pi_cluster) * 0.005
    
    while (sum_cells_A / N < target_coverage) or (sum_cells_B / M < target_coverage):
        cand_A = set()
        for c in core_A: 
            cand_A.update(np.where(adj_A[c])[0])
        cand_A -= core_A
        
        cand_B = set()
        for c in core_B: 
            cand_B.update(np.where(adj_B[c])[0])
        cand_B -= core_B
        
        if not cand_A and not cand_B: break
            
        best_cand_A, best_score_A = None, -1
        for c in cand_A:
            score = sum(Pi_cluster[c, j] for j in core_B)
            if score > best_score_A:
                best_score_A = score; best_cand_A = c
                
        best_cand_B, best_score_B = None, -1
        for c in cand_B:
            score = sum(Pi_cluster[i, c] for i in core_A)
            if score > best_score_B:
                best_score_B = score; best_cand_B = c
                
        can_add_A = best_cand_A is not None and best_score_A > min_score
        can_add_B = best_cand_B is not None and best_score_B > min_score
        
        if not can_add_A and not can_add_B: break
            
        current_ratio = sum_cells_A / max(1, sum_cells_B)
        added = False
        
        if can_add_A and can_add_B:
            if current_ratio < ratio_target:
                core_A.add(best_cand_A); sum_cells_A += sizes_A[best_cand_A]
            else:
                core_B.add(best_cand_B); sum_cells_B += sizes_B[best_cand_B]
            added = True
        elif can_add_A:
            core_A.add(best_cand_A); sum_cells_A += sizes_A[best_cand_A]; added = True
        elif can_add_B:
            core_B.add(best_cand_B); sum_cells_B += sizes_B[best_cand_B]; added = True
            
        if not added: break
            
    print(f"Selected A: {sum_cells_A}/{N} ({sum_cells_A/N:.2f})")
    print(f"Selected B: {sum_cells_B}/{M} ({sum_cells_B/M:.2f})")
    
test_extract()
