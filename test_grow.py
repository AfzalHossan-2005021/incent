import numpy as np
from sklearn.neighbors import NearestNeighbors

def test_grow():
    # Simulate 20 clusters in A, 24 in B
    np.random.seed(42)
    spatial_A = np.random.rand(20, 2)
    spatial_B = np.random.rand(24, 2)
    
    sizes_A = np.random.randint(100, 500, size=20)
    sizes_B = np.random.randint(100, 500, size=24)
    
    Pi = np.random.rand(20, 24) * 0.01
    
    # Make a true core of 5-6 clusters match heavily
    for a, b in zip(range(5), range(6)):
        Pi[a, b] = 100.0
    Pi = Pi / Pi.sum()
    
    target_coverage = 0.6  # We want 60% of total cells
    
    nn_A = NearestNeighbors(n_neighbors=min(6, len(spatial_A))).fit(spatial_A)
    adj_A = nn_A.kneighbors_graph(mode='connectivity').toarray()
    
    nn_B = NearestNeighbors(n_neighbors=min(6, len(spatial_B))).fit(spatial_B)
    adj_B = nn_B.kneighbors_graph(mode='connectivity').toarray()
    
    idx_max = np.argmax(Pi)
    seed_A, seed_B = np.unravel_index(idx_max, Pi.shape)
    
    core_A = {seed_A}
    core_B = {seed_B}
    
    sum_cells_A = sizes_A[seed_A]
    sum_cells_B = sizes_B[seed_B]
    
    total_A = np.sum(sizes_A)
    total_B = np.sum(sizes_B)
    ratio_target = total_A / float(total_B)
    
    min_score = np.max(Pi) * 0.01
    
    while sum_cells_A / total_A < target_coverage or sum_cells_B / total_B < target_coverage:
        cand_A = set()
        for c in core_A: cand_A.update(np.where(adj_A[c])[0])
        cand_A = cand_A - core_A
        
        cand_B = set()
        for c in core_B: cand_B.update(np.where(adj_B[c])[0])
        cand_B = cand_B - core_B
        
        if not cand_A and not cand_B:
            break
            
        best_cand_A, best_score_A = None, -1
        for c in cand_A:
            score = sum(Pi[c, j] for j in core_B)
            if score > best_score_A:
                best_score_A = score
                best_cand_A = c
                
        best_cand_B, best_score_B = None, -1
        for c in cand_B:
            score = sum(Pi[i, c] for i in core_A)
            if score > best_score_B:
                best_score_B = score
                best_cand_B = c
                
        can_add_A = best_cand_A is not None and best_score_A > min_score
        can_add_B = best_cand_B is not None and best_score_B > min_score
        
        if not can_add_A and not can_add_B:
            break
            
        current_ratio = sum_cells_A / max(1, sum_cells_B)
        
        added = False
        
        if can_add_A and can_add_B:
            if current_ratio < ratio_target:
                core_A.add(best_cand_A)
                sum_cells_A += sizes_A[best_cand_A]
            else:
                core_B.add(best_cand_B)
                sum_cells_B += sizes_B[best_cand_B]
            added = True
        elif can_add_A:
            core_A.add(best_cand_A)
            sum_cells_A += sizes_A[best_cand_A]
            added = True
        elif can_add_B:
            core_B.add(best_cand_B)
            sum_cells_B += sizes_B[best_cand_B]
            added = True
            
        if not added: break
        
    print(f"Selected {sum_cells_A}/{total_A} in A ({sum_cells_A/total_A:.2f})")
    print(f"Selected {sum_cells_B}/{total_B} in B ({sum_cells_B/total_B:.2f})")
    print(f"Core A: {core_A}")
    print(f"Core B: {core_B}")

test_grow()
