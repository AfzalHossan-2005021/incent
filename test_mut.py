import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

def test_algorithm(N_cells, C_A, C_B):
    Pi_cluster = np.random.rand(C_A, C_B)
    
    best_A = np.argmax(Pi_cluster, axis=1)
    best_B = np.argmax(Pi_cluster, axis=0)
    
    pairs = []
    for i in range(C_A):
        j = best_A[i]
        if best_B[j] == i:
            pairs.append((i, j))
            
    print(f"Mutually best pairs: {len(pairs)} out of {min(C_A, C_B)}")
    
    if len(pairs) == 0:
        return
        
    pairs = np.array(pairs)
    # Filter by mass to be robust against noise matches
    masses = Pi_cluster[pairs[:, 0], pairs[:, 1]]
    med_mass = np.median(masses)
    pairs = pairs[masses >= med_mass]
    
    print(f"High-mass mutual pairs: {len(pairs)}")

test_algorithm(100, 60, 70)
