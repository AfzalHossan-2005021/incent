import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

def test():
    N, M = 10, 10
    Pi = np.random.rand(N, M)
    best_A = np.argmax(Pi, axis=1)
    best_B = np.argmax(Pi, axis=0)
    
    mutual = []
    for i in range(N):
        j = best_A[i]
        if best_B[j] == i:
            mutual.append((i, j))
            
    print(f"Mutual matches: {mutual}")
    
test()
