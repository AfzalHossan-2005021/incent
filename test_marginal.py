import numpy as np

def sim():
    pA = np.random.rand(10)
    pB = np.random.rand(12)
    Pi_cluster = np.random.rand(10, 12) * 0.1
    # suppose clusters 0..4 in A match perfectly with 0..4 in B
    for i in range(5):
        Pi_cluster[i, i] = pA[i] * 0.8
        
    preserved_A = np.sum(Pi_cluster, axis=1) # mass conserved for each A cluster
    ratio_A = preserved_A / (pA + 1e-9)
    
    strong_A = np.where(ratio_A > 0.5)[0]
    print(f"strong A: {strong_A}")

sim()
