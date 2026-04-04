import numpy as np

def sim():
    N, M = 10, 10
    Pi_cluster = np.random.rand(10, 10)
    labels_A = np.arange(10)
    labels_B = np.arange(10)
    
    unique_A = np.unique(labels_A)
    # the assumption that lables run 0..N-1
    for i in range(Pi_cluster.shape[0]):
        idx_A = np.where(labels_A == i)[0]
        # this matches the current codebase assumption!
