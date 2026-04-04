import numpy as np

def sim():
    np.random.seed(42)
    Pi_cluster = np.zeros((10, 12))
    # sparse matching
    for i in range(10):
        j = min(i, 11)
        Pi_cluster[i, j] = np.random.rand() * 10
        if j+1 < 12:
            Pi_cluster[i, j+1] = np.random.rand() * 2
    
    Pi_cluster = Pi_cluster / Pi_cluster.sum()

    flat_pi = Pi_cluster.flatten()
    sorted_idx = np.argsort(flat_pi)[::-1]
    sorted_cumsum = np.cumsum(flat_pi[sorted_idx])
    total_mass = np.sum(flat_pi)

    cutoff = np.searchsorted(sorted_cumsum, total_mass * 0.4)
    selected_flat_idx = sorted_idx[:cutoff+1]

    strong_A, strong_B = np.unravel_index(selected_flat_idx, Pi_cluster.shape)        

    print(f"Top 40% mass involves {len(np.unique(strong_A))} clusters out of 10")     
    print(f"Top 40% mass involves {len(np.unique(strong_B))} clusters out of 12")     

    cutoff = np.searchsorted(sorted_cumsum, total_mass * 0.2)
    selected_flat_idx2 = sorted_idx[:cutoff+1]
    strong_A2, strong_B2 = np.unravel_index(selected_flat_idx2, Pi_cluster.shape)     
    print(f"Top 20% mass involves {len(np.unique(strong_A2))} clusters out of 10")    
    print(f"Top 20% mass involves {len(np.unique(strong_B2))} clusters out of 12")    

sim()
