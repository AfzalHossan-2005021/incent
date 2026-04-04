import numpy as np

def sim():
    np.random.seed(45)
    # Simulate Unbalanced OT results where half of the clusters don't overlap properly
    p_A = np.random.rand(10) * 10
    p_B = np.random.rand(12) * 10

    # Suppose clusters 0-4 strongly match 0-4
    Pi_cluster = np.zeros((10, 12))
    for i in range(10):
        if i < 5:
            Pi_cluster[i, i] = p_A[i] * 0.95 # highly preserved
        else:
            # The remaining clusters scatter a tiny amount or get destroyed
            Pi_cluster[i, 7] = p_A[i] * 0.1
            
    marg_A = np.sum(Pi_cluster, axis=1)
    ratio_A = marg_A / (p_A + 1e-12)
    print("Ratios A:", ratio_A)

    thresh_A = np.percentile(ratio_A[ratio_A > 0], 60) if np.any(ratio_A > 0) else 0.5
    strong_A = np.where(ratio_A >= thresh_A)[0]
    print(f"median > {thresh_A:.2f} -> Strong A: {strong_A}")

sim()
