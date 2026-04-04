import numpy as np
import pandas as pd
import anndata
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import itertools
from .INCENT import pairwise_align

def get_surviving_indices(slice1, slice2):
    """
    Finds the original indices of cells that survive cell-type filtering.
    """
    shared_cell_types = pd.Index(slice1.obs['cell_type_annot']).unique().intersection(pd.Index(slice2.obs['cell_type_annot']).unique())
    survivors_1 = np.where(slice1.obs['cell_type_annot'].isin(shared_cell_types))[0]
    survivors_2 = np.where(slice2.obs['cell_type_annot'].isin(shared_cell_types))[0]
    return survivors_1, survivors_2

def find_spatial_portions(adata: anndata.AnnData, max_portions: int = 4, silhouette_threshold: float = 0.35) -> tuple[int, np.ndarray]:
    """
    Detects the number of physical portions (e.g. 1 vs 2 hemispheres, or 4 for a heart) 
    based on spatial clustering. Evaluates up to `max_portions`.
    """
    coords = adata.obsm['spatial']
    
    best_k = 1
    best_labels = np.zeros(len(coords), dtype=int)
    best_score = -1

    for k in range(2, max_portions + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full', n_init=5)
        labels = gmm.fit_predict(coords)
        score = silhouette_score(coords, labels)
        
        if score > best_score and score > silhouette_threshold:
            best_score = score
            best_k = k
            best_labels = labels
            
    return best_k, best_labels

def smart_pairwise_align(sliceA, sliceB, **kwargs):
    """
    Automatically detects varying numbers of structural portions (e.g., matching a 1-portion slice 
    against a 4-portion heart slice) and perfectly aligns the smaller slice to the correct 
    geographic subset of the larger slice.
    """
    
    # 1. Detect structures dynamically based on spatial density
    k_A, labels_A = find_spatial_portions(sliceA)
    k_B, labels_B = find_spatial_portions(sliceB)
    
    print(f"[Smart Align] Slice A portions: {k_A} | Slice B portions: {k_B}")
    
    original_return_obj = kwargs.get('return_obj', False)
    kwargs['return_obj'] = True
    
    # Case 1: Slice A has fewer portions than Slice B
    if k_A < k_B:
        print(f"[Smart Align] Slice A ({k_A} portion) is smaller than Slice B ({k_B} portions). Finding best matching sub-geometry...")
        best_cost = float('inf')
        best_res = None
        best_idx_B = None
        
        # Test all mathematical combinations of K_A portions within Slice B
        for combo in itertools.combinations(range(k_B), k_A):
            idx_B_combo = np.where(np.isin(labels_B, combo))[0]
            sliceB_sub = sliceB[idx_B_combo].copy()
            
            surv_A, surv_B_sub = get_surviving_indices(sliceA, sliceB_sub)
            
            kwargs['sliceB_name'] = kwargs.get('sliceB_name', 'B') + f"_parts{combo}"
            res = pairwise_align(sliceA, sliceB_sub, **kwargs)
            cost = res[4] # using final_obj_gene 
            
            if cost < best_cost:
                best_cost = cost
                best_res = res
                best_idx_B = idx_B_combo
        
        print(f"[Smart Align] Chose Slice B portions mapping to combo (Cost: {best_cost:.4f})")
        
        # Reconstruct full Pi matrix into original global dimensions
        surv_A_best, surv_B_sub_best = get_surviving_indices(sliceA, sliceB[best_idx_B])
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[np.ix_(surv_A_best, best_idx_B[surv_B_sub_best])] = best_res[0]
        
        best_res_list = list(best_res)
        best_res_list[0] = full_pi

    # Case 2: Slice A has more portions than Slice B
    elif k_A > k_B:
        print(f"[Smart Align] Slice B ({k_B} portions) is smaller than Slice A ({k_A} portions). Finding best matching sub-geometry...")
        best_cost = float('inf')
        best_res = None
        best_idx_A = None
        
        # Test all mathematical combinations of K_B portions within Slice A
        for combo in itertools.combinations(range(k_A), k_B):
            idx_A_combo = np.where(np.isin(labels_A, combo))[0]
            sliceA_sub = sliceA[idx_A_combo].copy()
            
            surv_A_sub, surv_B = get_surviving_indices(sliceA_sub, sliceB)
            
            kwargs['sliceA_name'] = kwargs.get('sliceA_name', 'A') + f"_parts{combo}"
            res = pairwise_align(sliceA_sub, sliceB, **kwargs)
            cost = res[4] # using final_obj_gene
            
            if cost < best_cost:
                best_cost = cost
                best_res = res
                best_idx_A = idx_A_combo
        
        print(f"[Smart Align] Chose Slice A portions mapping to combo (Cost: {best_cost:.4f})")
        
        # Reconstruct full Pi matrix into original global dimensions
        surv_A_sub_best, surv_B_best = get_surviving_indices(sliceA[best_idx_A], sliceB)
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[np.ix_(best_idx_A[surv_A_sub_best], surv_B_best)] = best_res[0]
        
        best_res_list = list(best_res)
        best_res_list[0] = full_pi

    # Case 3: Both have the same number of portions
    else:
        print(f"[Smart Align] Slices are structurally identical ({k_A} vs {k_B} portions). Proceeding with standard alignment.")
        surv_A, surv_B = get_surviving_indices(sliceA, sliceB)
        best_res_list = list(pairwise_align(sliceA, sliceB, **kwargs))
        
        # Protect against cell type drop shape mismatches
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[np.ix_(surv_A, surv_B)] = best_res_list[0]
        best_res_list[0] = full_pi

    # Obey the user's original request for return objects
    if not original_return_obj:
        return best_res_list[0] # Just the pi matrix
    return tuple(best_res_list)