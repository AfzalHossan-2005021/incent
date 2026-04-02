import os
import ot
import time
import torch
import datetime
import numpy as np

from tqdm import tqdm
from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from .utils import fused_gromov_wasserstein_incent, to_dense_array, extract_data_matrix, jensenshannon_divergence_backend, pairwise_msd, to_backend


def pairwise_align(
    sliceA: AnnData, 
    sliceB: AnnData, 
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    use_rep: Optional[str] = None, 
    G_init = None, 
    a_distribution = None, 
    b_distribution = None,
    numItermax: int = 6000, 
    use_gpu: bool = False,
    data_type = np.float32,
    verbose: bool = False, 
    gpu_verbose: bool = True,
    dummy_cell: bool = True,
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    """

    This method is written by Anup Bhowmik, CSE, BUET

    Calculates and returns optimal alignment of two slices of single cell MERFISH data. 
    
    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha:  weight for spatial distance
        gamma: weight for gene expression distance (JSD)
        beta: weight for cell type one hot encoding
        radius: radius for cellular neighborhood

        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        numItermax: Max number of iterations during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        data_type: Data type for backend tensors. Default is float32.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT and cell-type matching metrics.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.
   
    Returns:
        - Alignment of spots (pi).

        If ``return_obj = True``, additionally returns:
        
        - initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene: Objective metrics
        - initial_cell_type_match, final_cell_type_match: Cell-type matching percentages 
    """

    epsilon = 1e-6
    
    # Determine if gpu or cpu is being used
    nx = None
    if use_gpu:
        if torch.cuda.is_available():
            nx = ot.backend.TorchBackend()
            if gpu_verbose:
                print("Using gpu with Pytorch backend.")
        else:
            use_gpu = False
            nx = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("CUDA is not available on your system. Reverting to CPU with Numpy backend.")
    else:
        if torch.cuda.is_available() and gpu_verbose:
            print("Tip: CUDA is available on your system. You can enable GPU support by setting use_gpu=True.")
        else:
            nx = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("Using cpu with Numpy backend.")
    
    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.")   
    
    # ────────────────────── Calculate spatial distances ──────────────────────
    D_A = to_backend(calculate_spatial_distance(sliceA), nx, data_type=data_type)
    D_B = to_backend(calculate_spatial_distance(sliceB), nx, data_type=data_type)

    # ────────────────────── Normalize spatial distances ──────────────────────
    scale = max(D_A.max(), D_B.max()) + epsilon
    D_A = D_A / scale
    D_B = D_B / scale
    
    # ────────────────────── Calculate gene expression dissimilarity ──────────────────────
    cosine_dist_gene_expr = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, eps=epsilon)

    # ────────────────────── Calculate cell-type mismatch penalty ──────────────────────
    M_celltype = calculate_cell_type_mismatch(sliceA, sliceB)


    # Combine gene expression dissimilarity and cell-type mismatch penalty into a single cost matrix M1
    M1_combined = (1 - beta) * cosine_dist_gene_expr + beta * M_celltype
    M1 = to_backend(M1_combined, nx, data_type=data_type)


    # ────────────────────── Calculate neighborhood distribution ──────────────────────
    neighborhood_distribution_sliceA = neighborhood_distribution(sliceA, radius = radius) + epsilon
    neighborhood_distribution_sliceB = neighborhood_distribution(sliceB, radius = radius) + epsilon


    neighborhood_distribution_sliceA = to_backend(neighborhood_distribution_sliceA, nx, data_type=data_type)
    neighborhood_distribution_sliceB = to_backend(neighborhood_distribution_sliceB, nx, data_type=data_type)


    # ────────────────────── Calculate neighborhood dissimilarity ──────────────────────
    js_dist_neighborhood = jensenshannon_divergence_backend(neighborhood_distribution_sliceA, neighborhood_distribution_sliceB)
    M2 = to_backend(js_dist_neighborhood, nx, data_type=data_type)


    # ── Dummy cell augmentation ────────────────────────────────────────────
    # Only add a dummy on the side that actually has a deficit.
    # _has_dummy_src: True if source has fewer cells → need dummy source (birth)
    # _has_dummy_tgt: True if target has fewer cells → need dummy target (death)
    _has_dummy_src = False
    _has_dummy_tgt = False

    if dummy_cell:
        from collections import Counter
        ns, nt = sliceA.shape[0], sliceB.shape[0]
        labels_A = sliceA.obs['cell_type_annot'].values
        labels_B = sliceB.obs['cell_type_annot'].values
        counts_A = Counter(labels_A)
        counts_B = Counter(labels_B)
        all_types = set(counts_A.keys()) | set(counts_B.keys())
        _budget = sum(max(counts_A.get(k, 0), counts_B.get(k, 0)) for k in all_types)
        _w_dummy_src = _budget - ns   # extra weight for dummy source cell (birth)
        _w_dummy_tgt = _budget - nt   # extra weight for dummy target cell (death)
        _has_dummy_src = _w_dummy_src > 0
        _has_dummy_tgt = _w_dummy_tgt > 0

        print(f"[dummy_cell] budget={_budget}, "
              f"dummy_src={'YES (birth)' if _has_dummy_src else 'NO'} w={_w_dummy_src}, "
              f"dummy_tgt={'YES (death)' if _has_dummy_tgt else 'NO'} w={_w_dummy_tgt}")

        _ns_aug = ns + (1 if _has_dummy_src else 0)
        _nt_aug = nt + (1 if _has_dummy_tgt else 0)

        # ---- Augment D_A if dummy source needed ----
        if _has_dummy_src:
            zeros_col = nx.zeros((ns, 1), type_as=D_A)
            D_A_temp = nx.concatenate([D_A, zeros_col], axis=1)
            zeros_row = nx.zeros((1, _ns_aug), type_as=D_A)
            D_A = nx.concatenate([D_A_temp, zeros_row], axis=0)

        # ---- Augment D_B if dummy target needed ----
        if _has_dummy_tgt:
            zeros_col = nx.zeros((nt, 1), type_as=D_B)
            D_B_temp = nx.concatenate([D_B, zeros_col], axis=1)
            zeros_row = nx.zeros((1, _nt_aug), type_as=D_B)
            D_B = nx.concatenate([D_B_temp, zeros_row], axis=0)
        
        # ---- Augment M1: add dummy row/col only where needed ----
        if _has_dummy_tgt:
            mean_col = nx.mean(M1, axis=1)
            mean_col = nx.reshape(mean_col, (-1, 1))
            M1 = nx.concatenate([M1, mean_col], axis=1)
        if _has_dummy_src:
            mean_row = nx.mean(M1, axis=0)
            mean_row = nx.reshape(mean_row, (1, -1))
            M1 = nx.concatenate([M1, mean_row], axis=0)
        if _has_dummy_src and _has_dummy_tgt:
            M1[-1, -1] = 0.0

        # ---- Augment M2: add dummy row/col only where needed ----
        if _has_dummy_tgt:
            mean_col = nx.mean(M2, axis=1)
            mean_col = nx.reshape(mean_col, (-1, 1))
            M2 = nx.concatenate([M2, mean_col], axis=1)
        if _has_dummy_src:
            mean_row = nx.mean(M2, axis=0)
            mean_row = nx.reshape(mean_row, (1, -1))
            M2 = nx.concatenate([M2, mean_row], axis=0)
        if _has_dummy_src and _has_dummy_tgt:
            M2[-1, -1] = 0.0

    # init distributions
    if a_distribution is None:
        if dummy_cell:
            if _has_dummy_src:
                a_vals = np.full(ns + 1, 1.0 / _budget, dtype=np.float64)
                a_vals[-1] = float(_w_dummy_src) / _budget
                a = nx.from_numpy(a_vals)
            else:
                a = nx.ones((ns,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            # uniform distribution, a = array([1/n, 1/n, ...])
            a = nx.ones((sliceA.shape[0],))/sliceA.shape[0]
    else:
        if dummy_cell:
            raise ValueError("Custom a_distribution is not supported with dummy_cell=True.")
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        if dummy_cell:
            if _has_dummy_tgt:
                b_vals = np.full(nt + 1, 1.0 / _budget, dtype=np.float64)
                b_vals[-1] = float(_w_dummy_tgt) / _budget
                b = nx.from_numpy(b_vals)
            else:
                b = nx.ones((nt,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        if dummy_cell:
            raise ValueError("Custom b_distribution is not supported with dummy_cell=True.")
        b = nx.from_numpy(b_distribution)

    a = to_backend(a, nx, data_type=data_type)
    b = to_backend(b, nx, data_type=data_type)
    
    
    # Run OT
    if G_init is not None:
        if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
            # Pad user-provided (ns x nt) G_init to augmented dims
            _gi = np.array(G_init, dtype=np.float64)
            _gi_aug = np.zeros((_ns_aug, _nt_aug), dtype=np.float64)
            _gi_aug[:ns, :nt] = _gi
            G_init = _gi_aug
        G_init = to_backend(G_init, nx, data_type=data_type)
    

    _fgw_extra = {'numItermaxEmd': 500_000} if dummy_cell else {}
    pi, logw = fused_gromov_wasserstein_incent(M1, M2, D_A, D_B, a, b, G_init = G_init, loss_fun='square_loss', alpha= alpha, gamma=gamma, log=True, numItermax=numItermax,verbose=verbose, **_fgw_extra)
    pi = nx.to_numpy(pi)

    # ── Dummy cell: strip dummy row/col, renormalize, report birth/death ────
    if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
        pi_full = pi.copy()

        # Compute birth / death mass before stripping
        birth_mass = float(pi_full[-1, :nt].sum()) if _has_dummy_src else 0.0
        death_mass = float(pi_full[:ns, -1].sum()) if _has_dummy_tgt else 0.0

        # Strip only the dummy row/col that was actually added
        if _has_dummy_src and _has_dummy_tgt:
            pi = pi_full[:ns, :nt]
        elif _has_dummy_src:
            pi = pi_full[:ns, :]      # strip dummy source row only
        elif _has_dummy_tgt:
            pi = pi_full[:, :nt]      # strip dummy target col only

        print(f"[dummy_cell] death_mass: {death_mass:.6f}, birth_mass: {birth_mass:.6f}")

    if isinstance(nx, ot.backend.TorchBackend):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pi


def neighborhood_distribution(slice, radius):
    """
    This method is added by Anup Bhowmik
    Args:
        slice: Slice to get niche distribution for.
        pairwise_distances: Pairwise distances between cells of a slice.
        radius: Radius of the niche.

    Returns:
        niche_distribution: Niche distribution for the slice.
    """

    unique_cell_types = np.array(list(slice.obs['cell_type_annot'].unique()))
    cell_type_to_index = dict(zip(unique_cell_types, list(range(len(unique_cell_types)))))
    cells_within_radius = np.zeros((slice.shape[0], len(unique_cell_types)), dtype=float)

    source_coords = slice.obsm['spatial']
    distances = euclidean_distances(source_coords, source_coords)

    for i in tqdm(range(slice.shape[0])):

        target_indices = np.where(distances[i] <= radius)[0]

        for ind in target_indices:
            cell_type_str_j = str(slice.obs['cell_type_annot'].iloc[ind])
            cells_within_radius[i][cell_type_to_index[cell_type_str_j]] += 1

    return np.array(cells_within_radius)


def calculate_spatial_distance(slice, spatial_key = 'spatial'):
    """
    Calculate spatial distance between cells in a slice.

    Args:
        slice: Slice for which to calculate spatial distance.
        spatial_key: Key for the spatial coordinates in the slice's obsm.

    Returns:
    D: Pairwise spatial distance matrix of the slice.
    """
    
    print("Calculating spatial distance between cells in the slice")

    coordinates = slice.obsm[spatial_key]

    return euclidean_distances(coordinates, coordinates)


def calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, eps = 1e-6):
    """
    Calculate cosine distance between gene expression profiles of slice A and slice B.
    
    Args:
    sliceA: First slice.
    sliceB: Second slice.
    use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity
                between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
    eps: Small constant to add to data matrices to avoid zero vectors.
    
    Returns:
    cosine_dist_gene_expr: Cosine distance matrix between gene expression profiles of slice A and slice B.
    """
    
    print("Calculating cosine distance between gene expression profiles of slice A and slice B")
    
    # Extract and prepare data matrices for cosine distance calculation
    A_X = extract_data_matrix(sliceA, use_rep)
    B_X = extract_data_matrix(sliceB, use_rep)

    # Convert to dense arrays and add small constant to avoid zero vectors
    A_X = to_dense_array(A_X) + eps
    B_X = to_dense_array(B_X) + eps

    # Use sklearn's optimized and numerically stable cosine_distances
    cosine_dist_gene_expr = cosine_distances(A_X, B_X)

    return cosine_dist_gene_expr


def calculate_cell_type_mismatch(sliceA, sliceB):
    """
    Calculate the cell-type mismatch penalty between two slices.

    Args:
        sliceA: First slice.
        sliceB: Second slice.

    Returns:
        M_celltype: Binary matrix indicating cell-type mismatches.
    """

    _lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    _lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)

    M_celltype = (_lab_A[:, None] != _lab_B[None, :]).astype(np.float64)

    return M_celltype


def calculate_neighborhood_similarity(js_dist_neighborhood, pi):
    """
    Calculate neighborhood similarity cost for a given alignment mapping.
    
    Args:
        js_dist_neighborhood: Jensen-Shannon distance matrix of neighborhood distributions.
        pi: Alignment mapping matrix (either uniform G or optimal transport solution).
    
    Returns:
        neighborhood_similarity: Weighted neighborhood dissimilarity cost.
    """
    max_indices = np.argmax(pi, axis=1)
    neighborhood_error = np.zeros(max_indices.shape)
    for i in range(len(max_indices)):
        neighborhood_error[i] = pi[i][max_indices[i]] * js_dist_neighborhood[i][max_indices[i]]
    
    neighborhood_similarity = np.sum(neighborhood_error)
    return neighborhood_similarity


def cell_type_matching(cell_type_mismatch, pi_mat):
    """
    Compute cell-type matching percentage from a pre-computed mismatch matrix.
    
    Args:
        cell_type_mismatch: Binary mismatch matrix (1 = mismatch, 0 = match) from calculate_cell_type_mismatch().
        pi_mat: Alignment mapping matrix (probabilistic transport plan).
    
    Returns:
        Percentage of transported mass representing cell-type matches (0-100).
    """
    M_match = 1 - cell_type_mismatch
    expected_matches = np.sum(M_match * pi_mat)
    total_mass = np.sum(pi_mat)
    
    if total_mass > 0:
        return (expected_matches / total_mass)
    return 0.0


def calculate_gene_expression_similarity(cosine_dist_gene_expr, pi):
    """
    Calculate gene expression similarity cost for a given alignment mapping.
    
    Args:
        cosine_dist_gene_expr: Cosine distance matrix of gene expression profiles.
        pi: Alignment mapping matrix (either uniform G or optimal transport solution).
    
    Returns:
        gene_expression_similarity: Weighted gene expression dissimilarity cost.
    """
    gene_expression_similarity = np.sum(cosine_dist_gene_expr * pi)
    return gene_expression_similarity


def calculate_performance_metrics(final_pi, init_pi=None, js_dist_neighborhood=None, cosine_dist_gene_expr=None, 
                               cell_type_mismatch=None, sliceA=None, sliceB=None, use_rep=None, radius=100.0):
    """
    Calculate all similarity metrics for alignment quality assessment.
    
    Args:
        final_pi: Final optimal transport alignment mapping (required).
        init_pi: Initial alignment mapping (optional). If None, uses uniform distribution.
        js_dist_neighborhood: Jensen-Shannon distance matrix of neighborhood distributions (optional).
                             If not provided and sliceA, sliceB, radius are given, will be calculated.
        cosine_dist_gene_expr: Cosine distance matrix of gene expression profiles (optional).
                              If not provided and sliceA, sliceB, use_rep are given, will be calculated.
        sliceA: First slice for calculating missing distance matrices (optional).
        sliceB: Second slice for calculating missing distance matrices (optional).
        use_rep: Representation key for gene expression (optional, used with cosine_dist_gene_expr calculation).
        radius: Radius for neighborhood calculation (optional, used with js_dist_neighborhood calculation).
    
    Returns:
        Dictionary with keys: 'initial_obj_neighbor', 'initial_obj_gene', 
                              'final_obj_neighbor', 'final_obj_gene',
                              'initial_cell_type_match', 'final_cell_type_match'
                              
    Raises:
        ValueError: If required parameters for distance calculation are missing.
    """
    # Use uniform distribution if init_pi not provided
    if init_pi is None:
        init_pi = np.ones(final_pi.shape) / (final_pi.shape[0] * final_pi.shape[1])
    
    # Calculate js_dist_neighborhood if not provided
    if js_dist_neighborhood is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to calculate js_dist_neighborhood")
        if radius is None:
            raise ValueError("radius must be provided to calculate js_dist_neighborhood")
        
        # Calculate neighborhood distributions
        nd_sliceA = neighborhood_distribution(sliceA, radius=radius) + 1e-6
        nd_sliceB = neighborhood_distribution(sliceB, radius=radius) + 1e-6
        js_dist_neighborhood = jensenshannon_divergence_backend(nd_sliceA, nd_sliceB)
        
        # Convert to numpy if necessary
        if isinstance(js_dist_neighborhood, torch.Tensor):
            js_dist_neighborhood = js_dist_neighborhood.detach().cpu().numpy()
    
    # Calculate cosine_dist_gene_expr if not provided
    if cosine_dist_gene_expr is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to calculate cosine_dist_gene_expr")
        cosine_dist_gene_expr = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep)

    # Calculate cell-type matching metrics if slices are provided
    if cell_type_mismatch is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to calculate cell_type_matching_percentage")
        cell_type_mismatch = calculate_cell_type_mismatch(sliceA, sliceB)

    results = {}
    
    # Calculate neighborhood similarities
    results['initial_obj_neighbor'] = calculate_neighborhood_similarity(js_dist_neighborhood, init_pi)
    results['final_obj_neighbor'] = calculate_neighborhood_similarity(js_dist_neighborhood, final_pi)
    
    # Calculate gene expression similarities
    results['initial_obj_gene'] = calculate_gene_expression_similarity(cosine_dist_gene_expr, init_pi)
    results['final_obj_gene'] = calculate_gene_expression_similarity(cosine_dist_gene_expr, final_pi)
    
    # Calculate cell-type matching percentages
    results['initial_cell_type_match'] = cell_type_matching(cell_type_mismatch, init_pi)
    results['final_cell_type_match'] = cell_type_matching(cell_type_mismatch, final_pi)

    print(f"Initial neighborhood similarity (jsd): {results['initial_obj_neighbor']}")
    print(f"Final neighborhood similarity (jsd): {results['final_obj_neighbor']}")
    print(f"Improvement in neighborhood similarity: {(results['initial_obj_neighbor'] - results['final_obj_neighbor']) * 100:.2f}%")
    print(f"Initial gene expression similarity: {results['initial_obj_gene']}")
    print(f"Final gene expression similarity: {results['final_obj_gene']}")
    print(f"Improvement in gene expression similarity: {(results['initial_obj_gene'] - results['final_obj_gene']) * 100:.2f}%")
    print(f"Initial cell-type matching percentage: {results['initial_cell_type_match']}")
    print(f"Final cell-type matching percentage: {results['final_cell_type_match']}")
    print(f"Improvement in cell-type matching percentage: {(results['final_cell_type_match'] - results['initial_cell_type_match']) * 100:.2f}%")

    return results
