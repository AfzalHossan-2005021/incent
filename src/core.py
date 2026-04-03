import ot
import torch
import numpy as np

from tqdm import tqdm
from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from .utils import select_backend, fused_gromov_wasserstein_incent, to_dense_array, extract_data_matrix, jensenshannon_divergence_backend, pairwise_msd, to_backend


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
    epsilon: float = 1e-6,
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
    
    # Determine if gpu or cpu is being used
    use_gpu, nx = select_backend(use_gpu=use_gpu, gpu_verbose=gpu_verbose)
    
    
    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.")   
    
    # ────────────────────── Calculate spatial distances ──────────────────────
    D_A, D_B = calculate_spatial_distance(sliceA, sliceB, nx, data_type=data_type, eps=epsilon)
    

    # ────────────────────── Calculate gene expression dissimilarity ──────────────────────
    cosine_dist_gene_expr = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep, eps=epsilon)


    # ────────────────────── Calculate cell-type mismatch penalty ──────────────────────
    cell_type_mismatch = calculate_cell_type_mismatch(sliceA, sliceB)


    # Combine gene expression dissimilarity and cell-type mismatch penalty into a single cost matrix M1
    M1_combined = (1 - beta) * cosine_dist_gene_expr + beta * cell_type_mismatch
    M1 = to_backend(M1_combined, nx, data_type=data_type)


    # ────────────────────── Calculate neighborhood dissimilarity ──────────────────────
    js_dist_neighborhood = calculate_neighborhood_dissimilarity(sliceA, sliceB, radius, nx, data_type=data_type, eps=epsilon)
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
    
    pi, logw = fused_gromov_wasserstein_incent(M1, M2, D_A, D_B, a, b, G_init = G_init, loss_fun='square_loss', alpha= alpha, gamma=gamma, log=True, numItermax=numItermax, verbose=verbose, **kwargs)
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


def neighborhood_distribution(slice, radius, n_shells=2, n_sectors=4, distance_decay='linear', include_self=False, cell_types=None):
    """
    Calculate the neighborhood distribution using a Local Reference Frame (Density Dipole).
    This descriptor is rotation-invariant, preserves relative angular positions (co-localization
    vs repulsion), prevents empty bin edge-case explosions, and is strictly non-negative for JSD.

    Args:
        slice: Slice to get niche distribution for.
        radius: Radius of the niche.
        n_shells: Number of radial bins (distance layers).
        n_sectors: Number of angular bins relative to the local principal axis.
        distance_decay: 'linear', 'gaussian', or None. Weights closer cells more.
        include_self: Whether to include the focal cell in its own neighborhood.
        cell_types: Global array of cell types to ensure consistent feature columns.

    Returns:
        niche_distribution: Features for each cell containing local-frame binned neighbor weights.
    """
    print("Calculating local-reference neighborhood distribution...")

    if cell_types is None:
        unique_cell_types = np.array(sorted(list(slice.obs['cell_type_annot'].unique())))
    else:
        unique_cell_types = np.array(cell_types)

    num_types = len(unique_cell_types)
    cell_type_to_index = dict(zip(unique_cell_types, list(range(num_types))))
    
    num_cells = slice.shape[0]
    num_features = num_types * n_shells * n_sectors + 1  # +1 for empty neighborhood feature
    cells_within_radius = np.zeros((num_cells, num_features), dtype=float)

    source_coords = slice.obsm['spatial']
    distances = euclidean_distances(source_coords, source_coords)
    
    shell_edges = np.linspace(0, radius, n_shells + 1)
    angle_edges = np.linspace(0, 2 * np.pi, n_sectors + 1)

    for i in tqdm(range(num_cells)):
        target_indices = np.where(distances[i] <= radius)[0]
        
        if not include_self:
            target_indices = target_indices[target_indices != i]

        if len(target_indices) == 0:
            cells_within_radius[i, -1] = 1.0  # Assign all mass to the "empty" feature bin
            continue

        target_coords = source_coords[target_indices]
        rel_coords = target_coords - source_coords[i]
        target_dists = distances[i, target_indices]
        
        # 1. Calculate weights
        if distance_decay == "linear":
            weights = np.maximum(0.0, 1.0 - target_dists / radius)
        elif distance_decay == "gaussian":
            sigma = radius / 2.0
            weights = np.exp(-(target_dists ** 2) / (2 * sigma * sigma))
        else:
            weights = np.ones_like(target_dists)

        # 2. Establish Local Principal Axis (Density Dipole)
        angles = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
        dipole_x = np.sum(weights * np.cos(angles))
        dipole_y = np.sum(weights * np.sin(angles))
        principal_angle = np.arctan2(dipole_y, dipole_x)  # "Local North"
        
        # 3. Align angles to Local Reference Frame
        aligned_angles = (angles - principal_angle) % (2 * np.pi)
        
        # 4. Discretize into Shells and Sectors
        rad_bins = np.digitize(target_dists, shell_edges[1:], right=True)
        rad_bins = np.clip(rad_bins, 0, n_shells - 1)
        
        ang_bins = np.digitize(aligned_angles, angle_edges[:-1]) - 1
        ang_bins = np.clip(ang_bins, 0, n_sectors - 1)

        # Build feature histogram for this cell
        local_feat = np.zeros((num_types, n_shells, n_sectors), dtype=float)

        for idx, ind in enumerate(target_indices):
            cell_type_str_j = str(slice.obs['cell_type_annot'].iloc[ind])
            if cell_type_str_j not in cell_type_to_index:
                continue
                
            cell_type_idx = cell_type_to_index[cell_type_str_j]
            ww = weights[idx]
            rb = rad_bins[idx]
            ab = ang_bins[idx]
            
            local_feat[cell_type_idx, rb, ab] += ww

        cells_within_radius[i, :-1] = local_feat.flatten()
        
        # 5. Normalize to proper probability distribution for Jensen-Shannon Divergence
        row_sum = np.sum(cells_within_radius[i, :-1])
        if row_sum > 0:
            cells_within_radius[i, :-1] /= row_sum
        else:
            cells_within_radius[i, -1] = 1.0

    return np.array(cells_within_radius)


def calculate_spatial_distance(sliceA, sliceB, nx, data_type=np.float32, spatial_key = 'spatial', eps=1e-6):
    """
    Calculate spatial distance between cells in a slice.

    Args:
        sliceA: First slice for which to calculate spatial distance.
        sliceB: Second slice for which to calculate spatial distance.
        nx: Backend to use for calculations (e.g., NumpyBackend or TorchBackend).
        data_type: Data type for backend tensors (default is float32).
        spatial_key: Key for the spatial coordinates in the slice's obsm.
        eps: Small constant to add to distance matrices to avoid zero vectors.
    Returns:
    D_A, D_B: Pairwise spatial distance matrices of the slices.
    """
    
    print("Calculating spatial distance between cells in slice A and slice B")

    coordinates_A = sliceA.obsm[spatial_key]
    coordinates_B = sliceB.obsm[spatial_key]

    D_A = euclidean_distances(coordinates_A, coordinates_A)
    D_B = euclidean_distances(coordinates_B, coordinates_B)

    D_A = to_backend(D_A, nx, data_type=data_type)
    D_B = to_backend(D_B, nx, data_type=data_type)

    scale = max(D_A.max(), D_B.max()) + eps
    D_A = D_A / scale
    D_B = D_B / scale

    return D_A, D_B


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
        cell_type_mismatch: Binary matrix indicating cell-type mismatches.
    """

    _lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    _lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)

    cell_type_mismatch = (_lab_A[:, None] != _lab_B[None, :]).astype(np.float64)

    return cell_type_mismatch


def calculate_neighborhood_dissimilarity(sliceA, sliceB, radius, nx, data_type=np.float32, eps=1e-6):
    """
    Calculate neighborhood dissimilarity between two slices based on Jensen-Shannon distance of neighborhood distributions.
    Args:
        sliceA: First slice.
        sliceB: Second slice.
        radius: Radius for neighborhood calculation.
        nx: Backend to use for calculations (e.g., NumpyBackend or TorchBackend).
        data_type: Data type for backend tensors (default is float32).
        eps: Small constant to add to neighborhood distributions to avoid zero vectors.
    Returns:
        js_dist_neighborhood: Jensen-Shannon distance matrix of neighborhood distributions between sliceA and sliceB.
    """
    # 1. Establish a global cell type vocabulary so A and B features align to the exact same columns!
    all_types = np.array(sorted(list(set(sliceA.obs['cell_type_annot'].unique()).union(set(sliceB.obs['cell_type_annot'].unique())))))

    neighborhood_distribution_sliceA = neighborhood_distribution(sliceA, radius=radius, cell_types=all_types) + eps
    neighborhood_distribution_sliceB = neighborhood_distribution(sliceB, radius=radius, cell_types=all_types) + eps

    neighborhood_distribution_sliceA = to_backend(neighborhood_distribution_sliceA, nx, data_type=data_type)
    neighborhood_distribution_sliceB = to_backend(neighborhood_distribution_sliceB, nx, data_type=data_type)

    js_dist_neighborhood = jensenshannon_divergence_backend(neighborhood_distribution_sliceA, neighborhood_distribution_sliceB)

    return js_dist_neighborhood

