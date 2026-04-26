"""Method wrappers for the DLPFC benchmark.

Each wrapper has signature ``(sliceA, sliceB, **kwargs) -> np.ndarray`` and
returns the (n_A, n_B) soft transport plan ``pi``.

We deliberately ship hyperparameters as defaults that match each method's
published recommendation for DLPFC; if you want to sweep, override per call.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import anndata as ad
import numpy as np

# Methods that depend on heavy optional packages are imported lazily inside
# their wrappers so a missing baseline does not block the rest of the harness.

MethodFn = Callable[[ad.AnnData, ad.AnnData], np.ndarray]


def run_incent(A: ad.AnnData, B: ad.AnnData,
               alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.5,
               unbalanced: bool = False, hierarchical: bool = True,
               use_gpu: bool = False) -> np.ndarray:
    import incent
    if hierarchical and min(A.n_obs, B.n_obs) > 500:
        return incent.hierarchical_pairwise_align(
            A, B, alpha=alpha, beta=beta, gamma=gamma,
            target_cluster_size=200, label_key="cell_type_annot",
            use_gpu=use_gpu, verbose=False,
        )
    return incent.pairwise_align(
        A, B, alpha=alpha, beta=beta, gamma=gamma,
        unbalanced=unbalanced, use_gpu=use_gpu,
        gpu_verbose=False, verbose=False,
    )


def run_paste(A: ad.AnnData, B: ad.AnnData, alpha: float = 0.1) -> np.ndarray:
    import paste
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi = paste.pairwise_align(A, B, alpha=alpha, dissimilarity="kl",
                                  norm=True, verbose=False)
    return np.asarray(pi)


def run_paste2(A: ad.AnnData, B: ad.AnnData,
               alpha: float = 0.1, s: float = 0.99) -> np.ndarray:
    """PASTE2 partial alignment.

    ``s`` is the overlap fraction; ``s=0.99`` matches the full-overlap setup
    used in the PASTE2 paper for the DLPFC benchmark.
    """
    from paste2.PASTE2 import partial_pairwise_align
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi = partial_pairwise_align(A, B, s=s, alpha=alpha,
                                    dissimilarity="kl",
                                    verbose=False, norm=True)
    return np.asarray(pi)


def run_moscot(A: ad.AnnData, B: ad.AnnData,
               alpha: float = 0.5, epsilon: float = 0.01) -> np.ndarray:
    from moscot.problems.space import AlignmentProblem
    # Moscot needs both slices in a single AnnData with a batch column.
    AB = ad.concat([A, B], label="batch", keys=["A", "B"], join="outer",
                   merge="same", index_unique=None)
    AB.obs["batch"] = AB.obs["batch"].astype(str)
    AB.obsm["spatial"] = np.vstack([A.obsm["spatial"], B.obsm["spatial"]])
    problem = AlignmentProblem(AB)
    problem = problem.prepare(batch_key="batch", spatial_key="spatial",
                              policy="sequential")
    problem = problem.solve(alpha=alpha, epsilon=epsilon, jit=True)
    # Extract the (n_A, n_B) plan from the only sub-problem.
    (key,) = problem.problems.keys()
    sol = problem.solutions[key]
    pi = np.asarray(sol.transport_matrix)
    return pi


METHODS: dict[str, MethodFn] = {
    "incent": run_incent,
    "paste": run_paste,
    "paste2": run_paste2,
    "moscot": run_moscot,
}
