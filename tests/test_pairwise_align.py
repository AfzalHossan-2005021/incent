"""End-to-end smoke tests for the public alignment entry points."""

from __future__ import annotations

import numpy as np
import pytest

import incent


@pytest.mark.parametrize("unbalanced", [False, True])
def test_pairwise_align_returns_valid_plan(small_pair, unbalanced: bool) -> None:
    A, B = small_pair
    pi = incent.pairwise_align(
        A, B,
        alpha=0.5, beta=0.3, gamma=0.5,
        use_gpu=False, gpu_verbose=False,
        unbalanced=unbalanced,
        verbose=False,
    )
    assert pi.shape == (A.n_obs, B.n_obs)
    assert np.all(np.isfinite(pi))
    assert np.all(pi >= -1e-9)
    if not unbalanced:
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-6)


def test_pairwise_align_self_alignment_concentrates_mass(small_pair) -> None:
    """A slice aligned against itself should put most mass on the diagonal."""
    A, _ = small_pair
    pi = incent.pairwise_align(
        A, A,
        alpha=0.5, beta=0.5, gamma=0.5,
        use_gpu=False, gpu_verbose=False,
        unbalanced=False,
        verbose=False,
    )
    n = A.n_obs
    diag_mass = float(np.trace(pi))
    total_mass = float(pi.sum())
    assert diag_mass / total_mass > 0.5, (
        f"Self-alignment should concentrate mass on the diagonal; "
        f"got diag/total={diag_mass / total_mass:.3f}"
    )
    # The argmax of every row should be the row itself for the majority of cells.
    self_match = (pi.argmax(axis=1) == np.arange(n)).mean()
    assert self_match > 0.5


def test_calculate_performance_metrics_keys(small_pair) -> None:
    A, B = small_pair
    pi = incent.pairwise_align(
        A, B, alpha=0.5, beta=0.3, gamma=0.5,
        use_gpu=False, gpu_verbose=False, unbalanced=False,
    )
    metrics = incent.calculate_performance_metrics(pi, sliceA=A, sliceB=B, use_gpu=False)
    expected = {
        "initial_obj_neighbor", "final_obj_neighbor",
        "initial_obj_gene", "final_obj_gene",
        "initial_cell_type_match", "final_cell_type_match",
    }
    assert expected <= set(metrics.keys())
    # FGW should reduce neighborhood + gene-expression cost on synthetic data.
    assert metrics["final_obj_neighbor"] <= metrics["initial_obj_neighbor"] + 1e-6
    assert metrics["final_obj_gene"] <= metrics["initial_obj_gene"] + 1e-6


def test_calculate_forward_reverse_compactness_keys(small_pair) -> None:
    A, B = small_pair
    pi = incent.pairwise_align(
        A, B, alpha=0.5, beta=0.3, gamma=0.5,
        use_gpu=False, gpu_verbose=False, unbalanced=False,
    )
    out = incent.calculate_forward_reverse_compactness(pi, A, B)
    assert {"forward_compactness", "reverse_compactness",
            "effective_support_fwd", "effective_support_rev"} <= set(out.keys())
    for v in out.values():
        assert np.isfinite(float(v))
        assert float(v) >= 0.0


def test_hierarchical_alignment_smoke(small_pair) -> None:
    A, B = small_pair
    pi = incent.hierarchical_pairwise_align(
        A, B,
        alpha=0.5, beta=0.3, gamma=0.5,
        target_cluster_size=10,
        use_gpu=False, verbose=False,
    )
    assert pi.shape == (A.n_obs, B.n_obs)
    assert np.all(np.isfinite(pi))
    assert np.all(pi >= -1e-9)
