"""Unit tests for the individual cost-matrix building blocks of INCENT."""

from __future__ import annotations

import numpy as np
import ot
import pytest

from incent.core import (
    calculate_cell_type_mismatch,
    calculate_gene_expression_cosine_distance,
    calculate_neighborhood_dissimilarity,
    calculate_spatial_distance,
    estimate_characteristic_spacing,
)


def test_cell_type_mismatch_matches_pairwise_label_indicator(tiny_pair) -> None:
    A, B = tiny_pair
    M = calculate_cell_type_mismatch(A, B)
    assert M.shape == (A.n_obs, B.n_obs)
    expected = (A.obs["cell_type_annot"].to_numpy()[:, None]
                != B.obs["cell_type_annot"].to_numpy()[None, :]).astype(np.float64)
    np.testing.assert_array_equal(M, expected)
    assert set(np.unique(M).tolist()) <= {0.0, 1.0}


def test_gene_expression_cosine_distance_bounds(tiny_pair) -> None:
    A, B = tiny_pair
    D = calculate_gene_expression_cosine_distance(A, B, use_rep=None)
    assert D.shape == (A.n_obs, B.n_obs)
    assert np.all(D >= -1e-6)
    assert np.all(D <= 2.0 + 1e-6)


def test_spatial_distance_is_normalized_and_symmetric(tiny_pair) -> None:
    A, B = tiny_pair
    nx = ot.backend.NumpyBackend()
    D_A, D_B = calculate_spatial_distance(A, B, nx, data_type=np.float64)
    np.testing.assert_allclose(D_A, D_A.T, atol=1e-9)
    np.testing.assert_allclose(D_B, D_B.T, atol=1e-9)
    np.testing.assert_allclose(np.diag(D_A), 0.0, atol=1e-9)
    # Per-cell scale normalization: max should be O(1) tens, not e.g. e+6.
    assert D_A.max() < 1e3
    assert D_B.max() < 1e3


def test_estimate_characteristic_spacing_positive(tiny_pair) -> None:
    A, _ = tiny_pair
    s = estimate_characteristic_spacing(A, k=3)
    assert s > 0


def test_neighborhood_dissimilarity_is_finite_and_nonneg(tiny_pair) -> None:
    A, B = tiny_pair
    nx = ot.backend.NumpyBackend()
    D = calculate_neighborhood_dissimilarity(A, B, radius=10.0, nx=nx)
    D_np = nx.to_numpy(D) if hasattr(nx, "to_numpy") else np.asarray(D)
    assert D_np.shape == (A.n_obs, B.n_obs)
    assert np.all(np.isfinite(D_np))
    assert np.all(D_np >= -1e-6)


@pytest.mark.parametrize("radius", [None, 8.0])
def test_neighborhood_dissimilarity_handles_default_and_explicit_radius(tiny_pair, radius) -> None:
    A, B = tiny_pair
    nx = ot.backend.NumpyBackend()
    D = calculate_neighborhood_dissimilarity(A, B, radius=radius, nx=nx)
    D_np = nx.to_numpy(D) if hasattr(nx, "to_numpy") else np.asarray(D)
    assert D_np.shape == (A.n_obs, B.n_obs)
    assert np.all(np.isfinite(D_np))
