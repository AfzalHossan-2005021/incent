"""Shared fixtures for the INCENT test suite."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest


def _make_slice(
    n: int,
    n_genes: int,
    n_types: int,
    coord_scale: float,
    cell_types: list[str],
    seed: int,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    expression = rng.poisson(2.0, size=(n, n_genes)).astype(np.float32) + 0.1
    coords = rng.normal(size=(n, 2)).astype(np.float32) * coord_scale
    labels = rng.choice(cell_types[:n_types], size=n)
    adata = ad.AnnData(X=expression)
    adata.obsm["spatial"] = coords
    adata.obs["cell_type_annot"] = labels
    return adata


@pytest.fixture(scope="session")
def small_pair() -> tuple[ad.AnnData, ad.AnnData]:
    """A 30-vs-35-cell synthetic pair used for fast smoke tests."""
    cell_types = ["T1", "T2", "T3", "T4"]
    sliceA = _make_slice(30, 50, 3, coord_scale=10.0, cell_types=cell_types, seed=0)
    sliceB = _make_slice(35, 50, 3, coord_scale=10.0, cell_types=cell_types, seed=1)
    return sliceA, sliceB


@pytest.fixture(scope="session")
def tiny_pair() -> tuple[ad.AnnData, ad.AnnData]:
    """An 8-vs-8 pair for unit-style assertions on intermediate matrices."""
    cell_types = ["A", "B"]
    sliceA = _make_slice(8, 12, 2, coord_scale=5.0, cell_types=cell_types, seed=10)
    sliceB = _make_slice(8, 12, 2, coord_scale=5.0, cell_types=cell_types, seed=11)
    return sliceA, sliceB
