"""Smoke tests for the benchmark harness.

These tests use the existing :func:`tests.conftest.small_pair` fixture and
do not download anything; they verify that the metric and orchestration code
in ``benchmarks/`` keeps working as the public API of INCENT evolves.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks import methods as methods_mod
from benchmarks import metrics as metrics_mod
from benchmarks.run_dlpfc import _run_one


def test_metrics_recover_perfect_alignment() -> None:
    """A doubly-stochastic identity-like plan on identical labels must score 1.0."""
    labels = np.array(["A", "A", "B", "B", "C"])
    n = labels.size
    pi = np.eye(n) / n
    assert metrics_mod.layer_transfer_accuracy(pi, labels, labels) == pytest.approx(1.0)
    assert metrics_mod.mass_on_same_layer(pi, labels, labels) == pytest.approx(1.0)
    assert metrics_mod.label_transfer_ari(pi, labels, labels) == pytest.approx(1.0)


def test_metrics_handle_label_mismatch() -> None:
    labels_A = np.array(["A", "A", "B", "B"])
    labels_B = np.array(["B", "B", "A", "A"])
    pi = np.eye(4) / 4
    # Argmax partner for spot i is i, but labels are flipped → 0% accuracy.
    assert metrics_mod.layer_transfer_accuracy(pi, labels_A, labels_B) == pytest.approx(0.0)
    assert metrics_mod.mass_on_same_layer(pi, labels_A, labels_B) == pytest.approx(0.0)


def test_run_one_incent_smoke(small_pair) -> None:
    """The orchestration wrapper must produce a complete result row for INCENT
    on a small synthetic pair, without hitting the network."""
    A, B = small_pair
    row = _run_one("incent", A, B)
    assert row["method"] == "incent"
    assert row["error"] == ""
    for key in ("runtime_s", "peak_mem_mb", "pi_sum", "layer_acc",
                "mass_same_layer", "label_ari"):
        assert key in row, f"missing key {key} in {row}"
    assert 0.0 <= float(row["layer_acc"]) <= 1.0
    assert float(row["pi_sum"]) > 0.0


def test_methods_registry_only_advertises_callables() -> None:
    for name, fn in methods_mod.METHODS.items():
        assert callable(fn), f"{name} is not callable"
