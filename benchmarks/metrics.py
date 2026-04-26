"""Standard DLPFC alignment metrics.

All metrics take an ``(n_A, n_B)`` plan ``pi`` and the cell-type label arrays
of the two slices. We deliberately reproduce the metrics used in the PASTE
and PASTE2 papers so numbers can be cross-checked against published tables.
"""

from __future__ import annotations

import numpy as np


def layer_transfer_accuracy(pi: np.ndarray, labels_A: np.ndarray,
                            labels_B: np.ndarray) -> float:
    """For each spot in slice A, transfer the layer of its argmax partner in B
    and check it against A's true layer.

    Equivalent to "1-NN label transfer using ``pi`` as the similarity matrix".
    """
    partner = pi.argmax(axis=1)
    return float((labels_B[partner] == labels_A).mean())


def mass_on_same_layer(pi: np.ndarray, labels_A: np.ndarray,
                       labels_B: np.ndarray) -> float:
    """Total transported mass between spot pairs that share a layer label.

    Reported in the PASTE / PASTE2 papers as their primary metric. Lies in
    [0, 1] for a normalized plan.
    """
    same_layer = (labels_A[:, None] == labels_B[None, :]).astype(np.float64)
    total = float(pi.sum())
    if total <= 0:
        return float("nan")
    return float((pi * same_layer).sum() / total)


def label_transfer_ari(pi: np.ndarray, labels_A: np.ndarray,
                       labels_B: np.ndarray) -> float:
    """Adjusted Rand Index between transferred and true layer labels in A."""
    from sklearn.metrics import adjusted_rand_score
    partner = pi.argmax(axis=1)
    return float(adjusted_rand_score(labels_A, labels_B[partner]))


METRICS = {
    "layer_acc": layer_transfer_accuracy,
    "mass_same_layer": mass_on_same_layer,
    "label_ari": label_transfer_ari,
}
