"""
Reproduce the INCENT quickstart figure used in README.md.

Generates two synthetic 200-cell slices that share a known rotation +
local perturbation, runs ``incent.pairwise_align``, computes recovery
metrics, and writes ``docs/_static/quickstart_alignment.png``.

Run with:

    python scripts/_make_readme_figure.py
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

import incent


def _build_pair(seed: int = 0) -> tuple[ad.AnnData, ad.AnnData, np.ndarray]:
    """Build two slices that share a ground-truth correspondence.

    The target slice is a rotated, jittered copy of the source slice; gene
    expression and cell-type labels are preserved (so pairwise_align should
    recover the identity correspondence after un-rotating).
    """
    rng = np.random.default_rng(seed)
    n_cells = 200
    n_genes = 60
    cell_types = np.array(["Excitatory", "Inhibitory", "Astrocyte", "Microglia"])

    # Source layout: four spatial blobs, one per cell type, jittered.
    type_centers = np.array([[-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [3.0, 3.0]])
    type_assign = rng.integers(0, 4, size=n_cells)
    coords_A = type_centers[type_assign] + rng.normal(scale=0.6, size=(n_cells, 2))
    labels_A = cell_types[type_assign]

    # Cell-type-specific expression mean.
    type_means = rng.normal(scale=2.0, size=(4, n_genes))
    XA = type_means[type_assign] + rng.normal(scale=0.5, size=(n_cells, n_genes))

    # Target slice: rotate by 35° + small per-cell jitter; identical labels and expression.
    theta = np.deg2rad(35.0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    coords_B = coords_A @ R.T + rng.normal(scale=0.15, size=(n_cells, 2))
    labels_B = labels_A.copy()
    XB = XA + rng.normal(scale=0.2, size=(n_cells, n_genes))

    A = ad.AnnData(X=XA.astype(np.float32))
    A.obsm["spatial"] = coords_A.astype(np.float32)
    A.obs["cell_type_annot"] = labels_A
    B = ad.AnnData(X=XB.astype(np.float32))
    B.obsm["spatial"] = coords_B.astype(np.float32)
    B.obs["cell_type_annot"] = labels_B

    truth = np.arange(n_cells)  # cell i in A corresponds to cell i in B
    return A, B, truth


def main(out_path: Path = Path("docs/_static/quickstart_alignment.png")) -> None:
    A, B, truth = _build_pair(seed=0)

    pi = incent.pairwise_align(
        A, B,
        alpha=0.5, beta=0.3, gamma=0.5,
        use_gpu=False, gpu_verbose=False,
        unbalanced=False, verbose=False,
    )

    # Top-1 recovery: argmax of each row in pi should equal the truth.
    pred = pi.argmax(axis=1)
    top1 = float((pred == truth).mean())
    label_acc = float((B.obs["cell_type_annot"].to_numpy()[pred]
                       == A.obs["cell_type_annot"].to_numpy()).mean())

    print(f"top-1 cell-correspondence recovery: {top1:.3f}")
    print(f"cell-type label-transfer accuracy:  {label_acc:.3f}")

    # Run the metric reporter for diagnostics.
    incent.calculate_performance_metrics(pi, sliceA=A, sliceB=B, use_gpu=False)

    # Build a 1x3 figure: source, target, target after rigid alignment driven by pi.
    aligned = incent.visualize.stack_slices_pairwise([A, B], [pi])
    coords_A_aligned = aligned[0].obsm["spatial"]
    coords_B_aligned = aligned[1].obsm["spatial"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    type_to_color = {"Excitatory": "#1f77b4", "Inhibitory": "#ff7f0e",
                     "Astrocyte": "#2ca02c", "Microglia": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, slice_, title in [
        (axes[0], A, "Source slice (A)"),
        (axes[1], B, "Target slice (B), rotated 35°"),
    ]:
        for ct, color in type_to_color.items():
            mask = slice_.obs["cell_type_annot"].to_numpy() == ct
            ax.scatter(slice_.obsm["spatial"][mask, 0],
                       slice_.obsm["spatial"][mask, 1],
                       s=18, alpha=0.85, c=color, label=ct)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[2]
    ax.scatter(coords_A_aligned[:, 0], coords_A_aligned[:, 1], s=18, alpha=0.5,
               c="#e41a1c", label="A (aligned)")
    ax.scatter(coords_B_aligned[:, 0], coords_B_aligned[:, 1], s=18, alpha=0.5,
               c="#377eb8", label="B")
    ax.set_title(f"After INCENT + Procrustes\ntop-1={top1:.2%}, label-acc={label_acc:.2%}",
                 fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=9)

    fig.suptitle("INCENT quickstart: synthetic 200-cell rotation recovery", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
