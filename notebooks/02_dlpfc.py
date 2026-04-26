# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # INCENT on adjacent DLPFC Visium slices
#
# This notebook aligns two adjacent slices from the human dorsolateral
# prefrontal cortex (DLPFC) Visium dataset of [Maynard et al.,
# 2021](https://doi.org/10.1038/s41593-020-00787-0). The data is fetched from
# the [`spatialLIBD`](http://research.libd.org/spatialLIBD/) project; each
# sample is ~3,600 spots × 33,000 genes, so the two-slice alignment fits
# comfortably on a single 16 GB GPU (or in ~12 minutes on CPU).
#
# Pairs commonly used in benchmarks:
#
# | Subject | Adjacent pair |
# | ------- | ------------- |
# | Br5292  | 151507 ↔ 151508 |
# | Br5292  | 151509 ↔ 151510 |
# | Br8100  | 151669 ↔ 151670 |
# | Br8100  | 151671 ↔ 151672 |
# | Br3942  | 151673 ↔ 151674 |
# | Br3942  | 151675 ↔ 151676 |
#
# We use **151673 ↔ 151674** by default (Br3942 layer 3-4 boundary, well
# studied in PASTE / PASTE2 papers).

# %% [markdown]
# ## 1. Data setup
#
# We pull the pre-processed `.h5ad` files distributed with the
# [`paste3`](https://github.com/raphael-group/paste3) package (the original
# data is from `spatialLIBD` / Maynard et al., re-saved by the PASTE authors
# for direct loading with `anndata`). Each file is ~47 MB.
#
# > **Network required.** Skip this cell if you have already mirrored the
# > files locally; just point `path_A` / `path_B` at your copies.

# %%
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path("data/dlpfc")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = ["151673", "151674"]
URL_TEMPLATE = (
    "https://raw.githubusercontent.com/raphael-group/paste3/main/"
    "tests/data/{sample}.h5ad"
)

for sample in SAMPLES:
    target = DATA_DIR / f"{sample}.h5ad"
    if target.exists():
        print(f"already cached: {target}")
        continue
    url = URL_TEMPLATE.format(sample=sample)
    print(f"downloading {url} -> {target}")
    urlretrieve(url, target)

path_A = DATA_DIR / f"{SAMPLES[0]}.h5ad"
path_B = DATA_DIR / f"{SAMPLES[1]}.h5ad"

# %% [markdown]
# ## 2. Load and pre-process
#
# We:
#
# 1. Read the two `.h5ad` files.
# 2. Use the existing layer annotation `layer_guess_reordered` (the manual
#    annotation distributed with `spatialLIBD`) as the cell-type column.
#    INCENT expects this column at `obs["cell_type_annot"]`.
# 3. Normalize total counts per spot (CPM) and log1p-transform — standard
#    Visium pre-processing.
# 4. Subset to the top 2000 highly variable genes shared between slices.

# %%
import anndata as ad
import numpy as np
import scanpy as sc

A = sc.read_h5ad(path_A)
B = sc.read_h5ad(path_B)

LABEL_COL = "layer_guess_reordered"

for slc in (A, B):
    if LABEL_COL not in slc.obs:
        raise KeyError(
            f"expected obs[{LABEL_COL!r}]; available columns: {list(slc.obs.columns)}"
        )
    # Drop spots that lack a manual layer annotation.
    slc._inplace_subset_obs(~slc.obs[LABEL_COL].isna())
    slc.obs["cell_type_annot"] = slc.obs[LABEL_COL].astype(str)

for slc in (A, B):
    sc.pp.normalize_total(slc, target_sum=1e4)
    sc.pp.log1p(slc)
    # ``cell_ranger`` flavor is dependency-free; ``seurat_v3`` would require
    # the optional ``scikit-misc`` package.
    sc.pp.highly_variable_genes(slc, n_top_genes=2000, flavor="cell_ranger", subset=False)

# Keep genes that are HVG in either slice and present in both slices.
hvg = (set(A.var_names[A.var["highly_variable"]])
       | set(B.var_names[B.var["highly_variable"]]))
shared = sorted(set(A.var_names) & set(B.var_names) & hvg)
A = A[:, shared].copy()
B = B[:, shared].copy()

print(A, B, sep="\n\n")

# %% [markdown]
# ## 3. Run hierarchical INCENT
#
# DLPFC slices have ~3,500 spots each, which is large enough that the
# coarse-to-fine schedule is much faster than the flat solver and produces
# better large-scale geometry.

# %%
import incent

pi = incent.hierarchical_pairwise_align(
    A, B,
    alpha=0.5, beta=0.3, gamma=0.5,
    target_cluster_size=200,
    label_key="cell_type_annot",
    use_gpu=False,        # set to True if a CUDA GPU is available
    verbose=True,
)
print(f"pi.shape = {pi.shape}, pi.sum() = {pi.sum():.4f}")

# %% [markdown]
# ## 4. Evaluate
#
# Two standard DLPFC metrics:
#
# - **Layer-transfer accuracy.** For each spot in slice A, transfer the
#   layer label of the `argmax` partner in slice B. Compare against the
#   manual annotation of that spot in A.
# - **Mapping accuracy (PASTE definition).** Total transported mass between
#   spots that share a layer label.

# %%
A_labels = A.obs["cell_type_annot"].to_numpy()
B_labels = B.obs["cell_type_annot"].to_numpy()

partner = pi.argmax(axis=1)
layer_acc = float((B_labels[partner] == A_labels).mean())

same_layer = (A_labels[:, None] == B_labels[None, :]).astype(np.float64)
mapping_acc = float((pi * same_layer).sum() / pi.sum())

print(f"layer-transfer accuracy:           {layer_acc:.2%}")
print(f"mass-on-same-layer (PASTE metric): {mapping_acc:.2%}")

# %% [markdown]
# ## 5. Visualize
#
# Stack the two slices using a Procrustes alignment driven by `pi` and color
# by layer. Adjacent DLPFC slices are physically ~10 µm apart and their
# layer boundaries should overlap almost exactly when correctly aligned.

# %%
import matplotlib.pyplot as plt

aligned = incent.visualize.stack_slices_pairwise([A, B], [pi])
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, slc, title in [
    (axes[0], A, f"Source ({SAMPLES[0]})"),
    (axes[1], B, f"Target ({SAMPLES[1]})"),
    (axes[2], aligned[1], f"Target after INCENT alignment\nlayer-acc = {layer_acc:.2%}"),
]:
    coords = slc.obsm["spatial"]
    labels = slc.obs["cell_type_annot"].astype("category")
    cats = labels.cat.categories
    palette = plt.get_cmap("tab10", len(cats))
    for i, ct in enumerate(cats):
        mask = (labels == ct).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.85,
                   color=palette(i), label=ct)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(loc="upper right", fontsize=8, markerscale=2)
fig.tight_layout()

# %% [markdown]
# ## What to try next
#
# - Repeat with `unbalanced=True` and add some "missing" spots in one slice;
#   INCENT should leave them unmatched.
# - Compare against `paste.pairwise_align` and `moscot.problems.space.AlignmentProblem`
#   on the same pair (see `benchmarks/run_dlpfc.py`).
# - Sweep `alpha` ∈ {0.1, 0.3, 0.5, 0.7, 0.9} and report layer accuracy.
# - Run all 12 DLPFC samples by changing `SAMPLES` to all six adjacent pairs
#   and aggregate the layer-accuracy table.
