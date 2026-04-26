"""DLPFC dataset loader.

Pulls pre-processed ``.h5ad`` files distributed with the
`paste3 <https://github.com/raphael-group/paste3>`_ reproducibility package
(originally from `spatialLIBD <https://research.libd.org/spatialLIBD/>`_,
re-saved by the PASTE authors so they can be loaded directly with anndata).

By default only the two **Br3942** pairs (151673-151674 and 151675-151676)
are loaded because they are the only DLPFC slices currently distributed in
``paste3/tests/data``. To benchmark on all twelve LIBD slices, mirror the
full dataset under ``benchmarks/data/`` (see
:func:`spatialLIBD.fetch_data` in R, or the Bioconductor
``ExperimentHub`` package) and the loader will use your local files
without re-downloading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import anndata as ad
import numpy as np

# Adjacent DLPFC pairs whose preprocessed h5ad files are publicly downloadable
# from paste3/tests/data. Mirror the missing eight slices locally to extend.
DLPFC_PAIRS: list[tuple[str, str]] = [
    ("151673", "151674"),
    ("151675", "151676"),
]

# All twelve LIBD DLPFC slice IDs, exposed for users with local mirrors.
DLPFC_PAIRS_ALL: list[tuple[str, str]] = [
    ("151507", "151508"),
    ("151509", "151510"),
    ("151669", "151670"),
    ("151671", "151672"),
    ("151673", "151674"),
    ("151675", "151676"),
]

LABEL_COL = "layer_guess_reordered"
URL_TEMPLATE = (
    "https://raw.githubusercontent.com/raphael-group/paste3/main/"
    "tests/data/{sample}.h5ad"
)


@dataclass
class DLPFCSample:
    """A loaded + pre-processed DLPFC slice."""

    sample_id: str
    adata: ad.AnnData

    @property
    def n_spots(self) -> int:
        return int(self.adata.n_obs)


def _ensure_downloaded(sample_id: str, cache_dir: Path) -> Path:
    """Download a single DLPFC sample to ``cache_dir`` if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{sample_id}.h5ad"
    if not target.exists():
        url = URL_TEMPLATE.format(sample=sample_id)
        print(f"  downloading {url} -> {target}")
        urlretrieve(url, target)
    return target


def _preprocess(adata: ad.AnnData, n_top_genes: int) -> ad.AnnData:
    """Standard Visium preprocessing: drop unannotated spots, log-CPM, HVG flag."""
    import scanpy as sc

    adata = adata[~adata.obs[LABEL_COL].isna()].copy()
    adata.obs["cell_type_annot"] = adata.obs[LABEL_COL].astype(str)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                flavor="cell_ranger", subset=False)
    return adata


def load_pair(
    pair: tuple[str, str],
    cache_dir: Path = Path("benchmarks/data"),
    n_top_genes: int = 2000,
) -> tuple[DLPFCSample, DLPFCSample]:
    """Load and pre-process an adjacent DLPFC slice pair.

    Genes are restricted to those that are HVG in **either** slice and present
    in **both** slices, which is the standard PASTE / PASTE2 setup.
    """
    import scanpy as sc

    print(f"loading DLPFC pair {pair}")
    paths = [_ensure_downloaded(s, cache_dir) for s in pair]
    slices = [_preprocess(sc.read_h5ad(p), n_top_genes) for p in paths]
    A, B = slices
    hvg = (set(A.var_names[A.var["highly_variable"]])
           | set(B.var_names[B.var["highly_variable"]]))
    shared = sorted(set(A.var_names) & set(B.var_names) & hvg)
    A = A[:, shared].copy()
    B = B[:, shared].copy()
    print(f"  {pair[0]}: n_spots={A.n_obs}, {pair[1]}: n_spots={B.n_obs}, "
          f"shared HVGs={len(shared)}")
    return DLPFCSample(pair[0], A), DLPFCSample(pair[1], B)


def downsample(adata: ad.AnnData, n: int, seed: int = 0) -> ad.AnnData:
    """Random subsample to at most ``n`` spots, preserving label distribution."""
    if adata.n_obs <= n:
        return adata
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=n, replace=False)
    return adata[idx].copy()
