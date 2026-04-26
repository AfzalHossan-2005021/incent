"""Run the DLPFC benchmark.

By default this runs all four methods (``incent``, ``paste``, ``paste2``,
``moscot``) on every adjacent pair in :data:`benchmarks.data.DLPFC_PAIRS`,
records layer-transfer accuracy / mass-on-same-layer / runtime / peak memory,
and writes both a CSV (``benchmarks/results/dlpfc.csv``) and a summary plot
(``benchmarks/results/dlpfc.png``).

Quickstart on a single pair (~minutes on CPU)::

    python -m benchmarks.run_dlpfc --pairs 151673,151674 --methods incent,paste

Full benchmark (~hours on CPU; minutes on GPU)::

    python -m benchmarks.run_dlpfc

If a baseline package is missing it is silently skipped. Failures from a
single (pair, method) cell are recorded as ``error`` in the CSV but do not
abort the rest of the run.
"""

from __future__ import annotations

import argparse
import gc
import time
import tracemalloc
from pathlib import Path

from . import data as data_mod
from . import methods as methods_mod
from . import metrics as metrics_mod


def _run_one(method_name: str, A, B,
             timeout_s: float | None = None) -> dict[str, object]:
    """Run a single method on a pair; return a flat dict of metrics + meta."""
    fn = methods_mod.METHODS[method_name]
    out: dict[str, object] = {"method": method_name, "error": ""}

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        pi = fn(A, B)
    except Exception as exc:  # noqa: BLE001 — keep the harness running
        elapsed = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        out.update(error=f"{type(exc).__name__}: {exc}",
                   runtime_s=elapsed, peak_mem_mb=peak / 1e6)
        return out
    elapsed = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    labels_A = A.obs["cell_type_annot"].to_numpy()
    labels_B = B.obs["cell_type_annot"].to_numpy()

    out["runtime_s"] = elapsed
    out["peak_mem_mb"] = peak / 1e6
    out["pi_sum"] = float(pi.sum())
    for name, fn_metric in metrics_mod.METRICS.items():
        out[name] = fn_metric(pi, labels_A, labels_B)
    return out


def _parse_pair(s: str) -> tuple[str, str]:
    a, b = s.split(",")
    return a.strip(), b.strip()


def main(pairs: list[tuple[str, str]] | None = None,
         method_names: list[str] | None = None,
         max_spots: int | None = None,
         out_dir: Path = Path("benchmarks/results")):
    import pandas as pd

    pairs = pairs or list(data_mod.DLPFC_PAIRS)
    method_names = method_names or list(methods_mod.METHODS)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for pair in pairs:
        sample_A, sample_B = data_mod.load_pair(pair)
        A, B = sample_A.adata, sample_B.adata
        if max_spots is not None:
            A = data_mod.downsample(A, max_spots, seed=0)
            B = data_mod.downsample(B, max_spots, seed=1)
        for method_name in method_names:
            print(f"  running {method_name} on {pair} (n_A={A.n_obs}, n_B={B.n_obs})")
            row = _run_one(method_name, A, B)
            row.update(pair=f"{pair[0]}-{pair[1]}",
                       sample_A=pair[0], sample_B=pair[1],
                       n_A=A.n_obs, n_B=B.n_obs)
            print(f"    -> layer_acc={row.get('layer_acc')}, "
                  f"mass_same_layer={row.get('mass_same_layer')}, "
                  f"runtime_s={row.get('runtime_s'):.1f}"
                  f"{' (' + str(row['error']) + ')' if row['error'] else ''}")
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "dlpfc.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    try:
        _plot(df, out_dir / "dlpfc.png")
    except Exception as exc:  # noqa: BLE001
        print(f"plotting failed (non-fatal): {exc}")
    return df


def _plot(df, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    df = df[df["error"] == ""].copy()
    if df.empty:
        print("no successful runs to plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pivot_acc = df.pivot_table(index="pair", columns="method",
                               values="layer_acc", aggfunc="mean")
    pivot_t = df.pivot_table(index="pair", columns="method",
                             values="runtime_s", aggfunc="mean")
    pivot_acc.plot(kind="bar", ax=axes[0])
    axes[0].set_ylabel("layer-transfer accuracy")
    axes[0].set_title("DLPFC layer-transfer accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="lower right", fontsize=9)

    pivot_t.plot(kind="bar", ax=axes[1], logy=True)
    axes[1].set_ylabel("runtime (s, log scale)")
    axes[1].set_title("DLPFC runtime")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=str, default=None,
                   help="Semicolon-separated list of <A>,<B> pairs "
                        "(default: all six DLPFC pairs).")
    p.add_argument("--methods", type=str, default=None,
                   help=f"Comma-separated subset of {{{','.join(methods_mod.METHODS)}}}.")
    p.add_argument("--max-spots", type=int, default=None,
                   help="Optional per-slice spot cap for fast smoke tests.")
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    pair_list = ([_parse_pair(s) for s in args.pairs.split(";")]
                 if args.pairs else None)
    method_list = (args.methods.split(",") if args.methods else None)

    main(pair_list, method_list, args.max_spots, args.out_dir)


if __name__ == "__main__":
    _cli()
