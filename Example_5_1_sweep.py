"""Example_5_1_sweep.py

§5.1 ambiguity sweep: widen the discount band [0, beta_hi] and record how the
conservative value and stopping behaviour respond. Reuses Example_5_1_discount.run.

For each beta_hi we run several replicates with paired seeds and aggregate
mean +/- std into summary.csv + sweep.png.
"""

import matplotlib
matplotlib.use("Agg")

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from Example_5_1_discount import run

OUT = "Example_5_1_sweep/"
BETA_HI_GRID = [0.0, 0.05, 0.10, 0.20, 0.40]
REPLICATES = 3


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for gi, beta_hi in enumerate(BETA_HI_GRID):
        for r in range(REPLICATES):
            sub = f"{OUT}run_{gi:02d}b{beta_hi:.2f}_r{r}/"
            s = run(beta_lo=0.0, beta_hi=beta_hi, out_dir=sub, seed=r,
                    diagnose=True)
            rows.append({"beta_hi": beta_hi, "rep": r, "Y0": s["Y0"],
                         "frac_asset": s["frac_asset_interior"],
                         "mean_tau": s["mean_tau_star"]})

    with open(OUT + "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _plot(rows)
    print(f"[5.1 sweep] wrote {OUT}summary.csv and sweep.png")


def _agg_paired(rows, key):
    """Per-seed differences from the beta_hi = 0 baseline, then aggregate.

    The seed effect is a common offset on the level of Y0, and across the band
    it is larger than the trend itself, so unpaired error bars hide the effect.
    Pairing within each seed cancels that offset; the error bars below therefore
    measure variation of the *effect*, not of the level.
    """
    grid = sorted({r["beta_hi"] for r in rows})
    base = {r["rep"]: r[key] for r in rows if r["beta_hi"] == grid[0]}
    curves = {}
    for rep in sorted({r["rep"] for r in rows}):
        if base.get(rep) is None:
            continue
        vals = []
        for b in grid:
            v = next((r[key] for r in rows
                      if r["beta_hi"] == b and r["rep"] == rep), None)
            vals.append(np.nan if v is None else v - base[rep])
        curves[rep] = np.array(vals, dtype=float)
    stack = np.array(list(curves.values()))
    return np.array(grid), np.nanmean(stack, 0), np.nanstd(stack, 0), curves


def _plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = (
        (axes[0], "Y0", "C0", r"$Y_0(\bar\beta) - Y_0(0)$",
         "Conservative value vs discount-band width"),
        (axes[1], "mean_tau", "C1",
         r"mean $\tau^*(\bar\beta) - $ mean $\tau^*(0)$",
         "Stopping vs discount-band width"),
    )
    for ax, key, col, ylab, title in panels:
        g, m, sd, curves = _agg_paired(rows, key)
        for c in curves.values():
            ax.plot(g, c, color="0.75", lw=0.8, marker=".", ms=4, zorder=1)
        ax.errorbar(g, m, yerr=sd, marker="o", capsize=3, lw=1.6, color=col,
                    zorder=2, label=r"mean $\pm$ s.d. (paired)")
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel(r"band width $\bar\beta$ (band $[0,\bar\beta]$)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT + "sweep.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
