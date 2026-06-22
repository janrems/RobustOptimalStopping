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


def _agg(rows, key):
    grid = sorted({r["beta_hi"] for r in rows})
    mean, std = [], []
    for b in grid:
        vals = [r[key] for r in rows if r["beta_hi"] == b and r[key] is not None]
        mean.append(np.mean(vals) if vals else np.nan)
        std.append(np.std(vals) if vals else np.nan)
    return np.array(grid), np.array(mean), np.array(std)


def _plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    g, m, s = _agg(rows, "Y0")
    axes[0].errorbar(g, m, yerr=s, marker="o", capsize=3)
    axes[0].set_xlabel(r"band width $\bar\beta$ (band $[0,\bar\beta]$)")
    axes[0].set_ylabel(r"$Y_0$")
    axes[0].set_title(r"Conservative value vs discount-band width")
    axes[0].grid(True)

    g, m, s = _agg(rows, "mean_tau")
    axes[1].errorbar(g, m, yerr=s, marker="o", capsize=3, color="C1")
    axes[1].set_xlabel(r"band width $\bar\beta$")
    axes[1].set_ylabel(r"mean $\tau^*$ (early stops)")
    axes[1].set_title("Stopping vs discount-band width")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUT + "sweep.png")
    plt.close()


if __name__ == "__main__":
    main()
