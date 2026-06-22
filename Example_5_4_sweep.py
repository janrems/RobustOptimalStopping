"""Example_5_4_sweep.py

§5.4 ambiguity sweep: vary the entropic radius gamma_bar (worst-case discount
delta_bar fixed at the risk-free rate) and record the conservative put price,
time value and stopping behaviour. Reuses Example_5_4_put.run.

gamma_bar = 0 is the classical American put at rate delta_bar — the anchor as
ambiguity vanishes. Several replicates per grid point, paired seeds.
"""

import matplotlib
matplotlib.use("Agg")

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from Example_5_4_put import run

OUT = "Example_5_4_sweep/"
GAMMA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]
DELTA_BAR = 0.05
REPLICATES = 3


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for gi, gamma_bar in enumerate(GAMMA_GRID):
        for r in range(REPLICATES):
            sub = f"{OUT}run_{gi:02d}g{gamma_bar:.1f}_r{r}/"
            s = run(gamma_bar=gamma_bar, delta_bar=DELTA_BAR, out_dir=sub, seed=r,
                    diagnose=True)
            rows.append({"gamma_bar": gamma_bar, "rep": r,
                         "put_price": s["put_price"], "time_value": s["time_value"],
                         "mean_tau": s["mean_tau_star"],
                         "max_violation": s["max_obstacle_violation"]})

    with open(OUT + "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _plot(rows)
    print(f"[5.4 sweep] wrote {OUT}summary.csv and sweep.png")


def _agg(rows, key):
    grid = sorted({r["gamma_bar"] for r in rows})
    mean, std = [], []
    for gm in grid:
        vals = [r[key] for r in rows if r["gamma_bar"] == gm and r[key] is not None]
        mean.append(np.mean(vals) if vals else np.nan)
        std.append(np.std(vals) if vals else np.nan)
    return np.array(grid), np.array(mean), np.array(std)


def _plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    g, m, s = _agg(rows, "put_price")
    axes[0].errorbar(g, m, yerr=s, marker="o", capsize=3)
    axes[0].set_xlabel(r"entropic radius $\bar\gamma$")
    axes[0].set_ylabel(r"conservative put price $-Y_0$")
    axes[0].set_title(r"Put price vs entropic ambiguity")
    axes[0].grid(True)

    g, m, s = _agg(rows, "time_value")
    axes[1].errorbar(g, m, yerr=s, marker="o", capsize=3, color="C2")
    axes[1].set_xlabel(r"entropic radius $\bar\gamma$")
    axes[1].set_ylabel(r"time value $-Y_0-(K-S_0)$")
    axes[1].set_title("Time value vs entropic ambiguity")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUT + "sweep.png")
    plt.close()


if __name__ == "__main__":
    main()
