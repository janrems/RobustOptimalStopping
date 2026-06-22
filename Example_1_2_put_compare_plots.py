"""Example_1_2_put_compare_plots.py

Overlay the full-driver and z-only put sweeps on shared axes to make the
contribution of the y / y^2 terms visible at a glance.

Reads:
    Example_1_2_put_sweep/summary.csv         (full driver,  gamma_alpha = gamma_beta)
    Example_1_2_put_sweep_zonly/summary.csv   (z-only,       gamma_beta = beta_bar = 0)

Writes (to Example_1_2_put_compare/):
    minus_Y0_compare.png       — robust put price vs gamma_alpha, both curves
    frac_stop_compare.png      — fraction stopping early vs gamma_alpha, both curves
    mean_tau_compare.png       — mean conditional tau* vs gamma_alpha, both curves
    loss_compare.png           — max per-step final loss vs gamma_alpha, both curves
"""

import os
import glob
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FULL_ROOT  = "Example_1_2_put_sweep/"
ZONLY_ROOT = "Example_1_2_put_sweep_zonly/"
OUT_DIR    = "Example_1_2_put_compare/"
os.makedirs(OUT_DIR, exist_ok=True)


def load_aggregate(sweep_root):
    """Return per-gamma_alpha aggregates from a sweep folder."""
    csv_path = sweep_root + "summary.csv"
    df = pd.read_csv(csv_path)
    numeric_cols = ["Y0_train", "minus_Y0",
                    "loss_terminal_step", "loss_initial_step", "max_step_loss",
                    "frac_stop_early", "mean_tau", "train_time_s",
                    "gamma_alpha", "gamma_beta", "beta_bar"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df_ok = df[(df["train_ok"] == True) & (df["test_ok"] == True)].copy()
    agg = df_ok.groupby("gamma_alpha").agg(
        minus_Y0_mean=("minus_Y0", "mean"),
        minus_Y0_std=("minus_Y0", "std"),
        mean_tau_mean=("mean_tau", "mean"),
        mean_tau_std=("mean_tau", "std"),
        frac_stop_early_mean=("frac_stop_early", "mean"),
        frac_stop_early_std=("frac_stop_early", "std"),
        max_step_loss_mean=("max_step_loss", "mean"),
    ).reset_index().sort_values("gamma_alpha")
    return agg


def load_market(sweep_root):
    files = sorted(glob.glob(sweep_root + "run_*/params.json"))
    if not files:
        return 1.1, 1.0
    with open(files[0]) as fh:
        p = json.load(fh)
    return float(p["K"]), float(p["S0"])


agg_full  = load_aggregate(FULL_ROOT)
agg_zonly = load_aggregate(ZONLY_ROOT)
K, S0 = load_market(FULL_ROOT)
INTRINSIC = max(K - S0, 0.0)

LABEL_FULL  = r"full driver ($\gamma_\alpha = \gamma_\beta$, $\bar\beta = r$)"
LABEL_ZONLY = r"z-only ($\gamma_\beta = \bar\beta = 0$)"


# ---- 1) -Y0  (robust put price)
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(agg_full["gamma_alpha"], agg_full["minus_Y0_mean"],
            yerr=agg_full["minus_Y0_std"],
            marker="o", capsize=3, label=LABEL_FULL)
ax.errorbar(agg_zonly["gamma_alpha"], agg_zonly["minus_Y0_mean"],
            yerr=agg_zonly["minus_Y0_std"],
            marker="s", capsize=3, label=LABEL_ZONLY)
ax.axhline(INTRINSIC, color="r", linestyle="--", alpha=0.4,
           label=f"intrinsic = {INTRINSIC:.2f}")
ax.set_xscale("log")
ax.set_xlabel(r"$\gamma_\alpha$")
ax.set_ylabel(r"$-Y_0$  (robust put price)")
ax.set_title("Robust put price: full driver vs z-only driver")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + "minus_Y0_compare.png")
plt.close(fig)


# ---- 2) Fraction stopping early
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(agg_full["gamma_alpha"], agg_full["frac_stop_early_mean"],
            yerr=agg_full["frac_stop_early_std"],
            marker="o", capsize=3, label=LABEL_FULL)
ax.errorbar(agg_zonly["gamma_alpha"], agg_zonly["frac_stop_early_mean"],
            yerr=agg_zonly["frac_stop_early_std"],
            marker="s", capsize=3, label=LABEL_ZONLY)
ax.set_xscale("log")
ax.set_xlabel(r"$\gamma_\alpha$")
ax.set_ylabel("fraction stopping early")
ax.set_title("Early-exercise fraction: full vs z-only")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + "frac_stop_compare.png")
plt.close(fig)


# ---- 3) Mean conditional tau*
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(agg_full["gamma_alpha"], agg_full["mean_tau_mean"],
            yerr=agg_full["mean_tau_std"],
            marker="o", capsize=3, label=LABEL_FULL)
ax.errorbar(agg_zonly["gamma_alpha"], agg_zonly["mean_tau_mean"],
            yerr=agg_zonly["mean_tau_std"],
            marker="s", capsize=3, label=LABEL_ZONLY)
ax.set_xscale("log")
ax.set_xlabel(r"$\gamma_\alpha$")
ax.set_ylabel(r"mean conditional $\tau^*$")
ax.set_title("Optimal stopping time: full vs z-only")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + "mean_tau_compare.png")
plt.close(fig)


# ---- 4) Max per-step final loss
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(agg_full["gamma_alpha"], agg_full["max_step_loss_mean"],
        marker="o", label=LABEL_FULL)
ax.plot(agg_zonly["gamma_alpha"], agg_zonly["max_step_loss_mean"],
        marker="s", label=LABEL_ZONLY)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\gamma_\alpha$")
ax.set_ylabel("max(per-step final loss)  [log]")
ax.set_title("Solver convergence: full vs z-only")
ax.grid(True, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + "loss_compare.png")
plt.close(fig)


print(f"Wrote 4 plots to {OUT_DIR}")
