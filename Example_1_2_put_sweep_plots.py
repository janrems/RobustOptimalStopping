"""Example_1_2_put_sweep_plots.py

Read a put-sweep summary CSV and write a handful of additional plots beyond
breakdown.png. All plots go to <sweep_root>/Plots/.

Plots produced:
  1. stopping_dynamics.png  — mean tau* and frac_stop_early vs gamma
  2. time_value.png         — (-Y0 - intrinsic) vs gamma, with zero line
  3. replicate_paths.png    — per-seed -Y0 trajectories across gamma
  4. loss_decomp.png        — terminal-step vs initial-step loss vs gamma
  5. phase.png              — -Y0 and frac_stop_early on twin axes

Usage:
    python Example_1_2_put_sweep_plots.py                    # default sweep root
    python Example_1_2_put_sweep_plots.py <sweep_root>/      # any sweep folder

Reads market parameters (K, S0) from any run's params.json so it stays
correct if the sweep is rerun with different settings.
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- Sweep root: argv override, else default
DEFAULT_SWEEP_ROOT = "Example_1_2_put_sweep/"
SWEEP_ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SWEEP_ROOT
if not SWEEP_ROOT.endswith("/"):
    SWEEP_ROOT += "/"
CSV_PATH = SWEEP_ROOT + "summary.csv"
OUT_DIR = SWEEP_ROOT + "Plots/"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"reading from {SWEEP_ROOT}")


# ---- Load market parameters from the first run's params.json
params_files = sorted(glob.glob(SWEEP_ROOT + "run_*/params.json"))
if params_files:
    with open(params_files[0]) as fh:
        p = json.load(fh)
    K = float(p["K"])
    S0 = float(p["S0"])
else:
    K, S0 = 1.1, 1.0   # fallback to defaults
INTRINSIC = max(K - S0, 0.0)


# ---- Load the summary CSV, coerce numeric columns
df = pd.read_csv(CSV_PATH)
numeric_cols = ["Y0_train", "minus_Y0",
                "loss_terminal_step", "loss_initial_step", "max_step_loss",
                "frac_stop_early", "mean_tau", "train_time_s"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# successful runs only
df_ok = df[(df["train_ok"] == True) & (df["test_ok"] == True)].copy()

# adaptive x-axis label: joint sweep vs gamma_alpha-only sweep
gb_unique = df_ok["gamma_beta"].astype(float).unique()
if len(gb_unique) == 1 and gb_unique[0] == 0.0:
    GAMMA_LABEL = r"$\gamma_\alpha$"
elif np.allclose(df_ok["gamma_alpha"].astype(float).values,
                 df_ok["gamma_beta"].astype(float).values):
    GAMMA_LABEL = r"$\gamma_\alpha = \gamma_\beta$"
else:
    GAMMA_LABEL = r"$\gamma_\alpha$"

# per-gamma aggregates
agg = df_ok.groupby("gamma_alpha").agg(
    minus_Y0_mean=("minus_Y0", "mean"),
    minus_Y0_std=("minus_Y0", "std"),
    minus_Y0_min=("minus_Y0", "min"),
    minus_Y0_max=("minus_Y0", "max"),
    mean_tau_mean=("mean_tau", "mean"),
    mean_tau_std=("mean_tau", "std"),
    frac_stop_early_mean=("frac_stop_early", "mean"),
    frac_stop_early_std=("frac_stop_early", "std"),
    loss_terminal_mean=("loss_terminal_step", "mean"),
    loss_initial_mean=("loss_initial_step", "mean"),
    train_time_mean=("train_time_s", "mean"),
).reset_index()

cs = agg["gamma_alpha"].values


# ---- 1) Stopping dynamics
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.errorbar(cs, agg["mean_tau_mean"], yerr=agg["mean_tau_std"],
             marker="o", color="C0", capsize=3, label=r"mean $\tau^*$")
ax2.errorbar(cs, agg["frac_stop_early_mean"],
             yerr=agg["frac_stop_early_std"],
             marker="s", color="C1", capsize=3,
             label="frac stop early")
ax1.set_xscale("log")
ax1.set_xlabel(GAMMA_LABEL)
ax1.set_ylabel(r"mean conditional $\tau^*$", color="C0")
ax2.set_ylabel("fraction stopping early", color="C1")
ax1.tick_params(axis="y", labelcolor="C0")
ax2.tick_params(axis="y", labelcolor="C1")
ax1.set_title("Stopping behaviour vs ambiguity radius")
ax1.grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR + "stopping_dynamics.png")
plt.close(fig)


# ---- 2) Time value
fig, ax = plt.subplots(figsize=(8, 5))
tv_mean = agg["minus_Y0_mean"] - INTRINSIC
tv_std = agg["minus_Y0_std"]
ax.errorbar(cs, tv_mean, yerr=tv_std, marker="o", capsize=3)
ax.axhline(0, color="r", linestyle="--", alpha=0.5,
           label=r"price = intrinsic ($-Y_0 - 0.10 = 0$)")
ax.set_xscale("log")
ax.set_xlabel(GAMMA_LABEL)
ax.set_ylabel(r"$-Y_0 - $ intrinsic  (option time value)")
ax.set_title("Robust time value of the put vs ambiguity")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR + "time_value.png")
plt.close(fig)


# ---- 3) Replicate paths (paired-seed)
fig, ax = plt.subplots(figsize=(8, 5))
df_sorted = df_ok.sort_values(["seed", "gamma_alpha"])
for seed_val, g in df_sorted.groupby("seed"):
    ax.plot(g["gamma_alpha"], g["minus_Y0"], marker="o",
            alpha=0.85, label=f"seed = {int(seed_val)}")
ax.axhline(INTRINSIC, color="r", linestyle="--", alpha=0.4,
           label=f"intrinsic = {INTRINSIC:.2f}")
ax.set_xscale("log")
ax.set_xlabel(GAMMA_LABEL)
ax.set_ylabel(r"$-Y_0$  (robust price)")
ax.set_title("Per-replicate price (paired Brownian increments across $\\gamma$)")
ax.grid(True)
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig(OUT_DIR + "replicate_paths.png")
plt.close(fig)


# ---- 4) Loss decomposition: terminal vs initial step
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cs, agg["loss_terminal_mean"], marker="o",
        label=r"final loss at terminal step ($n = N-2$)")
ax.plot(cs, agg["loss_initial_mean"], marker="s",
        label=r"final loss at initial step ($n = 0$)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(GAMMA_LABEL)
ax.set_ylabel("final-iteration loss [log]")
ax.set_title("Where does the solver work hardest?")
ax.grid(True, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + "loss_decomp.png")
plt.close(fig)


# ---- 5) Phase view: -Y0 and frac_stop_early on twin axes
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.errorbar(cs, agg["minus_Y0_mean"], yerr=agg["minus_Y0_std"],
             marker="o", color="C0", capsize=3, label=r"$-Y_0$")
ax2.plot(cs, agg["frac_stop_early_mean"],
         marker="s", color="C2", linestyle="-", label="frac stop early")
ax1.axhline(INTRINSIC, color="r", linestyle="--", alpha=0.4,
            label=f"intrinsic = {INTRINSIC:.2f}")
ax1.set_xscale("log")
ax1.set_xlabel(GAMMA_LABEL)
ax1.set_ylabel(r"$-Y_0$  (robust price)", color="C0")
ax2.set_ylabel("fraction stopping early", color="C2")
ax1.tick_params(axis="y", labelcolor="C0")
ax2.tick_params(axis="y", labelcolor="C2")
ax1.set_title("Price falls, exercise becomes immediate as $\\gamma$ grows")
ax1.grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR + "phase.png")
plt.close(fig)


print(f"Wrote 5 plots to {OUT_DIR}")
for name in ("stopping_dynamics.png", "time_value.png",
             "replicate_paths.png", "loss_decomp.png", "phase.png"):
    print(f"  {OUT_DIR}{name}")
