"""Example_1_2_sweep.py

Sweep gamma_alpha = gamma_beta over increasing values to find the breakdown
point of the deep BSDE solver on the upper-reflected quadratic-driver problem
from Example 1.2.

Each (coefficient, replicate) run is trained from scratch and tested. Results
go to Example_1_2_sweep/run_<id>/, with a top-level summary.csv and an
aggregate breakdown.png.
"""

import torch
import numpy as np
import os
import json
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RBSDE import fbsde, BSDEiter, Model, Result


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ---- Fixed problem / solver settings (mirror Example_1_2.py defaults)
T = 1.0
DIM_X = DIM_Y = DIM_D = 1
DIM_H = 50
N = 50
ITR = 100
BATCH_SIZE = 2 ** 10
MULTIPLIER = 5
X0_VALUE = 0.0

LOWER_SENTINEL = -1e6           # makes lower reflection non-binding
DIVERGENCE_THRESHOLD = 1e6      # any |loss| > this  ->  flag as diverged


def make_problem(gamma_alpha, gamma_beta, beta_bar):
    """Return (b, sigma, f, g, lower, upper) closures for Example 1.2."""

    def b(t, x):
        return torch.zeros_like(x)

    def sigma(t, x):
        return torch.ones(x.size(0), DIM_X, DIM_X, device=x.device)

    def xi(t, x):
        return x

    def f(t, x, y, z):
        z_sq = (z ** 2).sum(dim=-1)
        return 0.5 * gamma_alpha * z_sq - beta_bar * y + 0.5 * gamma_beta * y ** 2

    def g_term(x):
        return -xi(T, x)

    def lower(t, x):
        return torch.full_like(x, LOWER_SENTINEL)

    def upper(t, x):
        return -xi(t, x)

    return b, sigma, f, g_term, lower, upper


def has_bad_loss(loss):
    """True if any per-step loss is non-finite or above DIVERGENCE_THRESHOLD."""
    for step_losses in loss:
        a = np.asarray(step_losses, dtype=float)
        if a.size == 0:
            continue
        if not np.isfinite(a).all() or np.max(np.abs(a)) > DIVERGENCE_THRESHOLD:
            return True
    return False


def tail_mean(arr, frac=0.1):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return float("nan")
    k = max(1, int(frac * a.size))
    return float(np.mean(a[-k:]))


def run_experiment(gamma_alpha, gamma_beta, beta_bar, run_dir, seed):
    """Train + test one (coefficients, seed) configuration. Returns a flat dict."""
    os.makedirs(run_dir, exist_ok=True)
    path_prefix = run_dir if run_dir.endswith("/") else run_dir + "/"

    torch.manual_seed(seed)
    np.random.seed(seed)

    b, sigma, f, g_term, lower, upper = make_problem(
        gamma_alpha, gamma_beta, beta_bar)

    x_0 = torch.tensor(X0_VALUE, dtype=torch.float32, device=device)
    equation = fbsde(x_0, b, sigma, f, g_term, lower, upper,
                     T, DIM_X, DIM_Y, DIM_D)

    params = dict(
        gamma_alpha=gamma_alpha, gamma_beta=gamma_beta, beta_bar=beta_bar,
        seed=seed, dim_x=DIM_X, dim_y=DIM_Y, dim_d=DIM_D, dim_h=DIM_H,
        N=N, itr=ITR, batch_size=BATCH_SIZE, multiplier=MULTIPLIER,
        T=T, x0_value=X0_VALUE,
    )
    with open(path_prefix + "params.json", "w") as fh:
        json.dump(params, fh, indent=2)

    summary = dict(
        run_dir=run_dir, seed=seed,
        gamma_alpha=gamma_alpha, gamma_beta=gamma_beta, beta_bar=beta_bar,
    )

    # ---- Train
    t0 = time.time()
    train_ok = True
    loss = []
    Y0_train = float("nan")
    try:
        bsde_itr = BSDEiter(equation, DIM_H)
        loss, y_train = bsde_itr.train_whole(
            BATCH_SIZE, N, path_prefix, ITR, MULTIPLIER)
        Y0_train = float(y_train[0, 0])
    except Exception as e:
        train_ok = False
        print(f"  ! train exception: {type(e).__name__}: {e}")
    summary["train_time_s"] = time.time() - t0
    summary["train_ok"] = train_ok
    summary["Y0_train"] = Y0_train

    if train_ok:
        with open(path_prefix + "loss.json", "w") as fh:
            json.dump(loss, fh, indent=2)
        with open(path_prefix + "Y0.json", "w") as fh:
            json.dump([Y0_train], fh, indent=2)

    diverged = (not train_ok) or has_bad_loss(loss)
    summary["diverged"] = bool(diverged)

    if loss:
        per_step_tail = [tail_mean(s) for s in loss]
        # loss is appended in order n = N-2, N-3, ..., 0
        summary["loss_terminal_step"] = per_step_tail[0]   # n = N-2
        summary["loss_initial_step"] = per_step_tail[-1]   # n = 0
        summary["max_step_loss"] = float(np.nanmax(per_step_tail))
    else:
        summary["loss_terminal_step"] = float("nan")
        summary["loss_initial_step"] = float("nan")
        summary["max_step_loss"] = float("nan")

    # ---- Test (if training looks usable)
    summary["test_ok"] = False
    summary["frac_stop"] = float("nan")
    summary["mean_tau"] = float("nan")

    if train_ok and not diverged:
        try:
            model = Model(equation, DIM_H)
            model.eval()
            result = Result(model, equation)

            # generate clean Brownian + state path
            x = None
            for _ in range(10):
                W = result.gen_b_motion(BATCH_SIZE, N)
                x_try = result.gen_x(BATCH_SIZE, N, W)
                if not torch.isnan(x_try).any():
                    x = x_try
                    break
            if x is None:
                raise RuntimeError("gen_x produced NaN repeatedly")

            y, _ = result.predict(N, BATCH_SIZE, x, path_prefix)

            t = torch.linspace(0, T, N)
            y_np = y.detach().cpu().numpy()
            upper_np = upper(t, x).detach().cpu().numpy()

            tol = 1e-3
            exit_idx = np.full(BATCH_SIZE, N, dtype=int)
            for j in range(BATCH_SIZE):
                diff = upper_np[j, 0, :] - y_np[j, 0, :]
                hits = diff < tol
                if hits.any():
                    exit_idx[j] = int(np.argmax(hits))
            exit_times = exit_idx / N
            stopped = exit_times < 1.0
            summary["frac_stop"] = float(stopped.mean())
            summary["mean_tau"] = (float(exit_times[stopped].mean())
                                    if stopped.any() else float("nan"))
            summary["test_ok"] = True

            # diagnostic plot per run
            fig, ax = plt.subplots(figsize=(8, 5))
            for j in np.random.choice(BATCH_SIZE, size=3, replace=False):
                ax.plot(t, y_np[j, 0, :], label=f"Y[{j}]")
                ax.plot(t, upper_np[j, 0, :], "--", alpha=0.5,
                        label=f"-xi[{j}]")
            ax.set_xlabel("t")
            ax.grid(True)
            ax.set_title(rf"$\gamma_\alpha=\gamma_\beta={gamma_alpha}$, "
                         rf"seed={seed}, $Y_0\approx{Y0_train:.3f}$")
            ax.legend(loc="best", fontsize=7)
            plt.tight_layout()
            plt.savefig(path_prefix + "Y_trajectories.png")
            plt.close()
        except Exception as e:
            print(f"  ! test exception: {type(e).__name__}: {e}")

    # save summary inside the run folder too
    with open(path_prefix + "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary


def main():
    sweep_root = "Example_1_2_sweep/"
    os.makedirs(sweep_root, exist_ok=True)

    coefficients = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_replicates = 3
    beta_bar = 0.05

    results = []
    for i, c in enumerate(coefficients):
        for r in range(n_replicates):
            run_id = f"run_{i:02d}r{r}_ga{c}_gb{c}"
            run_dir = sweep_root + run_id + "/"
            # paired across coefficients within a replicate index
            seed = 100 * (r + 1) + 7

            print(f"\n=== {run_id}  (seed={seed}) ===")
            try:
                s = run_experiment(c, c, beta_bar, run_dir, seed)
            except Exception as e:
                print(f"  ! run-level exception: {type(e).__name__}: {e}")
                s = dict(
                    run_dir=run_dir, seed=seed,
                    gamma_alpha=c, gamma_beta=c, beta_bar=beta_bar,
                    train_ok=False, test_ok=False, diverged=True,
                    Y0_train=float("nan"), train_time_s=float("nan"),
                    loss_terminal_step=float("nan"),
                    loss_initial_step=float("nan"),
                    max_step_loss=float("nan"),
                    frac_stop=float("nan"), mean_tau=float("nan"),
                )
            results.append(s)
            print(f"  Y0={s.get('Y0_train', float('nan')):.4f}  "
                  f"diverged={s.get('diverged')}  "
                  f"frac_stop={s.get('frac_stop', float('nan'))}  "
                  f"max_step_loss={s.get('max_step_loss', float('nan')):.3e}")

    # ---- Summary CSV
    csv_path = sweep_root + "summary.csv"
    keys = ["run_dir", "seed", "gamma_alpha", "gamma_beta", "beta_bar",
            "train_ok", "diverged", "test_ok",
            "Y0_train", "loss_terminal_step", "loss_initial_step",
            "max_step_loss", "frac_stop", "mean_tau", "train_time_s"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for s in results:
            writer.writerow({k: s.get(k, "") for k in keys})
    print(f"\nWrote {csv_path}")

    # ---- Aggregate plot
    by_c = {c: [s for s in results if s["gamma_alpha"] == c]
            for c in coefficients}
    cs = np.array(coefficients)
    Y0_means, Y0_stds, loss_means = [], [], []
    for c in cs:
        ys = np.array([s["Y0_train"] for s in by_c[c]], dtype=float)
        ls = np.array([s.get("max_step_loss", float("nan"))
                       for s in by_c[c]], dtype=float)
        Y0_means.append(np.nanmean(ys))
        Y0_stds.append(np.nanstd(ys))
        loss_means.append(np.nanmean(ls))
    Y0_means = np.array(Y0_means)
    Y0_stds = np.array(Y0_stds)
    loss_means = np.array(loss_means)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(cs, Y0_means, yerr=Y0_stds, marker="o", capsize=3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\gamma_\alpha = \gamma_\beta$")
    axes[0].set_ylabel(r"$Y_0$ (mean $\pm$ std)")
    axes[0].set_title(rf"Initial value vs coefficient ($\bar\beta = {beta_bar}$)")
    axes[0].grid(True)

    axes[1].plot(cs, loss_means, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\gamma_\alpha = \gamma_\beta$")
    axes[1].set_ylabel("max(per-step final loss)  [log]")
    axes[1].set_title("Convergence indicator")
    axes[1].grid(True, which="both")

    plt.tight_layout()
    plt.savefig(sweep_root + "breakdown.png")
    plt.close()
    print(f"Wrote {sweep_root}breakdown.png")


if __name__ == "__main__":
    main()
