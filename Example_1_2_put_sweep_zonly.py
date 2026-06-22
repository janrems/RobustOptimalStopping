"""Example_1_2_put_sweep_zonly.py

Stress test the deep BSDE solver on the geometric American put with the driver
collapsed to its z-only piece:

    g(t, y, z) = (gamma_alpha / 2) |z|^2,        beta_bar = 0,  gamma_beta = 0.

Isolates the effect of L^2 model ambiguity from any discount / discount-ambiguity
contribution. Same problem setup as Example_1_2_put_sweep.py (slightly ITM put
with K=1.1, S0=1), and same diagnostic structure (per-run folders, summary.csv,
breakdown.png). Coefficient grid extended to gamma_alpha up to 50, since the
stabilizing y- and y^2-terms have been removed.
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

# ---- Market parameters (same as Example_1_2_put_sweep)
S0 = 1.0
K = 1.1
SIGMA_S = 0.2
R = 0.05
T = 1.0

# ---- Solver settings
DIM_X = DIM_Y = DIM_D = 1
DIM_H = 50
N = 50
ITR = 100
BATCH_SIZE = 2 ** 10
MULTIPLIER = 5
X0_VALUE = float(np.log(S0))

LOWER_SENTINEL = -1e6
DIVERGENCE_THRESHOLD = 1e6


def make_problem(gamma_alpha, gamma_beta, beta_bar):
    """GBM (log-price) + Example 1.2 quadratic driver + put obstacle."""

    def b(t, x):
        return torch.full_like(x, R - 0.5 * SIGMA_S ** 2)

    def sigma(t, x):
        return torch.full((x.size(0), DIM_X, DIM_X), SIGMA_S, device=x.device)

    def xi(t, x):
        return torch.clamp(K - torch.exp(x), min=0.0)

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
        seed=seed,
        S0=S0, K=K, sigma_S=SIGMA_S, r=R,
        dim_x=DIM_X, dim_y=DIM_Y, dim_d=DIM_D, dim_h=DIM_H,
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
    summary["minus_Y0"] = -Y0_train if train_ok else float("nan")

    if train_ok:
        with open(path_prefix + "loss.json", "w") as fh:
            json.dump(loss, fh, indent=2)
        with open(path_prefix + "Y0.json", "w") as fh:
            json.dump([Y0_train], fh, indent=2)

    diverged = (not train_ok) or has_bad_loss(loss)
    summary["diverged"] = bool(diverged)

    if loss:
        per_step_tail = [tail_mean(s) for s in loss]
        summary["loss_terminal_step"] = per_step_tail[0]
        summary["loss_initial_step"] = per_step_tail[-1]
        summary["max_step_loss"] = float(np.nanmax(per_step_tail))
    else:
        summary["loss_terminal_step"] = float("nan")
        summary["loss_initial_step"] = float("nan")
        summary["max_step_loss"] = float("nan")

    summary["test_ok"] = False
    summary["frac_stop_early"] = float("nan")
    summary["mean_tau"] = float("nan")

    if train_ok and not diverged:
        try:
            model = Model(equation, DIM_H)
            model.eval()
            result = Result(model, equation)

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
            S_np = np.exp(x.detach().cpu().numpy())

            tol = 1e-3
            exit_idx = np.full(BATCH_SIZE, N, dtype=int)
            for j in range(BATCH_SIZE):
                diff = upper_np[j, 0, :] - y_np[j, 0, :]
                hits = diff[:-1] < tol
                if hits.any():
                    exit_idx[j] = int(np.argmax(hits))
            exit_times = exit_idx / N
            stopped_early = exit_times < (N - 1) / N
            summary["frac_stop_early"] = float(stopped_early.mean())
            summary["mean_tau"] = (float(exit_times[stopped_early].mean())
                                    if stopped_early.any() else float("nan"))
            summary["test_ok"] = True

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for j in np.random.choice(BATCH_SIZE, size=3, replace=False):
                axes[0].plot(t, S_np[j, 0, :], label=f"S[{j}]")
                axes[1].plot(t, y_np[j, 0, :], label=f"Y[{j}]")
                axes[1].plot(t, upper_np[j, 0, :], "--", alpha=0.5,
                             label=f"-xi[{j}]")
            axes[0].axhline(K, color="k", linestyle=":", alpha=0.6,
                            label=f"K={K}")
            axes[0].set_title(rf"$S_t$ — $\gamma_\alpha={gamma_alpha}$, seed={seed}")
            axes[0].set_xlabel("t"); axes[0].grid(True)
            axes[0].legend(loc="best", fontsize=7)
            axes[1].set_title(
                rf"$Y_t$ vs $-\xi_t$, $-Y_0\approx{-Y0_train:.4f}$")
            axes[1].set_xlabel("t"); axes[1].grid(True)
            axes[1].legend(loc="best", fontsize=7)
            plt.tight_layout()
            plt.savefig(path_prefix + "Y_trajectories.png")
            plt.close()
        except Exception as e:
            print(f"  ! test exception: {type(e).__name__}: {e}")

    with open(path_prefix + "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary


def main():
    sweep_root = "Example_1_2_put_sweep_zonly/"
    os.makedirs(sweep_root, exist_ok=True)

    # Coefficient grid (option b): same low end as full-driver sweep, push higher
    coefficients = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    n_replicates = 3

    # z-only setting: kill discount and y^2 terms
    gamma_beta = 0.0
    beta_bar = 0.0

    results = []
    for i, c in enumerate(coefficients):
        for r in range(n_replicates):
            run_id = f"run_{i:02d}r{r}_ga{c}"
            run_dir = sweep_root + run_id + "/"
            seed = 100 * (r + 1) + 7

            print(f"\n=== {run_id}  (seed={seed}) ===")
            try:
                s = run_experiment(c, gamma_beta, beta_bar, run_dir, seed)
            except Exception as e:
                print(f"  ! run-level exception: {type(e).__name__}: {e}")
                s = dict(
                    run_dir=run_dir, seed=seed,
                    gamma_alpha=c, gamma_beta=gamma_beta, beta_bar=beta_bar,
                    train_ok=False, test_ok=False, diverged=True,
                    Y0_train=float("nan"), minus_Y0=float("nan"),
                    train_time_s=float("nan"),
                    loss_terminal_step=float("nan"),
                    loss_initial_step=float("nan"),
                    max_step_loss=float("nan"),
                    frac_stop_early=float("nan"), mean_tau=float("nan"),
                )
            results.append(s)
            print(f"  Y0={s.get('Y0_train', float('nan')):.4f}  "
                  f"-Y0={s.get('minus_Y0', float('nan')):.4f}  "
                  f"diverged={s.get('diverged')}  "
                  f"frac_stop_early={s.get('frac_stop_early', float('nan'))}  "
                  f"max_step_loss={s.get('max_step_loss', float('nan')):.3e}")

    # ---- Summary CSV
    csv_path = sweep_root + "summary.csv"
    keys = ["run_dir", "seed", "gamma_alpha", "gamma_beta", "beta_bar",
            "train_ok", "diverged", "test_ok",
            "Y0_train", "minus_Y0",
            "loss_terminal_step", "loss_initial_step", "max_step_loss",
            "frac_stop_early", "mean_tau", "train_time_s"]
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
    minusY0_means, minusY0_stds, loss_means = [], [], []
    for c in cs:
        ys = np.array([s["minus_Y0"] for s in by_c[c]], dtype=float)
        ls = np.array([s.get("max_step_loss", float("nan"))
                       for s in by_c[c]], dtype=float)
        minusY0_means.append(np.nanmean(ys))
        minusY0_stds.append(np.nanstd(ys))
        loss_means.append(np.nanmean(ls))
    minusY0_means = np.array(minusY0_means)
    minusY0_stds = np.array(minusY0_stds)
    loss_means = np.array(loss_means)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(cs, minusY0_means, yerr=minusY0_stds,
                     marker="o", capsize=3)
    axes[0].axhline(max(K - S0, 0), color="r", linestyle="--", alpha=0.5,
                    label=f"intrinsic = {max(K - S0, 0):.3f}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\gamma_\alpha$")
    axes[0].set_ylabel(r"$-Y_0$  (robust put price, z-only driver)")
    axes[0].set_title(rf"K={K}, $S_0$={S0}, $\sigma$={SIGMA_S}, "
                      rf"$\bar\beta=\gamma_\beta=0$")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(cs, loss_means, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\gamma_\alpha$")
    axes[1].set_ylabel("max(per-step final loss)  [log]")
    axes[1].set_title("Convergence indicator")
    axes[1].grid(True, which="both")

    plt.tight_layout()
    plt.savefig(sweep_root + "breakdown.png")
    plt.close()
    print(f"Wrote {sweep_root}breakdown.png")


if __name__ == "__main__":
    main()
