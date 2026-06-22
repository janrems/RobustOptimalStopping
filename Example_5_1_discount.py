"""Example_5_1_discount.py

Paper §5.1 — bounded discount-rate ambiguity.

Risk measure: the discount rate is ambiguous in a band [beta_lo, beta_hi];
the probability measure is fixed. The reflected-BSDE driver is

    g(t, y, z) = sup_{beta in [beta_lo, beta_hi]} { -beta y }
               = max(-beta_lo * y, -beta_hi * y).

No z-term: pure discount ambiguity. Both band endpoints act only if Y changes
sign, so we use a SIGN-CHANGING payoff (xi_t = X_t, dX_t = dW_t). Then beta_hi
discounts the liability side (y < 0) and beta_lo the asset side (y >= 0); that
asymmetry is the cash-subadditivity §5.1 is about. A nonnegative payoff would
pin Y <= 0 and collapse the band to beta_hi (see §5.4 / the put).

Upper-reflected BSDE:
    Y_t = -xi_T + int_t^T g(s, Y_s, Z_s) ds - int_t^T Z_s dW_s - (K_T - K_t),
    Y_t <= -xi_t,    tau* = inf{ s >= t : Y_s = -xi_s }.

Trains the backward scheme, then reports Y_0, stopping times, and the
asset/liability split that shows both band endpoints are active.
"""

import matplotlib
matplotlib.use("Agg")

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from RBSDE import BSDEiter, Model, Result, fbsde

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOWER_SENTINEL = -1e6


def run(beta_lo, beta_hi, out_dir, seed=0, N=50, itr=100, dim_h=50,
        batch_size=2 ** 10, multiplier=5, T=1.0, x0_value=0.0, diagnose=True):
    """Train §5.1 for a discount band [beta_lo, beta_hi]; return summary dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim_x = dim_y = dim_d = 1
    path = out_dir + "state_dicts/"
    graph_path = out_dir + "Graphs/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(graph_path, exist_ok=True)

    def b(t, x):
        return torch.zeros_like(x)

    def sigma(t, x):
        return torch.ones(x.size(0), dim_x, dim_x, device=x.device)

    def xi(t, x):
        # sign-changing obstacle process xi_t = X_t
        return x

    def f(t, x, y, z):
        # g(t, y, z) = sup_beta {-beta y} = max(-beta_lo y, -beta_hi y); no z-term
        return torch.maximum(-beta_lo * y, -beta_hi * y)

    def g(x):
        return -xi(T, x)

    def lower_barrier(t, x):
        return torch.full_like(x, LOWER_SENTINEL)

    def upper_barrier(t, x):
        return -xi(t, x)

    x_0 = torch.tensor(x0_value, dtype=torch.float32, device=device)
    equation = fbsde(x_0, b, sigma, f, g, lower_barrier, upper_barrier,
                     T, dim_x, dim_y, dim_d)

    params = dict(dim_x=dim_x, dim_y=dim_y, dim_d=dim_d, dim_h=dim_h, N=N,
                  itr=itr, batch_size=batch_size, multiplier=multiplier,
                  x0_value=x0_value, T=T, beta_lo=beta_lo, beta_hi=beta_hi,
                  seed=seed)
    with open(os.path.join(path, "params.json"), "w") as h:
        json.dump(params, h, indent=2)

    start = time.time()
    loss, y = bsde_train(equation, dim_h, batch_size, N, path, itr, multiplier)
    mins = (time.time() - start) / 60.0
    Y0 = float(y[0, 0])
    print(f"[5.1] band [{beta_lo:.3f},{beta_hi:.3f}]  Y_0={Y0:+.5f}  ({mins:.1f} min)")

    with open(path + "loss.json", "w") as p:
        json.dump(loss, p, indent=2)
    with open(path + "Y0.json", "w") as p:
        json.dump({"Y0": Y0}, p, indent=2)

    summary = {"beta_lo": beta_lo, "beta_hi": beta_hi, "Y0": Y0, "seed": seed}
    if diagnose:
        summary.update(_diagnose(equation, dim_h, path, graph_path, batch_size, N, T,
                                 upper_barrier, beta_lo, beta_hi))
    with open(out_dir + "summary.json", "w") as p:
        json.dump(summary, p, indent=2)
    return summary


def bsde_train(equation, dim_h, batch_size, N, path, itr, multiplier):
    return BSDEiter(equation, dim_h).train_whole(batch_size, N, path, itr, multiplier)


def _diagnose(equation, dim_h, path, graph_path, batch_size, N, T,
              upper_barrier, beta_lo, beta_hi):
    model = Model(equation, dim_h)
    model.eval()
    result = Result(model, equation)

    flag = True
    while flag:
        W = result.gen_b_motion(batch_size, N)
        x = result.gen_x(batch_size, N, W)
        flag = torch.isnan(x).any()

    y, z = result.predict(N, batch_size, x, path)

    t = torch.linspace(0, T, N)
    y_np = y.detach().cpu().numpy()
    upper_np = upper_barrier(t, x).detach().cpu().numpy()

    # ---- sample trajectories
    fig, ax = plt.subplots(figsize=(8, 5))
    for j in np.random.choice(batch_size, size=3, replace=False):
        ax.plot(t, y_np[j, 0, :], label=f"Y (sample {j})")
        ax.plot(t, upper_np[j, 0, :], "--", alpha=0.6, label=f"-xi (sample {j})")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("t")
    ax.set_title(r"$Y_t$ and upper obstacle $-\xi_t = -X_t$")
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "Y_trajectories.png")
    plt.close()

    # ---- asset/liability split: fraction with Y_t >= 0 over time
    #      Y >= 0 -> asset region (beta_lo binds);  Y < 0 -> liability (beta_hi binds)
    frac_asset = (y_np[:, 0, :] >= 0).mean(axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(t.numpy(), frac_asset)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("t")
    plt.ylabel(r"fraction of paths with $Y_t \geq 0$")
    plt.title(rf"Asset-region fraction (β_lo binds) — band [{beta_lo},{beta_hi}]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "asset_liability_split.png")
    plt.close()

    # both branches active iff Y visits both signs at interior times
    interior = y_np[:, 0, 1:-1]
    frac_asset_overall = float((interior >= 0).mean())

    # ---- stopping times
    tol = 1e-3
    exit_idx = []
    for j in range(batch_size):
        diff = upper_np[j, 0, :-1] - y_np[j, 0, :-1]
        hits = diff < tol
        exit_idx.append(int(np.argmax(hits)) if hits.any() else N)
    exit_times = np.array(exit_idx) / N
    stopped_early = exit_times < (N - 1) / N

    mean_tau = float(exit_times[stopped_early].mean()) if stopped_early.any() else None
    print(f"      asset-region fraction (interior): {frac_asset_overall:.3f}")
    print(f"      both branches active: {0.0 < frac_asset_overall < 1.0}")
    print(f"      fraction stopping early: {stopped_early.mean():.3f}")

    return {
        "frac_asset_interior": frac_asset_overall,
        "both_branches_active": bool(0.0 < frac_asset_overall < 1.0),
        "frac_stopping_early": float(stopped_early.mean()),
        "mean_tau_star": mean_tau,
    }


if __name__ == "__main__":
    # default illustrative band [0, 0.10]
    run(beta_lo=0.0, beta_hi=0.10, out_dir="Example_5_1_discount/")
