"""Example_5_4_put.py

Paper §5.4 — geometric American put under entropic-discount ambiguity.

Forward (log-price):  dX_t = (r - sigma^2/2) dt + sigma dW_t,  X_0 = log S_0.
Payoff:               xi_t = (K - S_t)^+ = (K - e^{X_t})^+.

Since Y <= -xi <= 0 everywhere, §5.4's box driver reduces (everywhere) to the
liability-region form, linear in y:

    g(t, y, z) = -delta_bar * y + (gamma_bar / 2) |z|^2.

This is the paper-faithful driver: gamma_beta = 0 (no y^2 penalty),
beta_bar = delta_bar (worst-case discount), gamma_alpha = gamma_bar (entropic
model-ambiguity radius). At gamma_bar = 0 it collapses to the classical
American put at rate delta_bar.

Trains the backward scheme, then reports the conservative price -Y_0, the
time value, stopping behaviour and obstacle adherence.
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


def run(gamma_bar, delta_bar, out_dir, S0=1.0, K=1.1, sigma_S=0.2, r=0.05,
        seed=0, N=50, itr=100, dim_h=50, batch_size=2 ** 10, multiplier=5,
        T=1.0, diagnose=True):
    """Train §5.4 for entropic radius gamma_bar and worst-case discount delta_bar."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim_x = dim_y = dim_d = 1
    path = out_dir + "state_dicts/"
    graph_path = out_dir + "Graphs/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(graph_path, exist_ok=True)

    def b(t, x):
        return torch.full_like(x, r - 0.5 * sigma_S ** 2)

    def sigma(t, x):
        return torch.full((x.size(0), dim_x, dim_x), sigma_S, device=x.device)

    def xi(t, x):
        return torch.clamp(K - torch.exp(x), min=0.0)

    def f(t, x, y, z):
        # g(t, y, z) = (gamma_bar/2)|z|^2 - delta_bar y    (paper §5.4, linear in y)
        z_sq = (z ** 2).sum(dim=-1)
        return 0.5 * gamma_bar * z_sq - delta_bar * y

    def g(x):
        return -xi(T, x)

    def lower_barrier(t, x):
        return torch.full_like(x, LOWER_SENTINEL)

    def upper_barrier(t, x):
        return -xi(t, x)

    x0_value = float(np.log(S0))
    x_0 = torch.tensor(x0_value, dtype=torch.float32, device=device)
    equation = fbsde(x_0, b, sigma, f, g, lower_barrier, upper_barrier,
                     T, dim_x, dim_y, dim_d)

    params = dict(dim_x=dim_x, dim_y=dim_y, dim_d=dim_d, dim_h=dim_h, N=N,
                  itr=itr, batch_size=batch_size, multiplier=multiplier,
                  x0_value=x0_value, T=T, S0=S0, K=K, sigma_S=sigma_S, r=r,
                  gamma_bar=gamma_bar, delta_bar=delta_bar, seed=seed)
    with open(os.path.join(path, "params.json"), "w") as h:
        json.dump(params, h, indent=2)

    start = time.time()
    loss, y = BSDEiter(equation, dim_h).train_whole(batch_size, N, path, itr, multiplier)
    mins = (time.time() - start) / 60.0
    Y0 = float(y[0, 0])
    intrinsic = max(K - S0, 0.0)
    print(f"[5.4] gamma_bar={gamma_bar:.3f} delta_bar={delta_bar:.3f}  "
          f"-Y_0={-Y0:.5f}  intrinsic={intrinsic:.5f}  ({mins:.1f} min)")

    with open(path + "loss.json", "w") as p:
        json.dump(loss, p, indent=2)
    with open(path + "Y0.json", "w") as p:
        json.dump({"Y0": Y0}, p, indent=2)

    summary = {"gamma_bar": gamma_bar, "delta_bar": delta_bar, "Y0": Y0,
               "put_price": -Y0, "intrinsic": intrinsic,
               "time_value": -Y0 - intrinsic, "seed": seed}
    if diagnose:
        summary.update(_diagnose(equation, dim_h, path, graph_path, batch_size,
                                 N, T, upper_barrier, K, S0))
    with open(out_dir + "summary.json", "w") as p:
        json.dump(summary, p, indent=2)
    return summary


def _diagnose(equation, dim_h, path, graph_path, batch_size, N, T,
              upper_barrier, K, S0):
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
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    upper_np = upper_barrier(t, x).detach().cpu().numpy()
    S_np = np.exp(x_np)

    # ---- stock + Y trajectories
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for j in np.random.choice(batch_size, size=3, replace=False):
        axes[0].plot(t, S_np[j, 0, :], label=f"S (sample {j})")
        axes[1].plot(t, y_np[j, 0, :], label=f"Y (sample {j})")
        axes[1].plot(t, upper_np[j, 0, :], "--", alpha=0.5, label=f"-xi (sample {j})")
    axes[0].axhline(K, color="k", linestyle=":", alpha=0.6, label=f"K = {K}")
    axes[0].set_title(r"Stock price $S_t = e^{X_t}$")
    axes[0].set_xlabel("t"); axes[0].grid(True); axes[0].legend(fontsize=8)
    axes[1].set_title(r"$Y_t$ and upper obstacle $-\xi_t$")
    axes[1].set_xlabel("t"); axes[1].grid(True); axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(graph_path + "Y_trajectories.png")
    plt.close()

    # ---- obstacle adherence: max_i (Y + xi)^+ should be ~ 0
    viol = np.maximum(y_np[:, 0, :] - upper_np[:, 0, :], 0.0).max(axis=1)
    plt.figure(figsize=(8, 5))
    plt.hist(viol, bins=30, alpha=0.7)
    plt.xlabel(r"pathwise $\max_i (Y_{t_i} + \xi_{t_i})^+$")
    plt.ylabel("count")
    plt.title("Upper-obstacle violation (≈0 means constraint respected)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "obstacle_violation.png")
    plt.close()

    # ---- stopping times
    tol = 1e-3
    exit_idx = []
    for j in range(batch_size):
        diff = upper_np[j, 0, :-1] - y_np[j, 0, :-1]
        hits = diff < tol
        exit_idx.append(int(np.argmax(hits)) if hits.any() else N)
    exit_times = np.array(exit_idx) / N
    stopped_early = exit_times < (N - 1) / N

    plt.figure(figsize=(8, 5))
    if stopped_early.any():
        plt.hist(exit_times[stopped_early], bins=20, alpha=0.7)
    plt.xlabel(r"$\tau^*$"); plt.ylabel("count")
    plt.title("Early optimal stopping times")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "exit_times.png")
    plt.close()

    mean_tau = float(exit_times[stopped_early].mean()) if stopped_early.any() else None
    max_viol = float(viol.max())
    print(f"      max obstacle violation: {max_viol:.2e}")
    print(f"      fraction stopping early: {stopped_early.mean():.3f}")

    return {
        "max_obstacle_violation": max_viol,
        "frac_stopping_early": float(stopped_early.mean()),
        "mean_tau_star": mean_tau,
    }


if __name__ == "__main__":
    # default: entropic radius 0.1, worst-case discount = risk-free rate
    run(gamma_bar=0.1, delta_bar=0.05, out_dir="Example_5_4_put/")
