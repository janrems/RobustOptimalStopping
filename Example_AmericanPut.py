"""Example_AmericanPut.py

Pure American put under Black-Scholes (Case 1, no ambiguity).

Forward (log-price):  dX_t = (r - sigma^2/2) dt + sigma dW_t,  X_0 = log S_0.
Payoff:               xi_t = (K - S_t)^+ = (K - e^{X_t})^+.
Driver:               g(t, y, z) = -r * y    (linear, no sup, no z-term).

This is the classical American put reflected BSDE: upper-reflected, no model
ambiguity, no discount ambiguity. Serves as the baseline before the ambiguous
cases §5.1 and §5.4.
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


def american_put_binomial(S0, K, sigma, r, T, N=2000):
    """Cox-Ross-Rubinstein American put price; reference benchmark."""
    dt = T / N
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    # asset prices at maturity
    j = np.arange(N + 1)
    S = S0 * (u ** (N - j)) * (d ** j)
    V = np.maximum(K - S, 0.0)
    for step in range(N - 1, -1, -1):
        j = np.arange(step + 1)
        S = S0 * (u ** (step - j)) * (d ** j)
        cont = disc * (p * V[:step + 1] + (1.0 - p) * V[1:step + 2])
        exercise = np.maximum(K - S, 0.0)
        V = np.maximum(cont, exercise)
    return float(V[0])


def run(out_dir, S0=1.0, K=1.1, sigma_S=0.2, r=0.05,
        seed=0, N=50, itr=300, dim_h=50, batch_size=2 ** 10, multiplier=10,
        T=1.0, diagnose=True, xi_override=None):
    """Train the pure American put; return summary dict.

    xi_override(t, x): optional obstacle replacing the default put payoff
    (used by the property checks to feed shifted / alternate-strike obstacles).
    """
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

    if xi_override is None:
        def xi(t, x):
            return torch.clamp(K - torch.exp(x), min=0.0)
    else:
        xi = xi_override

    def f(t, x, y, z):
        # g(t, y, z) = -r y    (BS, no ambiguity, linear in y)
        return -r * y

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
                  seed=seed)
    with open(os.path.join(path, "params.json"), "w") as h:
        json.dump(params, h, indent=2)

    start = time.time()
    loss, y = BSDEiter(equation, dim_h).train_whole(batch_size, N, path, itr, multiplier)
    mins = (time.time() - start) / 60.0
    Y0 = float(y[0, 0])
    intrinsic = max(K - S0, 0.0)
    ref_price = american_put_binomial(S0, K, sigma_S, r, T, N=2000)
    print(f"[BS] r={r:.3f} sigma={sigma_S:.3f}  -Y_0={-Y0:.5f}  "
          f"intrinsic={intrinsic:.5f}  binomial_ref={ref_price:.5f}  "
          f"(gap {(-Y0) - ref_price:+.5f})  ({mins:.1f} min)")

    with open(path + "loss.json", "w") as p:
        json.dump(loss, p, indent=2)
    with open(path + "Y0.json", "w") as p:
        json.dump({"Y0": Y0}, p, indent=2)

    summary = {"Y0": Y0, "put_price": -Y0, "intrinsic": intrinsic,
               "time_value": -Y0 - intrinsic, "binomial_ref": ref_price,
               "ref_gap": (-Y0) - ref_price, "seed": seed}
    if diagnose:
        summary.update(_diagnose(equation, dim_h, path, graph_path, batch_size,
                                 N, T, upper_barrier, K, S0, loss))
    with open(out_dir + "summary.json", "w") as p:
        json.dump(summary, p, indent=2)
    return summary


def _diagnose(equation, dim_h, path, graph_path, batch_size, N, T,
              upper_barrier, K, S0, loss):
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
    z_np = z.detach().cpu().numpy()
    upper_np = upper_barrier(t, x).detach().cpu().numpy()

    # ---- Y vs obstacle (left) + control process Z (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_js = np.random.choice(batch_size, size=3, replace=False)
    t_z = t[:-1]
    for j in sample_js:
        axes[0].plot(t, y_np[j, 0, :], label=f"$Y$ (sample {j})")
        axes[0].plot(t, upper_np[j, 0, :], "--", alpha=0.5,
                     label=rf"$-\xi$ (sample {j})")
        axes[1].plot(t_z, z_np[j, 0, 0, :-1], label=f"$Z$ (sample {j})")
    axes[0].set_title(r"$Y_t$ and upper obstacle $-\xi_t$")
    axes[0].set_xlabel("t"); axes[0].grid(True); axes[0].legend(fontsize=7)
    axes[1].set_title(r"Control process $Z_t$")
    axes[1].set_xlabel("t"); axes[1].grid(True); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(graph_path + "Y_trajectories.png")
    plt.close()

    # ---- stopping times histogram
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
        plt.axvline(exit_times[stopped_early].mean(), color="red",
                    linestyle="--", lw=1.2,
                    label=f"mean {exit_times[stopped_early].mean():.3f}")
        plt.legend()
    plt.xlabel(r"$\tau^*$"); plt.ylabel("count")
    plt.title(rf"Distribution of optimal stopping times "
              rf"(early-stop frac {stopped_early.mean():.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "stopping_times.png")
    plt.close()

    # ---- training loss at terminal step (n = N-2)
    plt.figure(figsize=(8, 5))
    if loss and len(loss) > 0:
        plt.plot(loss[0], lw=0.7, color="C0")
        tail = loss[0][max(0, int(0.9 * len(loss[0]))):]
        if tail:
            plt.axhline(float(np.mean(tail)), color="red", linestyle="--",
                        lw=1.0, alpha=0.6, label=f"final 10% mean")
            plt.legend()
    plt.yscale("log")
    plt.xlabel("iteration"); plt.ylabel("loss")
    plt.title("Training loss at terminal step (n = N-2)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(graph_path + "loss.png")
    plt.close()

    mean_tau = float(exit_times[stopped_early].mean()) if stopped_early.any() else None
    print(f"      fraction stopping early: {stopped_early.mean():.3f}")
    if mean_tau is not None:
        print(f"      mean tau*: {mean_tau:.4f}")

    return {
        "frac_stopping_early": float(stopped_early.mean()),
        "mean_tau_star": mean_tau,
    }


if __name__ == "__main__":
    # baseline: standard American put, no ambiguity
    run(out_dir="Example_AmericanPut/")
