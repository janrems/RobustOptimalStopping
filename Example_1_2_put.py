"""Example_1_2_put.py

Geometric American put under paired ambiguity (Example 1.2 driver, Pham §5.3
forward setup). Same upper-reflected RBSDE machinery as Example_1_2.py; only
the forward dynamics, payoff and discount rate change.

Forward (log-price):
    X_t = log S_t,  dX_t = (r - sigma^2/2) dt + sigma dW_t,  X_0 = log(S_0).

Payoff (American put on S):
    xi_t = (K - S_t)^+ = (K - exp(X_t))^+.

Upper-reflected BSDE (Theorem 0.9):
    Y_t = -xi_T + int_t^T g(s, Y_s, Z_s) ds
          - int_t^T Z_s dW_s - (K_T - K_t),
    Y_t <= -xi_t,        tau* = inf{ s >= t : Y_s = -xi_s }.

Driver (Ex. 1.2 unconstrained):
    g(t, y, z) = (gamma_alpha/2) |z|^2 - beta_bar * y + (gamma_beta/2) y^2.

beta_bar = r, so the linear y-term implements the risk-free discount.
At gamma_alpha = gamma_beta = 0 this reduces to the classical American-put
RBSDE; Pham §5.3 reports u_0 ~ 0.0609 for ATM (K=S_0=1, sigma=0.2, r=0.05,
T=1). Our sign convention has -Y_0 corresponding to Pham's u_0.

Default here: slightly ITM (K=1.1, S_0=1) so the obstacle -xi_0 = -0.1 is
strictly below 0, keeping the problem non-trivial under any gamma.
"""

import torch
import numpy as np
import os
import json
import time

from RBSDE import fbsde, BSDEiter, Model, Result


new_folder = "Example_1_2_put/"
path = new_folder + "state_dicts/"
graph_path = new_folder + "Graphs/"
os.makedirs(path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mode = "Training"
#mode = "Testing"

# ---- Market parameters (Pham §5.3, slightly ITM)
S0 = 1.0
K = 1.1            # set K = 1.0 for Pham's ATM benchmark (-Y_0 ~ 0.0609 at gamma=0)
sigma_S = 0.2
r = 0.05

# ---- Ambiguity coefficients
gamma_alpha = 0.1
gamma_beta = 0.1
beta_bar = r       # discount rate = risk-free rate

LOWER_SENTINEL = -1e6


def b(t, x):
    return torch.full_like(x, r - 0.5 * sigma_S ** 2)


def sigma(t, x):
    return torch.full((x.size(0), dim_x, dim_x), sigma_S, device=x.device)


def xi(t, x):
    # xi_t = (K - S_t)^+ with S_t = exp(X_t)
    return torch.clamp(K - torch.exp(x), min=0.0)


def f(t, x, y, z):
    z_sq = (z ** 2).sum(dim=-1)
    return 0.5 * gamma_alpha * z_sq - beta_bar * y + 0.5 * gamma_beta * y ** 2


def g(x):
    # Y_T = -xi(T, X_T)
    return -xi(T, x)


def lower_barrier(t, x):
    # singly upper-reflected: lower projection is non-binding
    return torch.full_like(x, LOWER_SENTINEL)


def upper_barrier(t, x):
    # Y_t <= -xi(t, X_t)
    return -xi(t, x)


if mode == "Training":
    dim_x, dim_y, dim_d, dim_h, N, itr, batch_size = 1, 1, 1, 50, 50, 100, 2 ** 10
    multiplier = 5

    x0_value = float(np.log(S0))   # X_0 = log(S_0)
    T = 1.0

    run_parameters = {
        "dim_x": dim_x, "dim_y": dim_y, "dim_d": dim_d, "dim_h": dim_h,
        "N": N, "itr": itr, "batch_size": batch_size, "multiplier": multiplier,
        "x0_value": x0_value, "T": T,
        "S0": S0, "K": K, "sigma_S": sigma_S, "r": r,
        "gamma_alpha": gamma_alpha, "gamma_beta": gamma_beta, "beta_bar": beta_bar,
    }

    x_0 = torch.tensor(x0_value, dtype=torch.float32, device=device)

    equation = fbsde(x_0, b, sigma, f, g, lower_barrier, upper_barrier,
                     T, dim_x, dim_y, dim_d)

    with open(os.path.join(path, "params.json"), "w") as h:
        json.dump(run_parameters, h, indent=2)

    bsde_itr = BSDEiter(equation, dim_h)

    Y0 = []
    for i in range(1):
        print(f"iteration n {i}")
        start_time = time.time()
        loss, y = bsde_itr.train_whole(batch_size, N, path, itr, multiplier)
        end_time = time.time()
        Y0.append(float(y[0, 0]))
        print(f"Iteration {i} took {(end_time - start_time) / 60:.4f} minutes")
        print(f"  Y_0 ~ {Y0[-1]:.5f}    -Y_0 (put price) ~ {-Y0[-1]:.5f}")
        print(f"  intrinsic at t=0:  max(K - S_0, 0) = {max(K - S0, 0):.5f}")

    with open(path + "loss.json", "w") as p:
        json.dump(loss, p, indent=2)
    with open(path + "Y0.json", "w") as p:
        json.dump(Y0, p, indent=2)

else:
    import matplotlib.pyplot as plt

    with open(os.path.join(path, "params.json"), "r") as h:
        loaded = json.load(h)

    dim_x = loaded["dim_x"]; dim_y = loaded["dim_y"]; dim_d = loaded["dim_d"]
    dim_h = loaded["dim_h"]; N = loaded["N"]; itr = loaded["itr"]
    batch_size = loaded["batch_size"]; multiplier = loaded["multiplier"]
    x0_value = loaded["x0_value"]; T = loaded["T"]
    K_loaded = loaded.get("K", K)
    S0_loaded = loaded.get("S0", S0)

    x_0 = torch.tensor(x0_value, device=device)

    equation = fbsde(x_0, b, sigma, f, g, lower_barrier, upper_barrier,
                     T, dim_x, dim_y, dim_d)

    with open(path + "loss.json", "r") as fh:
        loss = json.load(fh)
    with open(path + "Y0.json", "r") as fh:
        Y0 = json.load(fh)

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

    # ---- Loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss[0])
    plt.title("Loss at terminal step (N-1)")
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "loss_N-1.png")
    plt.close()

    # ---- Stock price + Y trajectories
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_js = np.random.choice(batch_size, size=3, replace=False)
    for j in sample_js:
        axes[0].plot(t, S_np[j, 0, :], label=f"S (sample {j})")
        axes[1].plot(t, y_np[j, 0, :], label=f"Y (sample {j})")
        axes[1].plot(t, upper_np[j, 0, :], "--", alpha=0.5,
                     label=f"-xi (sample {j})")
    axes[0].axhline(K_loaded, color="k", linestyle=":", alpha=0.6,
                    label=f"K = {K_loaded}")
    axes[0].set_title(r"Stock price $S_t = e^{X_t}$")
    axes[0].set_xlabel("t"); axes[0].grid(True); axes[0].legend(fontsize=8)
    axes[1].set_title(r"$Y_t$ and upper obstacle $-\xi_t$")
    axes[1].set_xlabel("t"); axes[1].grid(True); axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(graph_path + "Y_trajectories.png")
    plt.close()

    # ---- Optimal stopping times tau* = first time (before T) Y_t hits -xi_t
    tol = 1e-3
    exit_idx = []
    for j in range(batch_size):
        diff = upper_np[j, 0, :] - y_np[j, 0, :]   # >= 0, 0 means hit
        # exclude the trivial terminal hit
        hits = diff[:-1] < tol
        exit_idx.append(int(np.argmax(hits)) if hits.any() else N)
    exit_idx = np.array(exit_idx)
    exit_times = exit_idx / N
    stopped_early = exit_times < (N - 1) / N

    plt.figure(figsize=(8, 5))
    if stopped_early.any():
        plt.hist(exit_times[stopped_early], bins=20, alpha=0.7)
    plt.xlabel(r"$\tau^*$"); plt.ylabel("count")
    plt.title("Early optimal stopping times (excludes terminal hit)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "exit_times.png")
    plt.close()

    intrinsic = max(K_loaded - S0_loaded, 0.0)
    print(f"Y_0 ~ {Y0[0]:.5f}")
    print(f"-Y_0 (put price under ambiguity) ~ {-Y0[0]:.5f}")
    print(f"intrinsic at t=0:  max(K - S_0, 0) = {intrinsic:.5f}")
    print(f"Fraction stopping early (before T):  {stopped_early.mean():.3f}")
    if stopped_early.any():
        print(f"Mean conditional tau*:  {exit_times[stopped_early].mean():.4f}")
