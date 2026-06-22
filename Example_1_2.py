import torch
import numpy as np
import os
import json
import time

from RBSDE import fbsde, BSDEiter, Model, Result

# ============================================================
# Example 1.2 — unconstrained quadratic driver
# (paper: "Risk Measures with Optimal Stopping under Paired-Ambiguity")
#
# Upper-reflected BSDE (Thm 0.9):
#   Y_t = -xi_T + int_t^T g(s, Y_s, Z_s) ds
#         - int_t^T Z_s dW_s - (K_T - K_t),
#   Y_t <= -xi_t,        tau* = inf{ s >= t : Y_s = -xi_s }.
#
# Driver (Ex. 1.2, beta unconstrained):
#   g(t, y, z) = (gamma_alpha / 2) |z|^2  -  beta_bar_t * y  +  (gamma_beta / 2) y^2.
#
# Forward SDE:  dX_t = dW_t,   X_0 = 0.
# Payoff:       xi_t = X_t.
#
# The existing solver does double reflection (RBSDE.py:157). Setting
# the lower barrier to a large negative sentinel makes the lower
# projection non-binding, recovering the singly upper-reflected case.
# ============================================================

new_folder = "Example_1_2/"
path = new_folder + "state_dicts/"
graph_path = new_folder + "Graphs/"
os.makedirs(path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mode = "Training"
mode = "Testing"

# Ambiguity / driver constants — small for stable training
gamma_alpha = 0.1   # L^2 model-ambiguity radius
gamma_beta  = 0.1   # discount-rate ambiguity radius
beta_bar    = 0.05  # baseline discount rate (constant)

LOWER_SENTINEL = -1e6   # makes lower reflection non-binding


def b(t, x):
    return torch.zeros_like(x)


def sigma(t, x):
    return torch.ones(x.size(0), dim_x, dim_x, device=x.device)


def xi(t, x):
    # adapted obstacle process — here xi_t = X_t
    return x


def f(t, x, y, z):
    # Example 1.2 unconstrained driver g(t, y, z)
    # z: [batch, dim_y, dim_d]; sum over dim_d gives |z|^2 in dim_y
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

    x0_value = 0.0
    T = 1.0

    run_parameters = {
        "dim_x": dim_x, "dim_y": dim_y, "dim_d": dim_d, "dim_h": dim_h,
        "N": N, "itr": itr, "batch_size": batch_size, "multiplier": multiplier,
        "x0_value": x0_value, "T": T,
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

    # ---- Loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss[0])
    plt.title("Loss at terminal step (N-1)")
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "loss_N-1.png")
    plt.close()

    # ---- Sample trajectories: Y_t vs upper obstacle -xi_t
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_js = np.random.choice(batch_size, size=3, replace=False)
    for j in sample_js:
        ax.plot(t, y_np[j, 0, :], label=f"Y (sample {j})")
        ax.plot(t, upper_np[j, 0, :], linestyle="--", alpha=0.6,
                label=f"-xi (sample {j})")
    ax.set_xlabel("t")
    ax.set_title(r"$Y_t$ and upper obstacle $-\xi_t = -X_t$")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "Y_trajectories.png")
    plt.close()

    # ---- Optimal stopping times: tau* = first time Y_t hits -xi_t
    tol = 1e-3
    exit_idx = []
    for j in range(batch_size):
        diff = upper_np[j, 0, :] - y_np[j, 0, :]   # >= 0
        hits = diff < tol
        exit_idx.append(int(np.argmax(hits)) if hits.any() else N)
    exit_idx = np.array(exit_idx)
    exit_times = exit_idx / N

    plt.figure(figsize=(8, 5))
    if (exit_times < 1.0).any():
        plt.hist(exit_times[exit_times < 1.0], bins=20, alpha=0.7)
    plt.xlabel(r"$\tau^*$"); plt.ylabel("count")
    plt.title("Optimal stopping times")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path + "exit_times.png")
    plt.close()

    print(f"Y_0 ~ {Y0[0]:.4f}")
    print(f"Fraction stopping before T: {(exit_times < 1.0).mean():.3f}")
    if (exit_times < 1.0).any():
        print(f"Mean conditional tau*: {exit_times[exit_times < 1.0].mean():.4f}")
