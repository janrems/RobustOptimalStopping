"""Example_5_1_discount.py

Paper §5.1 — bounded discount-rate ambiguity.

Risk measure: the discount rate is ambiguous in a band [beta_lo, beta_hi];
the probability measure is fixed. The reflected-BSDE driver is

    g(t, y, z) = sup_{beta in [beta_lo, beta_hi]} { -beta y }
               = max(-beta_lo * y, -beta_hi * y).

No z-term: pure discount ambiguity. Both band endpoints act only if Y changes
sign, so we use a BOUNDED SIGN-CHANGING collar obstacle

    xi_t = Phi_c(X_t) = c * tanh(X_t / c),    dX_t = dW_t,

with c = 1. The collar is bounded (|xi| <= c), which the convergence theory
requires, while still sign-changing. Then beta_hi discounts the liability side
(y < 0) and beta_lo the asset side (y >= 0); that asymmetry is the
cash-subadditivity §5.1 is about. A nonnegative payoff would pin Y <= 0 and
collapse the band to beta_hi (see §5.4 / the put).

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


def run(beta_lo, beta_hi, out_dir, seed=0, N=50, itr=300, dim_h=50,
        batch_size=2 ** 10, multiplier=10, T=1.0, c=1.0, x0_value=0.0,
        diagnose=True, xi_override=None):
    """Train §5.1 for a discount band [beta_lo, beta_hi]; return summary dict.

    c: collar level of the default obstacle xi_t = c * tanh(X_t / c).
    xi_override(t, x): optional obstacle replacing the default collar
    (used by the property checks to feed shifted / alternate obstacles).
    """
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

    if xi_override is None:
        def xi(t, x):
            # bounded sign-changing collar obstacle xi_t = c * tanh(X_t / c)
            return c * torch.tanh(x / c)
    else:
        xi = xi_override

    def law(t):
        # X_t = x_0 + W_t
        return x0_value, float(np.sqrt(max(t, 0.0)))

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
                  x0_value=x0_value, T=T, c=c, beta_lo=beta_lo, beta_hi=beta_hi,
                  seed=seed)
    with open(os.path.join(path, "params.json"), "w") as h:
        json.dump(params, h, indent=2)

    start = time.time()
    loss, y = bsde_train(equation, dim_h, batch_size, N, path, itr, multiplier, law)
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
                                 upper_barrier, law, beta_lo, beta_hi, loss))
    with open(out_dir + "summary.json", "w") as p:
        json.dump(summary, p, indent=2)
    return summary


def bsde_train(equation, dim_h, batch_size, N, path, itr, multiplier, law=None):
    return BSDEiter(equation, dim_h).train_whole(batch_size, N, path, itr, multiplier, law)


def _diagnose(equation, dim_h, path, graph_path, batch_size, N, T,
              upper_barrier, law, beta_lo, beta_hi, loss):
    model = Model(equation, dim_h)
    model.eval()
    result = Result(model, equation)

    flag = True
    while flag:
        W = result.gen_b_motion(batch_size, N)
        x = result.gen_x(batch_size, N, W)
        flag = torch.isnan(x).any()

    y, z = result.predict(N, batch_size, x, path, law)

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
        axes[0].plot(t, upper_np[j, 0, :], "--", alpha=0.6,
                     label=rf"$-\xi$ (sample {j})")
        axes[1].plot(t_z, z_np[j, 0, 0, :-1], label=f"$Z$ (sample {j})")
    axes[0].axhline(0.0, color="k", lw=0.8, alpha=0.6)
    axes[0].set_title(r"$Y_t$ and upper obstacle $-\xi_t = -c\,\tanh(X_t/c)$")
    axes[0].set_xlabel("t"); axes[0].grid(True); axes[0].legend(fontsize=8)
    axes[1].set_title(r"Control process $Z_t$")
    axes[1].set_xlabel("t"); axes[1].grid(True); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(graph_path + "Y_trajectories.png")
    plt.close()

    # both branches active iff Y visits both signs at interior times
    interior = y_np[:, 0, 1:-1]
    frac_asset_overall = float((interior >= 0).mean())

    # ---- stopping times histogram
    exit_idx = []
    for j in range(batch_size):
        diff = upper_np[j, 0, :-1] - y_np[j, 0, :-1]
        hits = diff <= 0.0
        exit_idx.append(int(np.argmax(hits)) if hits.any() else N)
    exit_times = np.array(exit_idx) / N
    stopped_early = exit_times < (N - 1) / N

    plt.figure(figsize=(8, 5))
    if stopped_early.any():
        plt.hist(exit_times[stopped_early], bins=np.linspace(0.0, 1.0, 26), alpha=0.7)
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
                        lw=1.0, alpha=0.6, label="final 10% mean")
            plt.legend()
    plt.yscale("log")
    plt.xlabel("iteration"); plt.ylabel("loss")
    plt.title("Training loss at terminal step (n = N-2)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(graph_path + "loss.png")
    plt.close()

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
