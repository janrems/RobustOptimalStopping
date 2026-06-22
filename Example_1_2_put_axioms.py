"""Example_1_2_put_axioms.py

Verify three dynamic risk-measure axioms on the trained network for the
z-only put example:

    Monotonicity            x_1 <= x_2  =>  u(0, x_1) <= u(0, x_2)
    Cash sub-additivity     Y(xi + c)   >=  Y(xi) - c     for c >= 0
    Convexity in xi         Y(xi^lambda) <= lambda Y(xi^1) + (1-lambda) Y(xi^2)

Base network is the trained z-only put at gamma_alpha = 1.0, replicate 0,
located at Example_1_2_put_sweep_zonly/run_02r0_ga1.0/. Auxiliary networks
(shifted obstacle, second strike, baskets at 10 lambda values) are trained
from scratch on the first run and cached in Example_1_2_put_axioms/aux_*/.

Single-file orchestrator: just run `python Example_1_2_put_axioms.py`.
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


# ============================================================
# Configuration
# ============================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reference to the base trained network
BASE_DIR = "Example_1_2_put_sweep_zonly/run_02r0_ga1.0/"

# Output
OUT_ROOT = "Example_1_2_put_axioms/"
RESULTS_DIR = OUT_ROOT + "results/"
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Problem parameters (must match the base run)
GAMMA_ALPHA = 1.0
GAMMA_BETA = 0.0
BETA_BAR = 0.0
K1 = 1.1       # base strike
K2 = 1.0       # second strike (for convexity)
S0 = 1.0
SIGMA_S = 0.2
R = 0.05
T = 1.0
X0_VALUE = float(np.log(S0))

# Solver hyperparameters (must match the base run)
DIM_X = DIM_Y = DIM_D = 1
DIM_H = 50
N = 50
ITR = 100
BATCH_SIZE = 2 ** 10
MULTIPLIER = 5
LOWER_SENTINEL = -1e6

# Test grids
C_VALUES = [0.01, 0.05, 0.10]
LAMBDA_VALUES = np.linspace(0, 1, 10)

# Reporting tolerance (numerical noise level below which we don't flag violations)
TOLERANCE = 1e-3
SEED = 107   # auxiliary trainings use this seed too (matches base seed)


# ============================================================
# Problem setup: forward dynamics, base obstacle, driver
# ============================================================

def b(t, x):
    return torch.full_like(x, R - 0.5 * SIGMA_S ** 2)


def sigma(t, x):
    return torch.full((x.size(0), DIM_X, DIM_X), SIGMA_S, device=x.device)


def base_xi(t, x):
    return torch.clamp(K1 - torch.exp(x), min=0.0)


def driver(t, x, y, z):
    z_sq = (z ** 2).sum(dim=-1)
    return 0.5 * GAMMA_ALPHA * z_sq - BETA_BAR * y + 0.5 * GAMMA_BETA * y ** 2


def make_equation(xi_func):
    """Build an fbsde with a custom obstacle. Forward and driver are fixed."""

    def lower(t, x):
        return torch.full_like(x, LOWER_SENTINEL)

    def upper(t, x):
        return -xi_func(t, x)

    def g_term(x):
        return -xi_func(T, x)

    x_0 = torch.tensor(X0_VALUE, dtype=torch.float32, device=device)
    return fbsde(x_0, b, sigma, driver, g_term, lower, upper,
                 T, DIM_X, DIM_Y, DIM_D)


# ---- Auxiliary obstacle factories ---------------------------------

def shifted_xi(c):
    def xi(t, x):
        return base_xi(t, x) + c
    return xi


def strike_xi(K):
    def xi(t, x):
        return torch.clamp(K - torch.exp(x), min=0.0)
    return xi


def basket_xi(K_low, K_high, lam):
    def xi(t, x):
        return (lam * torch.clamp(K_high - torch.exp(x), min=0.0)
                + (1.0 - lam) * torch.clamp(K_low - torch.exp(x), min=0.0))
    return xi


# ============================================================
# Training helper (caches networks under aux_*)
# ============================================================

def train_aux(name, xi_func):
    """Train an auxiliary network for the given obstacle. Cached by folder."""
    aux_dir = OUT_ROOT + f"aux_{name}/"
    sentinel = aux_dir + f"state_dict_{N-2}"
    if os.path.exists(sentinel):
        print(f"  [skip]  {name:<32}  (cached)")
        return aux_dir

    os.makedirs(aux_dir, exist_ok=True)
    print(f"  [train] {name:<32}", end="", flush=True)
    t0 = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    equation = make_equation(xi_func)
    bsde_itr = BSDEiter(equation, DIM_H)
    loss, y_train = bsde_itr.train_whole(BATCH_SIZE, N, aux_dir, ITR, MULTIPLIER)

    params = dict(
        name=name, gamma_alpha=GAMMA_ALPHA, gamma_beta=GAMMA_BETA, beta_bar=BETA_BAR,
        K1=K1, K2=K2, S0=S0, sigma_S=SIGMA_S, r=R, T=T,
        dim_x=DIM_X, dim_y=DIM_Y, dim_d=DIM_D, dim_h=DIM_H,
        N=N, itr=ITR, batch_size=BATCH_SIZE, multiplier=MULTIPLIER,
        seed=SEED, x0_value=X0_VALUE,
    )
    with open(aux_dir + "params.json", "w") as fh:
        json.dump(params, fh, indent=2)

    Y0_train = float(y_train[0, 0])
    with open(aux_dir + "Y0.json", "w") as fh:
        json.dump({"Y0_train": Y0_train}, fh, indent=2)

    dt = (time.time() - t0) / 60.0
    print(f"  Y0={Y0_train:+.4f}  ({dt:.1f} min)")
    return aux_dir


# ============================================================
# Evaluators
# ============================================================

def _project(y_raw, upper_vals):
    """Apply the singly upper-reflected projection (lower sentinel non-binding)."""
    lower = torch.full_like(y_raw, LOWER_SENTINEL)
    return torch.minimum(torch.maximum(y_raw, lower), upper_vals)


def evaluate_y0(state_dict_dir, xi_func):
    """Return the projected Y_0 at X_0 for the trained network at state_dict_dir."""
    equation = make_equation(xi_func)
    model = Model(equation, DIM_H)
    model.load_state_dict(
        torch.load(state_dict_dir + "state_dict_0", map_location="cpu"),
        strict=False,
    )
    model.eval()
    x_0 = torch.full((BATCH_SIZE, DIM_X), X0_VALUE, dtype=torch.float32)
    with torch.no_grad():
        y, _ = model(N, 0, x_0)
        upper = -xi_func(torch.tensor(0.0), x_0)
        y_proj = _project(y, upper)
    return float(y_proj[0, 0])


def simulate_paths(M, seed=None):
    """Simulate one batch of forward paths for the base SDE."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    eq = make_equation(base_xi)
    result = Result(Model(eq, DIM_H), eq)
    while True:
        W = result.gen_b_motion(M, N)
        x = result.gen_x(M, N, W)
        if not torch.isnan(x).any():
            return x


def evaluate_along_paths(state_dict_dir, xi_func, x):
    """Evaluate Y_t along given paths using the network at state_dict_dir."""
    M = x.shape[0]
    equation = make_equation(xi_func)
    model = Model(equation, DIM_H)
    result = Result(model, equation)
    y, _ = result.predict(N, M, x, state_dict_dir)
    return y.detach().cpu().numpy()


# ============================================================
# Test 1: Monotonicity
# ============================================================

def test_monotonicity(x_paths):
    """Scatter (X_t, Y_t) at interior time slices and check non-decreasing."""
    print("\n[Test 1] Monotonicity")

    y_paths = evaluate_along_paths(BASE_DIR, base_xi, x_paths)
    t_grid = np.linspace(0, T, N)
    x_np = x_paths.detach().cpu().numpy()

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    slices = [(N // 4, axes[0]), (N // 2, axes[1])]

    for t_idx, ax in slices:
        xs = x_np[:, 0, t_idx]
        ys = y_paths[:, 0, t_idx]
        order = np.argsort(xs)
        xs_sorted = xs[order]
        ys_sorted = ys[order]

        diffs = np.diff(ys_sorted)
        violations = int((diffs < -TOLERANCE).sum())
        max_drop = float(max(0.0, -diffs.min())) if len(diffs) else 0.0

        rows.append({
            "time_idx": int(t_idx),
            "time": float(t_grid[t_idx]),
            "n_samples": int(len(xs_sorted)),
            "n_violations": violations,
            "max_drop": max_drop,
        })

        ax.scatter(xs, ys, s=8, alpha=0.35, color="C0")
        ax.plot(xs_sorted, ys_sorted, color="red", alpha=0.3, lw=0.5)
        ax.set_xlabel(r"$X_t$  (log-price)")
        ax.set_ylabel(r"$Y_t$")
        ax.set_title(rf"$t = {t_grid[t_idx]:.2f}$   "
                     rf"violations: {violations}/{len(xs_sorted) - 1}   "
                     rf"max drop: {max_drop:.1e}")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR + "monotonicity.png")
    plt.close()

    with open(RESULTS_DIR + "monotonicity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return rows


# ============================================================
# Test 2: Cash sub-additivity
# ============================================================

def test_cash_subadditivity(aux_dirs):
    """For each c >= 0, check Y_0(xi+c) >= Y_0(xi) - c."""
    print("\n[Test 2] Cash sub-additivity")

    Y0_base = evaluate_y0(BASE_DIR, base_xi)

    rows = []
    for c in C_VALUES:
        Y0_shifted = evaluate_y0(aux_dirs[f"cash_c{c}"], shifted_xi(c))
        slack = Y0_shifted - (Y0_base - c)
        rows.append({
            "c": float(c),
            "Y0_base": Y0_base,
            "Y0_shifted": Y0_shifted,
            "lower_bound": Y0_base - c,
            "slack": slack,
            "satisfied": bool(slack >= -TOLERANCE),
        })

    # plot
    cs = [r["c"] for r in rows]
    Y0s = [r["Y0_shifted"] for r in rows]
    lower = [r["lower_bound"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cs, Y0s, marker="o", color="C0", label=r"$Y_0(\xi + c)$")
    ax.plot(cs, lower, marker="s", linestyle="--", color="C3",
            label=r"$Y_0(\xi) - c$   (sub-additive lower bound)")
    ax.set_xlabel(r"cash shift $c$")
    ax.set_ylabel(r"$Y_0$")
    ax.set_title(r"Cash sub-additivity:  $Y(\xi + c) \geq Y(\xi) - c$")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + "cash_subadditivity.png")
    plt.close()

    with open(RESULTS_DIR + "cash_subadditivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return rows


# ============================================================
# Test 3: Convexity
# ============================================================

def test_convexity(aux_dirs):
    """For each lambda, check Y_0(xi^lambda) <= lambda Y_0(xi^K1) + (1-lambda) Y_0(xi^K2)."""
    print("\n[Test 3] Convexity")

    Y0_K1 = evaluate_y0(BASE_DIR, strike_xi(K1))            # lambda = 1 endpoint
    Y0_K2 = evaluate_y0(aux_dirs["K2"], strike_xi(K2))      # lambda = 0 endpoint

    rows = []
    for lam in LAMBDA_VALUES:
        lam_f = float(lam)
        if lam_f == 0.0:
            Y0 = Y0_K2
        elif lam_f == 1.0:
            Y0 = Y0_K1
        else:
            key = f"basket_lam{lam_f:.4f}"
            Y0 = evaluate_y0(aux_dirs[key], basket_xi(K2, K1, lam_f))

        chord = lam_f * Y0_K1 + (1.0 - lam_f) * Y0_K2
        slack = chord - Y0   # >= 0 means convex (basket sits at or below chord)
        rows.append({
            "lambda": lam_f,
            "Y0_basket": Y0,
            "chord": chord,
            "slack": slack,
            "satisfied": bool(slack >= -TOLERANCE),
        })

    # plot
    lams = [r["lambda"] for r in rows]
    Y0s = [r["Y0_basket"] for r in rows]
    chords = [r["chord"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lams, Y0s, marker="o", color="C0",
            label=r"$Y_0(\xi^\lambda)$   (basket)")
    ax.plot(lams, chords, linestyle="--", color="C3",
            label=rf"$\lambda Y_0(\xi^{{K_1}}) + (1-\lambda) Y_0(\xi^{{K_2}})$   (chord)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$Y_0$")
    ax.set_title(
        rf"Convexity:  $Y_0(\xi^\lambda) \leq$ chord  "
        rf"($K_1 = {K1}$, $K_2 = {K2}$)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + "convexity.png")
    plt.close()

    with open(RESULTS_DIR + "convexity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return rows


# ============================================================
# Reporting
# ============================================================

def write_summary(mono, cash, conv, total_min):
    lines = []
    push = lines.append
    push("=" * 64)
    push("Risk-Measure Axiom Test  --  Summary")
    push(f"Base:       {BASE_DIR}")
    push(f"Output:     {RESULTS_DIR}")
    push(f"Tolerance:  {TOLERANCE}")
    push(f"Runtime:    {total_min:.2f} min")
    push("=" * 64)
    push("")

    push("[1] Monotonicity")
    for r in mono:
        ok = r["max_drop"] < TOLERANCE
        push(f"   t = {r['time']:.3f}   "
             f"violations {r['n_violations']:>4}/{r['n_samples'] - 1}   "
             f"max drop {r['max_drop']:.2e}   {'PASS' if ok else 'FAIL'}")
    push("")

    push("[2] Cash sub-additivity")
    for r in cash:
        ok = r["satisfied"]
        push(f"   c = {r['c']:.3f}   "
             f"Y0(xi+c) = {r['Y0_shifted']:+.4f}   "
             f"Y0(xi)-c = {r['lower_bound']:+.4f}   "
             f"slack {r['slack']:+.3e}   {'PASS' if ok else 'FAIL'}")
    push("")

    push("[3] Convexity")
    push(f"   {sum(r['satisfied'] for r in conv)}/{len(conv)} lambdas satisfied"
         f" (slack >= -{TOLERANCE})")
    for r in conv:
        ok = r["satisfied"]
        push(f"   lambda = {r['lambda']:.4f}   "
             f"Y0_basket = {r['Y0_basket']:+.4f}   "
             f"chord = {r['chord']:+.4f}   "
             f"slack {r['slack']:+.3e}   {'PASS' if ok else 'FAIL'}")
    push("")

    text = "\n".join(lines)
    print()
    print(text)
    with open(RESULTS_DIR + "summary.txt", "w") as fh:
        fh.write(text)

    config = {
        "BASE_DIR": BASE_DIR,
        "GAMMA_ALPHA": GAMMA_ALPHA, "GAMMA_BETA": GAMMA_BETA, "BETA_BAR": BETA_BAR,
        "K1": K1, "K2": K2, "S0": S0, "sigma_S": SIGMA_S, "r": R, "T": T,
        "N": N, "DIM_H": DIM_H, "BATCH_SIZE": BATCH_SIZE, "ITR": ITR,
        "MULTIPLIER": MULTIPLIER, "SEED": SEED,
        "C_VALUES": C_VALUES,
        "LAMBDA_VALUES": LAMBDA_VALUES.tolist(),
        "TOLERANCE": TOLERANCE,
    }
    with open(RESULTS_DIR + "config.json", "w") as fh:
        json.dump(config, fh, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()

    print(f"device: {device}")
    print(f"base:   {BASE_DIR}")
    print(f"out:    {OUT_ROOT}")

    # Sanity check: confirm base params agree with our config
    base_params_path = BASE_DIR + "params.json"
    if os.path.exists(base_params_path):
        with open(base_params_path) as fh:
            base_p = json.load(fh)
        for key, val in [("gamma_alpha", GAMMA_ALPHA), ("gamma_beta", GAMMA_BETA),
                          ("beta_bar", BETA_BAR), ("K", K1), ("S0", S0),
                          ("sigma_S", SIGMA_S), ("r", R), ("T", T),
                          ("N", N), ("dim_h", DIM_H)]:
            if key in base_p and base_p[key] != val:
                print(f"  WARNING: base param {key} = {base_p[key]} "
                      f"!= script setting {val}")

    print("\nTraining auxiliary networks (cached when present):")
    aux_dirs = {}

    # second-strike endpoint for convexity
    aux_dirs["K2"] = train_aux(f"K2_{K2:.2f}", strike_xi(K2))

    # cash-shifted networks
    for c in C_VALUES:
        aux_dirs[f"cash_c{c}"] = train_aux(f"cash_c{c}", shifted_xi(c))

    # baskets at interior lambdas
    for lam in LAMBDA_VALUES:
        lam_f = float(lam)
        if lam_f == 0.0 or lam_f == 1.0:
            continue
        key = f"basket_lam{lam_f:.4f}"
        aux_dirs[key] = train_aux(key, basket_xi(K2, K1, lam_f))

    # Shared simulated batch for the monotonicity scatter
    print("\nSimulating shared batch of paths...")
    x_paths = simulate_paths(BATCH_SIZE, seed=SEED + 1)

    # Tests
    mono = test_monotonicity(x_paths)
    cash = test_cash_subadditivity(aux_dirs)
    conv = test_convexity(aux_dirs)

    # Report
    total_min = (time.time() - t_start) / 60.0
    write_summary(mono, cash, conv, total_min)


if __name__ == "__main__":
    main()
