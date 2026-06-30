"""Example_properties.py

Numerical check of Proposition 3.2 (Properties of phi) for all three examples:
  - Example_AmericanPut  (Case 1, BS, driver g(y) = -r y, affine)
  - Example_5_1_discount (Case 2, sup_beta {-beta y}, convex in y, not affine)
  - Example_5_4_put      (Case 3, -delta_bar y + (gamma_bar/2)|z|^2, not affine)

Properties checked (Prop 3.2):
  (a) decreasing monotonicity     phi(xi+m) <= phi(xi)            GATE
  (b) cash-subadditivity          phi(xi+m) >= phi(xi) - m        GATE
  (e) concavity (affine driver)   phi(mix) >= avg                 GATE for Case 1
                                                                  DIAG for Cases 2-3
  (f) convexity (Prop 3.2(f)):    not expected; only reported as a signed slack

So convexity is NEVER a gate; concavity is a gate only when the driver is
affine (Case 1). For the other cases we just report the signed gap, since
Prop 3.2 does not pin down a direction outside the affine regime.
"""

import json
import os

import torch

from Example_AmericanPut import run as run_BS
from Example_5_1_discount import run as run_51
from Example_5_4_put import run as run_54

OUT = "Example_properties/"
M_SHIFT = 0.2          # constant obstacle shift for monotonicity / cash-subadditivity
TOL = 0.02             # tolerance for NN approximation noise


def y0(run_fn, tag, xi_override, base_kw):
    s = run_fn(out_dir=f"{OUT}{tag}/", diagnose=False, xi_override=xi_override,
               **base_kw)
    return s["Y0"]


def check_example(name, run_fn, base_kw, obstacles, concavity_is_gate=False):
    print(f"\n=== {name} ===")
    yb = y0(run_fn, f"{name}_base", obstacles["base"], base_kw)
    ys = y0(run_fn, f"{name}_shift", obstacles["shift"], base_kw)
    y2 = y0(run_fn, f"{name}_conv2", obstacles["conv2"], base_kw)
    ym = y0(run_fn, f"{name}_mix", obstacles["mix"], base_kw)

    # (a) monotonicity:    Y0(xi+m) <= Y0(xi)
    mono_pass = ys <= yb + TOL
    # (b) cash-subadditivity:  Y0(xi+m) >= Y0(xi) - m
    cash_pass = ys >= yb - M_SHIFT - TOL
    # (e) concavity:       Y0(mix) >= avg                (slack = Y0(mix) - avg)
    avg = 0.5 * (yb + y2)
    conc_slack = ym - avg
    conc_pass = conc_slack >= -TOL

    res = {
        "Y0_base": yb, "Y0_shift": ys, "Y0_conv2": y2, "Y0_mix": ym,
        "avg_baseline_conv2": avg,
        "monotonicity": {
            "gate": True, "pass": bool(mono_pass),
            "margin": yb + TOL - ys,
        },
        "cash_subadditivity": {
            "gate": True, "pass": bool(cash_pass),
            "margin": ys - (yb - M_SHIFT),
        },
        "concavity": {
            "gate": bool(concavity_is_gate), "pass": bool(conc_pass),
            "signed_slack": conc_slack,
        },
    }

    print(f"  Y0  base={yb:+.4f}  shift={ys:+.4f}  conv2={y2:+.4f}  mix={ym:+.4f}")
    print(f"  [GATE] monotonicity      Y0(xi+m) <= Y0(xi):     "
          f"{'PASS' if mono_pass else 'FAIL'}  (margin {res['monotonicity']['margin']:+.4f})")
    print(f"  [GATE] cash-subadditivity Y0(xi+m) >= Y0(xi)-m:  "
          f"{'PASS' if cash_pass else 'FAIL'}  (margin {res['cash_subadditivity']['margin']:+.4f})")
    tag = "[GATE]" if concavity_is_gate else "[DIAG]"
    verdict = ("PASS" if conc_pass else "FAIL") if concavity_is_gate else \
              ("concave" if conc_slack >= 0 else "non-concave")
    print(f"  {tag}  concavity         Y0(mix) >= avg:         "
          f"{verdict}  (signed slack {conc_slack:+.4f})")
    return res


# ---- Obstacle bundles --------------------------------------------------------

def obstacles_BS(K=1.1, K2=0.9):
    """American put baseline + shift + second-strike + 50/50 basket."""
    put = lambda kk, x: torch.clamp(kk - torch.exp(x), min=0.0)
    return {
        "base":  lambda t, x: put(K, x),
        "shift": lambda t, x: put(K, x) + M_SHIFT,
        "conv2": lambda t, x: put(K2, x),
        "mix":   lambda t, x: 0.5 * (put(K, x) + put(K2, x)),
    }


def obstacles_51():
    """Sign-changing Brownian payoff + variants."""
    return {
        "base":  lambda t, x: x,
        "shift": lambda t, x: x + M_SHIFT,
        "conv2": lambda t, x: x ** 2 - 0.5,                  # curved, sign-changing
        "mix":   lambda t, x: 0.5 * (x + (x ** 2 - 0.5)),
    }


def obstacles_54(K=1.1, K2=0.9):
    put = lambda kk, x: torch.clamp(kk - torch.exp(x), min=0.0)
    return {
        "base":  lambda t, x: put(K, x),
        "shift": lambda t, x: put(K, x) + M_SHIFT,
        "conv2": lambda t, x: put(K2, x),
        "mix":   lambda t, x: 0.5 * (put(K, x) + put(K2, x)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    # Case 1 — BS American put. Driver -r y is affine, so by Prop 3.2(e)
    # concavity is a gate.
    report["BS"] = check_example(
        "BS", run_BS, {}, obstacles_BS(),
        concavity_is_gate=True,
    )

    # Case 2 — §5.1 bounded discount. Driver sup_beta {-beta y} is convex in y
    # (not affine). Concavity is diagnostic only.
    report["5.1"] = check_example(
        "5.1", run_51, {"beta_lo": 0.0, "beta_hi": 0.2}, obstacles_51(),
        concavity_is_gate=False,
    )

    # Case 3 — §5.4 entropic-discount put. Driver has |z|^2 term, not affine.
    # Concavity diagnostic only.
    report["5.4"] = check_example(
        "5.4", run_54, {"gamma_bar": 5.0, "delta_bar": 0.05}, obstacles_54(),
        concavity_is_gate=False,
    )

    with open(OUT + "report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    gate_fails = [(ex, prop)
                  for ex, r in report.items()
                  for prop in ("monotonicity", "cash_subadditivity", "concavity")
                  if r[prop]["gate"] and not r[prop]["pass"]]
    print("\n=== summary ===")
    if gate_fails:
        print("GATE FAILURES:", gate_fails)
    else:
        print("all gates passed across BS, 5.1, 5.4")


if __name__ == "__main__":
    main()
