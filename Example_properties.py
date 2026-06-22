"""Example_properties.py

Risk-measure property checks for §5.1 and §5.4, reusing the trained solver via
the xi_override hook on each example's run().

The provable properties are for the fixed-horizon rho. The stopped value
phi = essinf_tau rho keeps monotonicity and cash-subadditivity (an infimum
preserves both) but NOT convexity (an inf of convex functions need not be
convex). So:

    GATES (must pass):  monotonicity, cash-subadditivity
    DIAGNOSTIC:         convexity   (a failure on the stopped value is expected
                                     math, not a solver bug)

For each example we train the base obstacle and three variants and compare Y_0:
    monotonicity        xi + m >= xi   =>  Y_0(xi+m) <= Y_0(xi)
    cash-subadditivity  Y_0(xi+m) >= Y_0(xi) - m            (m >= 0 constant)
    convexity           Y_0(mix) <= 0.5 (Y_0(xi1) + Y_0(xi2))
"""

import json
import os

import torch

from Example_5_1_discount import run as run_51
from Example_5_4_put import run as run_54

OUT = "Example_properties/"
M_SHIFT = 0.2          # constant obstacle shift for monotonicity / cash-subadditivity
TOL = 0.02             # tolerance for NN approximation noise


def y0(run_fn, tag, xi_override, base_kw):
    s = run_fn(out_dir=f"{OUT}{tag}/", diagnose=False, xi_override=xi_override,
               **base_kw)
    return s["Y0"]


def check_example(name, run_fn, base_kw, obstacles):
    print(f"\n=== {name} ===")
    yb = y0(run_fn, f"{name}_base", obstacles["base"], base_kw)
    ys = y0(run_fn, f"{name}_shift", obstacles["shift"], base_kw)
    y2 = y0(run_fn, f"{name}_conv2", obstacles["conv2"], base_kw)
    ym = y0(run_fn, f"{name}_mix", obstacles["mix"], base_kw)

    mono_pass = ys <= yb + TOL
    cash_pass = ys >= yb - M_SHIFT - TOL
    conv_rhs = 0.5 * (yb + y2)
    conv_pass = ym <= conv_rhs + TOL

    res = {
        "Y0_base": yb, "Y0_shift": ys, "Y0_conv2": y2, "Y0_mix": ym,
        "monotonicity": {"gate": True, "pass": bool(mono_pass),
                         "margin": yb + TOL - ys},
        "cash_subadditivity": {"gate": True, "pass": bool(cash_pass),
                               "margin": ys - (yb - M_SHIFT)},
        "convexity": {"gate": False, "pass": bool(conv_pass),
                      "margin": conv_rhs + TOL - ym},
    }
    print(f"  Y0  base={yb:+.4f}  shift={ys:+.4f}  conv2={y2:+.4f}  mix={ym:+.4f}")
    print(f"  [GATE] monotonicity      Y0(xi+m) <= Y0(xi):      "
          f"{'PASS' if mono_pass else 'FAIL'}  (margin {res['monotonicity']['margin']:+.4f})")
    print(f"  [GATE] cash-subadditivity Y0(xi+m) >= Y0(xi)-m:   "
          f"{'PASS' if cash_pass else 'FAIL'}  (margin {res['cash_subadditivity']['margin']:+.4f})")
    print(f"  [DIAG] convexity         Y0(mix) <= avg:          "
          f"{'pass' if conv_pass else 'fail (expected possible)'}  "
          f"(margin {res['convexity']['margin']:+.4f})")
    return res


def obstacles_51():
    return {
        "base": lambda t, x: x,
        "shift": lambda t, x: x + M_SHIFT,
        "conv2": lambda t, x: x ** 2 - 0.5,                  # curved, sign-changing
        "mix": lambda t, x: 0.5 * (x + (x ** 2 - 0.5)),
    }


def obstacles_54(K=1.1, K2=0.9):
    put = lambda kk, x: torch.clamp(kk - torch.exp(x), min=0.0)
    return {
        "base": lambda t, x: put(K, x),
        "shift": lambda t, x: put(K, x) + M_SHIFT,
        "conv2": lambda t, x: put(K2, x),                    # second strike
        "mix": lambda t, x: 0.5 * (put(K, x) + put(K2, x)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    report["5.1"] = check_example(
        "5.1", run_51, {"beta_lo": 0.0, "beta_hi": 0.1}, obstacles_51())
    report["5.4"] = check_example(
        "5.4", run_54, {"gamma_bar": 0.1, "delta_bar": 0.05}, obstacles_54())

    with open(OUT + "report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    gates = [(ex, prop) for ex, r in report.items()
             for prop in ("monotonicity", "cash_subadditivity")
             if not r[prop]["pass"]]
    print("\n=== summary ===")
    if gates:
        print("GATE FAILURES:", gates)
    else:
        print("all gates passed (monotonicity, cash-subadditivity) for 5.1 and 5.4")


if __name__ == "__main__":
    main()
