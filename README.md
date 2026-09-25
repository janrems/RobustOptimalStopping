# Robust Optimal Stopping

Code for the numerical experiments in

> *Risk Measures under Paired-Ambiguity: A Deep Learning Reflected BSDE Framework*
> <https://arxiv.org/abs/2609.23768>

A reflected deep backward dynamic programming scheme for upper reflected BSDEs
with drivers of quadratic growth, applied to optimal stopping under dynamic risk
measures with simultaneous ambiguity in the probability measure and the discount
rate.

## Layout

| File | Paper section | What it does |
|---|---|---|
| `rbsde.py` | 5 | The scheme: networks, backward training loop, evaluation |
| `american_put.py` | 7.1 | American put under Black–Scholes, checked against a binomial reference |
| `discount_ambiguity.py` | 7.2 | Bounded discount-rate ambiguity, collared sign-changing obstacle |
| `entropic_put.py` | 7.3 | Geometric American put under entropic-discount ambiguity |
| `entropic_sweep.py` | 7.3 | Sweep over the entropic radius, reuses `entropic_put.run` |
| `properties.py` | 7.4 | Numerical check of monotonicity, cash-subadditivity and concavity |

## Running

Set up with [uv](https://docs.astral.sh/uv/) (see `SETUP.md`), then run any
example directly:

```
uv run python american_put.py
uv run python discount_ambiguity.py
uv run python entropic_put.py
uv run python properties.py
```

Each writes trained weights, a summary and diagnostic figures under its own
output directory. Those directories are not tracked.
