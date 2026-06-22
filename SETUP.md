# Python setup

This project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management.
The same convention is the standard for all projects on this machine.

## Run this project

```bash
uv sync          # build .venv from pyproject.toml + uv.lock (installs the pinned Python too)
uv run python Example_1_2.py
```

`uv run` executes inside the project venv — no manual `activate` needed. Lint and tests:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

## How it is wired

- **Python** pinned in `.python-version` (3.12); uv downloads it, the system Python is untouched.
- **Dependencies** declared in `pyproject.toml`, locked exactly in `uv.lock` (commit both).
- **`.venv/`** is local and gitignored — never committed, rebuilt by `uv sync`.
- **torch is CPU-only.** This machine has no GPU, so `pyproject.toml` points torch at the
  PyTorch CPU index. That keeps the wheel small and the install reproducible:

  ```toml
  [tool.uv.sources]
  torch = [{ index = "pytorch-cpu" }]

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```

## Start a new project with this standard

```bash
mkdir myproj && cd myproj
uv init --bare --python 3.12      # creates pyproject.toml only
uv add numpy matplotlib           # runtime deps
uv add --dev ruff pytest          # dev deps
git add -f .python-version        # global gitignore excludes it; force-add to pin the version
```

For an ML project, copy the torch CPU index block above into `pyproject.toml` before
`uv add torch`. Commit `pyproject.toml`, `uv.lock`, and `.python-version`.
