# CLAUDE.md

A map of this repository — where things live, how to run them, and the
conventions to respect. For the user-facing API tutorial, see `README.md`.

## What this is

`tspjax` — an offline-first, JAX-native TSPLIB / Traveling Salesman toolkit.
It bundles the TSPLIB instances (and their best-known tour lengths) in-package,
wraps them in a small `Problem` API whose cost functions are jax-jittable, and
ships space-filling-curve tour constructors plus a 2-opt local-search solver.
Notebook-first: the intended workflow is interactive in marimo.

## Directory map

```
.
├── src/tspjax/
│   ├── __init__.py          # public API surface + version
│   ├── problem.py           # Problem base + Euclidean/Matrix/Function variants
│   ├── loader.py            # load / loads / list_problems / best_known
│   ├── _tsplib.py           # TSPLIB file parser (pure Python, JAX-free)
│   ├── distances.py         # distance kernels w/ TSPLIB rounding
│   ├── construct.py         # hilbert/morton/moore tour constructors
│   ├── santa.py             # Kaggle Santa 2018 prime-path problem
│   ├── plot.py              # matplotlib tour plotting (JAX-free)
│   ├── solvers/
│   │   ├── __init__.py      # solver exports
│   │   ├── _local_search.py # shared best-improvement / windowing core
│   │   ├── two_opt.py       # windowed 2-opt
│   │   └── three_opt.py     # 3-opt
│   └── data/
│       ├── problems/tsp/    # 111 symmetric .tsp instances
│       ├── problems/atsp/   # 19 asymmetric .atsp instances
│       ├── solutions        # best-known lengths (symmetric)
│       ├── atsp_solutions   # best-known lengths (asymmetric)
│       └── santa/cities.csv.zip  # Santa 2018 coords (~198k cities, zipped)
├── tests/             # pytest suite, one test_*.py per module
├── notebooks/         # marimo notebooks — tsp_quickstart.py
├── Makefile           # install / test / notebooks / pypi targets
├── pyproject.toml     # metadata + deps (jax, numpy, matplotlib; dev: pytest, marimo)
├── conductor.json     # Conductor workspace setup/run scripts
├── uv.lock            # locked deps (uv)
└── README.md          # user-facing API tutorial
```

## Where to find things

| Responsibility | Location |
| --- | --- |
| Public API surface (`load`, `loads`, `list_problems`, `best_known`) | `src/tspjax/__init__.py` |
| Problem types (`Problem`, `EuclideanProblem`, `MatrixProblem`, `FunctionProblem`) | `src/tspjax/problem.py` |
| Instance loading + best-known lookup | `src/tspjax/loader.py` |
| TSPLIB file parser (pure Python, JAX-free) | `src/tspjax/_tsplib.py` |
| Distance kernels (EUC_2D, EUC_3D, GEO, ATT, CEIL_2D; TSPLIB rounding) | `src/tspjax/distances.py` |
| Tour constructors (`hilbert_tour`, `morton_tour`, `moore_tour`) | `src/tspjax/construct.py` |
| Solvers (`two_opt`, `three_opt` + shared `_local_search` core) | `src/tspjax/solvers/` |
| Kaggle Santa 2018 (prime-path cost) | `src/tspjax/santa.py` |
| Tour plotting (matplotlib, JAX-free) | `src/tspjax/plot.py` |
| Bundled data | `src/tspjax/data/` |

### Bundled data (`src/tspjax/data/`)

- `problems/tsp/*.tsp` — 111 symmetric TSPLIB instances
- `problems/atsp/*.atsp` — 19 asymmetric instances
- `solutions` — best-known lengths for symmetric instances
- `atsp_solutions` — best-known lengths for asymmetric instances
- `santa/cities.csv.zip` — Kaggle Santa 2018 coords (~198k cities), zipped

Everything is bundled so the package works fully offline.

### Tests (`tests/`)

- `test_problem.py` — `Problem` API and `tour_length` correctness
- `test_loader.py` — instance loading and best-known lookup
- `test_distances.py` — distance kernels vs TSPLIB spec
- `test_construct.py` — space-filling-curve constructors
- `test_solvers.py` — 2-opt improvement / convergence
- `test_santa.py` — Santa 2018 loading and prime-path cost

## Common commands

All via `uv` (see `Makefile`):

- `make install` — `uv sync --extra dev` (venv with dev deps + editable tspjax)
- `make test` — `uv run --extra dev pytest -q`
- `make notebooks` — `uv run --extra dev marimo edit --watch notebooks`
- `make pypi` — `uv build && uv publish`

## Conventions & architecture

- A **tour** is a `(n,)` int permutation of `0..n-1`, treated as a **closed
  cycle** (the last city returns to the first).
- `Problem.tour_length(tour)` is the single cost primitive — jax-jittable,
  returns a scalar. Everything else builds on it.
- **Module layering**: pure-Python parser (`_tsplib`) → JAX problems/distances
  (`problem`, `distances`) → high-level loader/construct/solvers.
- **Solvers take raw arrays and return raw arrays** (`distances`, `tour`) — they
  never touch `Problem` objects. They are jittable / vmap-able and loop
  on-device (`lax.while_loop` / `lax.fori_loop`); local search blocks the
  candidate grid to keep peak memory bounded.
- Free functions for stateless work (`construct.*`, `distances.*`, `plot.*`);
  methods on `Problem` only where they close over instance state.
- Offline-only: never reach for the network — all data is in `src/tspjax/data/`.
