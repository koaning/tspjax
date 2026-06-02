# tspjax

Load [TSPLIB](https://en.wikipedia.org/wiki/TSPLIB) traveling-salesman problems
and play with jax algorithms that solve them on a GPU.

The 111 symmetric instances (from [mastqe/tsplib](https://github.com/mastqe/tsplib))
and the 19 asymmetric ATSP instances (from the
[Heidelberg TSPLIB95](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
collection), together with their best-known tour lengths, are **bundled in the
package**, so everything works offline — which is the whole point, since the
original Heidelberg source is unreliable.

## Install

```bash
uv add tspjax          # or: pip install tspjax
```

## Quick start

```python
import jax, jax.numpy as jnp
import tspjax

tspjax.list_problems()          # ('a280', 'ali535', 'att48', ...)  — bundled instances

p = tspjax.load("berlin52")     # -> EuclideanProblem
p.dimension                     # 52
p.best_known                    # 7542  (proven optimum here; best-known for harder instances)
p.distances                     # (52, 52) jnp.ndarray, computed lazily + cached
p.tour_length(jnp.arange(52))   # length of a tour (a permutation of 0..n-1)
p.plot()                        # scatter the cities (renders inline in marimo)
```

Everything is built to drop straight into a **marimo** notebook: arrays are
jax, plots return a matplotlib `Axes`, and the loader needs no network.

## The `Problem` API

A `Problem` is a plain object you pass around host-side — not a jax pytree. It
holds jax arrays and exposes a jax-jittable `tour_length`. Pull the array out
and hand it to a jitted/vmapped solver:

```python
fast_len = jax.jit(p.tour_length)
lengths  = jax.vmap(fast_len)(batch_of_tours)   # (B,) on the GPU
```

`tour_length(tour)` is the one cost primitive every problem provides;
`distances` is an optional `(n, n)` fast path for matrix-aware solvers.

Three concrete flavours, all subclassing `Problem`:

| class | cost source | symmetry |
|-------|-------------|----------|
| `EuclideanProblem` | coordinates + a distance kernel (`EUC_2D`, `GEO`, `ATT`, `CEIL_2D`) | symmetric |
| `MatrixProblem` | an explicit `(n, n)` matrix | symmetric **or** asymmetric |
| `FunctionProblem` | any jax-compatible callable | your call |

```python
# asymmetric, from your own matrix
tspjax.MatrixProblem("mine", my_nxn_matrix)

# a fully custom cost
tspjax.FunctionProblem("mine", n, tour_cost=my_fn)
tspjax.FunctionProblem.from_pairwise("mine", n, lambda i, j: ...)
```

## Kaggle Santa 2018 (prime-path cost)

The [Traveling Santa 2018](https://www.kaggle.com/competitions/traveling-santa-2018-prime-paths)
instance doesn't fit a distance matrix: every 10th step is 10% longer *unless
the city being left has a prime CityId*, so the cost is position-dependent —
exactly what `FunctionProblem` is for. The ~198k-city coordinate file is bundled
(zipped), so it loads offline through the **same entry point** as everything else:

```python
p = tspjax.load("santa2018")           # -> FunctionProblem (n=197769)
p.tour_length(jnp.arange(p.dimension)) # score a path (must start at city 0)
```

The `tspjax.santa` module is the toolkit for what `load()` can't cover — your
own data and the reusable cost function:

```python
from tspjax import santa
santa.load("my_cities.csv.zip")        # load a Santa-style problem from your own CSV
cost = santa.prime_path_cost(coords)   # the raw jax cost, reusable on any coords
```

A tour is a permutation of `0..n-1` starting at city 0; it's scored as a closed
cycle, so the final edge returns to the North Pole automatically. For scoring
that matches the Kaggle leaderboard exactly, enable 64-bit jax
(`jax.config.update("jax_enable_x64", True)`) before loading. *(The coordinate
file is Kaggle competition data, bundled here for offline reproducibility.)*

## Solvers

`tspjax.solvers` is intentionally empty in v1 — build your jax algorithms
interactively against `problem.distances`, then promote the keepers. Solver
functions should take raw arrays (never a `Problem`) so they stay `jit`/`vmap`
friendly.

## Data

- `tspjax/data/problems/tsp/*.tsp` — 111 symmetric instances
- `tspjax/data/problems/atsp/*.atsp` — 19 asymmetric instances (load as `MatrixProblem`, `is_symmetric=False`)
- `tspjax/data/solutions`, `tspjax/data/atsp_solutions` — best-known lengths

Supported edge-weight types: `EUC_2D`, `EUC_3D`, `GEO`, `ATT`, `CEIL_2D`, and
`EXPLICIT` (`FULL_MATRIX`, `UPPER_ROW`, `UPPER_DIAG_ROW`, `LOWER_DIAG_ROW`).
