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

It isn't on PyPI — install straight from GitHub:

```bash
uv add "git+https://github.com/koaning/tspjax.git"     # or: pip install "git+https://github.com/koaning/tspjax.git"
```

Pin a tag or commit by appending `@<ref>`, e.g.
`git+https://github.com/koaning/tsplib.git@main`.

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

`tspjax.solvers` ships two local-search improvers: `two_opt` and `three_opt`.
Both take **raw arrays** — an `(n, n)` distance matrix and an optional `(n,)`
start tour — and return the improved tour, never touching a `Problem`, so they
stay `jit`/`vmap` friendly. Each runs entirely on device (the search loop is a
`lax.while_loop`) and scores candidates in a positional `window`, which keeps
peak memory bounded and — with a curve-ordered start (see below) — doubles as a
spatial neighbourhood.

```python
from tspjax.construct import hilbert_tour
from tspjax.solvers import two_opt, three_opt

p = tspjax.load("berlin52")
tour = hilbert_tour(p.coords)              # good starting tour
tour = two_opt(p.distances, tour, window=20)
tour = three_opt(p.distances, tour, window=10)
p.tour_length(tour)
```

## Perturbations

A local-search solver stops at a *local* optimum. To keep searching you *kick*
the tour and re-optimise — that's iterated local search. `tspjax.perturb`
provides the three classic kicks: `double_bridge`, `random_reversal`, and
`random_shuffle`. Each takes a `(tour, key, *, ...)` (a `jax.random` key, never a
`Problem`), returns a `(n,)` int32 permutation, and is `jit`/`vmap`-friendly.

```python
import jax
from tspjax.construct import hilbert_tour
from tspjax.perturb import double_bridge
from tspjax.solvers import two_opt

p = tspjax.load("berlin52")
tour = two_opt(p.distances, hilbert_tour(p.coords))   # local optimum
kicked = double_bridge(tour, jax.random.PRNGKey(0))   # escape it
tour = two_opt(p.distances, kicked)                   # re-optimise; keep if better
```

`random_reversal` and `random_shuffle` take a `max_len=` to cap the kick
strength.

## Iterated local search

`iterated_local_search` wires the kick → improve → accept loop together: it
improves once, then repeatedly perturbs and re-improves, keeping any candidate
that beats the current tour and tracking the best one ever seen. You compose the
improver yourself — `improve` is any `tour -> tour` callable — so you decide the
schedule (two_opt only, two_opt then three_opt, different windows, …):

```python
import jax
from tspjax.construct import hilbert_tour
from tspjax.perturb import double_bridge
from tspjax.solvers import two_opt, three_opt, iterated_local_search

p = tspjax.load("berlin52")
D = p.distances

# the improver schedule is yours: two_opt for a bit, then a 3-opt polish
improve = lambda t: three_opt(D, two_opt(D, t, window=20), window=10)

best, best_len, history = iterated_local_search(
    D, hilbert_tour(p.coords), improve, jax.random.PRNGKey(0),
    perturb=double_bridge,   # the kick (defaults to double_bridge)
    steps=100,
)
```

It returns `(best_tour, best_length, history)`. `history` is a `(steps + 1,)`
array of lengths — `history[0]` is the initial local optimum, the rest are each
iteration's candidate — so you can plot the search exploring;
`jnp.minimum.accumulate(history)` is the best-so-far convergence curve. It's a
host-level driver (it loops in Python and calls the on-device improvers each
step), which is the right altitude for a metaheuristic.

## Data

- `tspjax/data/problems/tsp/*.tsp` — 111 symmetric instances
- `tspjax/data/problems/atsp/*.atsp` — 19 asymmetric instances (load as `MatrixProblem`, `is_symmetric=False`)
- `tspjax/data/solutions`, `tspjax/data/atsp_solutions` — best-known lengths

Supported edge-weight types: `EUC_2D`, `EUC_3D`, `GEO`, `ATT`, `CEIL_2D`, and
`EXPLICIT` (`FULL_MATRIX`, `UPPER_ROW`, `UPPER_DIAG_ROW`, `LOWER_DIAG_ROW`).
