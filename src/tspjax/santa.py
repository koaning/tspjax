"""Kaggle "Traveling Santa 2018 — Prime Paths" instance.

This problem does not fit a static distance matrix: the cost of an edge depends
on *where it sits in the tour*. Concretely (per the competition scoring), the
path starts and ends at ``CityId 0`` and the length is the sum of Euclidean edge
distances, except every 10th step is 10% longer **unless the city being left on
that step has a prime CityId**. That position-dependence is exactly what
:class:`~tspjax.problem.FunctionProblem` exists for.

The ~198k-row ``cities.csv`` ships with the package (zipped), so ``load()``
works offline like the TSPLIB instances. You can also point :func:`load` at your
own downloaded file (``.csv`` or ``.csv.zip``).

A tour here is a permutation of ``0..n-1`` that **starts at city 0**. The closed
cycle that :meth:`Problem.tour_length` walks then ends with the edge back to
city 0 — matching the competition's "append the North Pole at the end" path.

Note on precision: coordinates are taken as-is. For scoring that matches the
Kaggle leaderboard bit-for-bit, enable 64-bit jax
(``jax.config.update("jax_enable_x64", True)``) before loading; otherwise the
default float32 accumulates rounding over ~200k edges.
"""

from __future__ import annotations

import io
import zipfile
from importlib import resources
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .problem import FunctionProblem

#: The bundled, zipped Kaggle coordinate file used by ``load()`` when no path is given.
_BUNDLED = resources.files("tspjax") / "data" / "santa" / "cities.csv.zip"

#: Multiplier applied to a penalised 10th-step edge.
_PENALTY = 1.1


def sieve(n: int) -> np.ndarray:
    """Boolean primality array for city ids ``0..n-1`` (``0`` and ``1`` are not prime)."""
    is_prime = np.ones(n, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return is_prime


def prime_path_cost(coords) -> Callable:
    """Build the position-dependent Santa cost for a set of city ``coords``.

    Returns a jax-jittable ``tour_cost(tour) -> scalar``. ``tour`` is a ``(n,)``
    permutation of ``0..n-1`` starting at city 0; it is scored as a closed cycle
    (the final edge returns to ``tour[0]``).
    """
    coords = jnp.asarray(coords)
    n = int(coords.shape[0])
    is_prime = jnp.asarray(sieve(n))
    # Edge at position p (tour[p] -> tour[p+1]) is step number p + 1.
    is_tenth_step = (jnp.arange(1, n + 1) % 10) == 0

    def tour_cost(tour) -> jnp.ndarray:
        tour = jnp.asarray(tour)
        nxt = jnp.roll(tour, -1)
        edge = jnp.sqrt(jnp.sum((coords[tour] - coords[nxt]) ** 2, axis=-1))
        # Penalise a 10th step only when the city being left (tour[p]) isn't prime.
        penalised = is_tenth_step & ~is_prime[tour]
        return jnp.sum(edge * jnp.where(penalised, _PENALTY, 1.0))

    return tour_cost


def _parse_csv(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``CityId,X,Y`` text into ``(ids, coords)``."""
    rows = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1)
    return rows[:, 0].astype(int), rows[:, 1:3]


def _read_zip(source) -> tuple[np.ndarray, np.ndarray]:
    """Read the single ``.csv`` member of a zip (path or file-like)."""
    with zipfile.ZipFile(source) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        return _parse_csv(zf.read(name).decode())


def _read_cities(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``CityId,X,Y`` file (plain ``.csv`` or single-member ``.zip``)."""
    path = Path(path)
    if path.suffix == ".zip":
        return _read_zip(path)
    return _parse_csv(path.read_text())


def load(path=None) -> FunctionProblem:
    """Load the Santa 2018 instance.

    With no argument, reads the coordinate file bundled with the package. Pass a
    ``path`` (``.csv`` or ``.csv.zip``) to use your own downloaded copy.

    Returns a :class:`~tspjax.problem.FunctionProblem` carrying the prime-path
    cost and the coordinates (so ``.plot(tour)`` works).
    """
    if path is None:
        ids, coords = _read_zip(io.BytesIO(_BUNDLED.read_bytes()))
    else:
        ids, coords = _read_cities(path)
    # Index so row i corresponds to CityId i (the file is already in order, but
    # don't rely on it).
    if not np.array_equal(ids, np.arange(len(ids))):
        coords = coords[np.argsort(ids)]
    return FunctionProblem(
        "santa2018",
        len(coords),
        tour_cost=prime_path_cost(coords),
        comment="Kaggle Traveling Santa 2018 - Prime Paths (start/end at CityId 0)",
        is_symmetric=False,
        coords=coords,
    )
