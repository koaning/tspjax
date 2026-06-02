"""Solver namespace.

Deliberately empty for the first cut — the loader, ``Problem`` hierarchy, and
plotting are the v1 surface. Fill this in interactively (e.g. from a marimo
notebook) with jax functions that operate on raw arrays, then promote the ones
you like to live here.

Conventions for anything added here:

* Take a ``(n, n)`` distance matrix (``problem.distances``) and/or a ``(n,)``
  tour, never a ``Problem`` object — so functions stay ``jax.jit``-able and
  ``jax.vmap``-able over batches of tours.
* Return jax arrays.
"""

from __future__ import annotations

from .candidates import all_pairs, longest_edge, nearest, windowed
from .iterated import iterated_local_search
from .three_opt import three_opt
from .two_opt import two_opt

__all__ = [
    "iterated_local_search",
    "three_opt",
    "two_opt",
    "all_pairs",
    "windowed",
    "nearest",
    "longest_edge",
]
