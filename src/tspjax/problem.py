"""The :class:`Problem` base class and its three concrete flavours.

A ``Problem`` is a plain Python object you pass around host-side — it is *not* a
registered jax pytree. It simply holds jax arrays (``coords``, ``distances``)
and exposes a jax-jittable :meth:`Problem.tour_length`. The intended pattern is
to pull the array out and feed it to a jitted solver::

    p = tspjax.load("berlin52")
    fast_len = jax.jit(p.tour_length)        # closes over p.distances
    length = fast_len(some_tour)

Subclasses:

* :class:`EuclideanProblem` — built from coordinates + a distance kernel.
* :class:`MatrixProblem`    — built from an explicit ``(n, n)`` matrix
  (symmetric *or* asymmetric).
* :class:`FunctionProblem`  — cost defined by an arbitrary callable.
"""

from __future__ import annotations

import abc
from functools import cached_property
from typing import Callable

import jax.numpy as jnp
import numpy as np

from . import distances as _distances
from .plot import plot_tour as _plot_tour


class Problem(abc.ABC):
    """Base class for a Traveling Salesman instance.

    Solvers are written against :meth:`tour_length`, the one cost primitive that
    every problem provides. :attr:`distances` is an optional fast path that
    matrix-aware solvers reach for when an ``(n, n)`` matrix is available.
    """

    name: str
    dimension: int
    best_known: int | None
    is_symmetric: bool
    comment: str = ""
    #: Coordinates for plotting, ``(n, 2)`` or ``None``. May be present even on
    #: matrix problems (via the file's DISPLAY_DATA_SECTION).
    coords: jnp.ndarray | None = None

    @property
    def distances(self) -> jnp.ndarray | None:
        """The ``(n, n)`` distance matrix, or ``None`` if the cost is matrix-free."""
        return None

    @abc.abstractmethod
    def tour_length(self, tour) -> jnp.ndarray:
        """Length of a tour given as a ``(n,)`` permutation of ``0..n-1``.

        The tour is treated as a closed cycle (the last city returns to the
        first). Written in jax, so ``jax.jit``/``jax.vmap`` it freely.
        """

    def plot(self, tour=None, ax=None, **kwargs):
        """Plot the cities (and ``tour`` if given). Requires :attr:`coords`."""
        if self.coords is None:
            raise ValueError(
                f"{self.name!r} has no coordinates to plot "
                "(explicit-matrix instance without DISPLAY_DATA)."
            )
        return _plot_tour(np.asarray(self.coords), tour=tour, ax=ax, title=self.name, **kwargs)

    def __repr__(self) -> str:
        opt = "?" if self.best_known is None else self.best_known
        kind = "sym" if self.is_symmetric else "asym"
        return (
            f"<{type(self).__name__} {self.name!r} n={self.dimension} "
            f"{kind} best_known={opt}>"
        )


def _matrix_tour_length(D: jnp.ndarray, tour) -> jnp.ndarray:
    """Sum the edge weights of a closed tour from a distance matrix."""
    tour = jnp.asarray(tour)
    nxt = jnp.roll(tour, -1)
    return jnp.sum(D[tour, nxt])


class EuclideanProblem(Problem):
    """A coordinate-based, symmetric instance (EUC_2D, GEO, ATT, CEIL_2D, ...)."""

    def __init__(
        self,
        name: str,
        coords,
        edge_weight_type: str = "EUC_2D",
        *,
        best_known: int | None = None,
        comment: str = "",
    ) -> None:
        self.name = name
        self.coords = jnp.asarray(coords, dtype=jnp.float32)
        self.edge_weight_type = edge_weight_type
        self.best_known = best_known
        self.comment = comment
        self.is_symmetric = True
        self.dimension = int(self.coords.shape[0])

    @cached_property
    def distances(self) -> jnp.ndarray:
        """Lazily computed and cached ``(n, n)`` distance matrix."""
        return _distances.distance_matrix(self.coords, self.edge_weight_type)

    def tour_length(self, tour) -> jnp.ndarray:
        return _matrix_tour_length(self.distances, tour)


class MatrixProblem(Problem):
    """An instance defined by an explicit ``(n, n)`` matrix (symmetric or not)."""

    def __init__(
        self,
        name: str,
        matrix,
        *,
        best_known: int | None = None,
        comment: str = "",
        coords=None,
    ) -> None:
        self.name = name
        self._matrix = jnp.asarray(matrix, dtype=jnp.float32)
        if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError(f"matrix must be square (n, n); got {self._matrix.shape}")
        self.best_known = best_known
        self.comment = comment
        self.coords = None if coords is None else jnp.asarray(coords, dtype=jnp.float32)
        self.dimension = int(self._matrix.shape[0])
        # Decide symmetry once, host-side, with numpy (avoids a traced bool).
        host = np.asarray(self._matrix)
        self.is_symmetric = bool(np.allclose(host, host.T))

    @property
    def distances(self) -> jnp.ndarray:
        return self._matrix

    def tour_length(self, tour) -> jnp.ndarray:
        return _matrix_tour_length(self._matrix, tour)


class FunctionProblem(Problem):
    """An instance whose tour cost is an arbitrary jax-compatible callable.

    Construct directly from a tour-cost function::

        FunctionProblem("custom", n, tour_cost=lambda tour: ...)

    or from a pairwise distance function via :meth:`from_pairwise`::

        FunctionProblem.from_pairwise("custom", n, lambda i, j: ...)
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        *,
        tour_cost: Callable,
        best_known: int | None = None,
        comment: str = "",
        is_symmetric: bool = False,
        coords=None,
    ) -> None:
        self.name = name
        self.dimension = int(dimension)
        self._tour_cost = tour_cost
        self.best_known = best_known
        self.comment = comment
        self.is_symmetric = is_symmetric
        self.coords = None if coords is None else jnp.asarray(coords, dtype=jnp.float32)

    @classmethod
    def from_pairwise(
        cls,
        name: str,
        dimension: int,
        dist_fn: Callable,
        *,
        best_known: int | None = None,
        comment: str = "",
        is_symmetric: bool = False,
        coords=None,
    ) -> "FunctionProblem":
        """Build a problem whose cost sums ``dist_fn(i, j)`` over tour edges."""

        def tour_cost(tour):
            tour = jnp.asarray(tour)
            nxt = jnp.roll(tour, -1)
            from jax import vmap

            return jnp.sum(vmap(dist_fn)(tour, nxt))

        return cls(
            name,
            dimension,
            tour_cost=tour_cost,
            best_known=best_known,
            comment=comment,
            is_symmetric=is_symmetric,
            coords=coords,
        )

    def tour_length(self, tour) -> jnp.ndarray:
        return self._tour_cost(jnp.asarray(tour))
