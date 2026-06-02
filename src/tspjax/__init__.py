"""tspjax — load TSPLIB problems and solve them with jax on the GPU.

A bundled, offline copy of the symmetric TSPLIB instances (and their best-known
tour lengths), wrapped in a small ``Problem`` API whose cost functions are
jax-jittable.

Quick start::

    import tspjax

    tspjax.list_problems()          # -> ('a280', 'ali535', ...)
    p = tspjax.load("berlin52")     # -> EuclideanProblem
    p.dimension                     # 52
    p.best_known                    # 7542  (proven optimum here; best-known for harder instances)
    p.distances                     # (52, 52) jnp.ndarray
    p.tour_length(jnp.arange(52))   # length of the identity tour
    p.plot()                        # scatter the cities
"""

from __future__ import annotations

from . import santa, solvers
from .loader import best_known, list_problems, load, loads
from .problem import EuclideanProblem, FunctionProblem, MatrixProblem, Problem

__all__ = [
    "load",
    "loads",
    "list_problems",
    "best_known",
    "Problem",
    "EuclideanProblem",
    "MatrixProblem",
    "FunctionProblem",
    "santa",
    "solvers",
]

__version__ = "0.1.0"
