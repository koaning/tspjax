"""Load bundled TSPLIB instances into :class:`~tspjax.problem.Problem` objects.

The instance files and best-known tour lengths ship inside the package
(``tspjax/data/``), so loading works fully offline — the original Heidelberg
source being unreliable is exactly why this package exists.

Layout::

    data/problems/tsp/*.tsp      111 symmetric instances (mastqe/tsplib)
    data/problems/atsp/*.atsp     19 asymmetric instances (Heidelberg TSPLIB95)
    data/solutions                symmetric best-known lengths
    data/atsp_solutions           asymmetric best-known lengths
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from . import _tsplib
from .problem import EuclideanProblem, MatrixProblem, Problem

_DATA = resources.files("tspjax") / "data"
# (subdirectory, file extension) pairs holding TSPLIB instance files.
_PROBLEM_DIRS = (("tsp", ".tsp"), ("atsp", ".atsp"))


def _load_santa() -> Problem:
    """Builder for the one bundled non-TSPLIB instance (custom prime-path cost)."""
    from . import santa  # lazy: santa imports jax/numpy and is only needed on demand

    return santa.load()


# Bundled problems that aren't TSPLIB files — loaded by a builder, not parsed.
_SYNTHETIC = {"santa2018": _load_santa}


@lru_cache(maxsize=1)
def _best_known_table() -> dict[str, int]:
    """Merged ``name -> best-known length`` map across both solution files."""
    table: dict[str, int] = {}
    for fname in ("solutions", "atsp_solutions"):
        table.update(_tsplib.parse_solutions((_DATA / fname).read_text()))
    return table


@lru_cache(maxsize=1)
def _index() -> dict[str, object]:
    """Map every instance name to its bundled file resource."""
    out: dict[str, object] = {}
    for subdir, ext in _PROBLEM_DIRS:
        for path in (_DATA / "problems" / subdir).iterdir():
            if path.name.endswith(ext):
                out[path.name[: -len(ext)]] = path
    return out


@lru_cache(maxsize=1)
def list_problems() -> tuple[str, ...]:
    """Names of every bundled instance, sorted (TSPLIB + ``santa2018``)."""
    return tuple(sorted({*_index(), *_SYNTHETIC}))


def best_known(name: str) -> int | None:
    """Best-known (often optimal) tour length for ``name``, or ``None``.

    For most small symmetric instances this is the proven optimum; for larger
    and asymmetric instances it is the best length published by TSPLIB.
    """
    return _best_known_table().get(name)


def load(name: str) -> Problem:
    """Load any bundled instance by name (e.g. ``load("berlin52")``).

    Returns an :class:`EuclideanProblem` for coordinate-based edge-weight types,
    a :class:`MatrixProblem` for ``EXPLICIT`` instances (including the asymmetric
    ``.atsp`` ones), or the :class:`FunctionProblem` for ``"santa2018"``. To load
    a Santa-style problem from your own CSV, use :func:`tspjax.santa.load`.
    """
    if name in _SYNTHETIC:
        return _SYNTHETIC[name]()
    path = _index().get(name)
    if path is None:
        raise FileNotFoundError(
            f"No bundled problem named {name!r}. See tspjax.list_problems()."
        )
    return loads(path.read_text(), best_known=best_known(name))


def loads(text: str, *, best_known: int | None = None) -> Problem:
    """Parse TSPLIB ``text`` into a :class:`Problem` (for custom/external files)."""
    inst = _tsplib.parse(text)
    ewt = inst.edge_weight_type.upper()

    if ewt == "EXPLICIT":
        if inst.matrix is None:
            raise ValueError(f"{inst.name!r} is EXPLICIT but has no EDGE_WEIGHT_SECTION")
        return MatrixProblem(
            inst.name,
            inst.matrix,
            best_known=best_known,
            comment=inst.comment,
            coords=inst.display_coords,  # may be None
        )

    if inst.coords is None:
        raise ValueError(f"{inst.name!r} ({ewt}) has no NODE_COORD_SECTION to load")
    return EuclideanProblem(
        inst.name,
        inst.coords,
        edge_weight_type=ewt,
        best_known=best_known,
        comment=inst.comment,
    )
