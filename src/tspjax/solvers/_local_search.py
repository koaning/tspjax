"""The shared, bounded-memory local-search core.

Every improver in :mod:`tspjax.solvers` is the same machine: repeatedly find the
best-improving move among a *windowed* set of candidates and apply it, until no
candidate improves the tour (or a step budget is hit). Only the move math differs
between algorithms (2-opt scores pairs, 3-opt scores triples), so that part is
supplied as two plain closures — there is no solver class hierarchy.

Two pieces live here:

* :func:`_windowed_best` — scans the candidate space in fixed-size *blocks* with a
  ``lax.fori_loop``, folding a running best. Peak memory is the size of one block's
  score grid, never the full ``O(n^k)`` candidate set. This is what keeps the search
  in memory for large tours (and what makes 3-opt feasible at all).
* :func:`_local_search` — the on-device best-improvement loop (``lax.while_loop``),
  so the whole optimisation is a single jit call with no per-step host sync.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import lax


def _windowed_best(score_block: Callable, n: int, inner: int, *, block: int):
    """Best move over a ``(outer, inner)`` candidate grid, scanned in blocks.

    ``score_block(outer_start)`` returns a ``(block, inner)`` array of candidate
    deltas (``inf`` for invalid candidates), where row ``r`` is outer index
    ``outer_start + r`` and column ``c`` is the algorithm's ``c``-th windowed
    offset. Returns ``(best_delta, best_outer, best_inner)``.
    """
    n_blocks = (n + block - 1) // block

    def step(b, carry):
        best_delta, best_outer, best_inner = carry
        outer_start = b * block
        deltas = score_block(outer_start).reshape(-1)  # (block * inner,)
        idx = jnp.argmin(deltas)
        d = deltas[idx]
        row = idx // inner
        col = idx % inner
        better = d < best_delta
        return (
            jnp.where(better, d, best_delta),
            jnp.where(better, outer_start + row, best_outer),
            jnp.where(better, col, best_inner),
        )

    init = (jnp.float32(jnp.inf), jnp.int32(0), jnp.int32(0))
    return lax.fori_loop(0, n_blocks, step, init)


def _local_search(D, tour, best_move: Callable, apply_move: Callable, *, max_steps: int):
    """Best-improvement loop: apply the best improving move until none remains.

    ``best_move(tour) -> (delta, move)`` finds the best candidate; ``apply_move(
    tour, move) -> tour`` applies it. The move is only applied while ``delta < 0``,
    so a final non-improving probe leaves the tour untouched. Runs entirely on
    device via ``lax.while_loop``.
    """

    def cond(carry):
        _tour, step, improved = carry
        return improved & (step < max_steps)

    def body(carry):
        cur, step, _ = carry
        delta, move = best_move(cur)
        improved = delta < 0
        nxt = lax.cond(improved, lambda: apply_move(cur, move), lambda: cur)
        return nxt, step + 1, improved

    init = (tour, jnp.int32(0), jnp.bool_(True))
    final_tour, _, _ = lax.while_loop(cond, body, init)
    return final_tour
