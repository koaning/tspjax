"""The shared, bounded-memory local-search core.

Every improver in :mod:`tspjax.solvers` is the same machine: repeatedly find the
best-improving move among a *windowed* set of candidates and apply it, until no
candidate improves the tour (or a step budget is hit). Only the move math differs
between algorithms (2-opt scores pairs, 3-opt scores triples), so that part is
supplied as two plain closures — there is no solver class hierarchy.

Two pieces live here:

* :func:`_fold_best` — scans the candidate space in fixed-size *blocks* with a
  ``lax.fori_loop``, folding a running best. Peak memory is the size of one block's
  score grid, never the full ``O(n^k)`` candidate set. This is what keeps the search
  in memory for large tours (and what makes 3-opt feasible at all). Each block reports
  both its candidate deltas and the *move fields* (the actual tour positions a move
  acts on), so the winning move is read straight off the fold with no index decoding —
  this is what lets the candidate set be an arbitrary, strategy-supplied grid rather
  than a fixed ``(outer, offset)`` lattice.
* :func:`_local_search` — the on-device best-improvement loop (``lax.while_loop``),
  so the whole optimisation is a single jit call with no per-step host sync.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import lax


def _fold_best(score_block: Callable, n_blocks: int, n_fields: int):
    """Best candidate over a blocked score grid, carrying its move fields.

    ``score_block(b)`` returns ``(delta, fields)`` for block ``b``: ``delta`` is a
    flat ``(L,)`` array of candidate deltas (``inf`` for invalid candidates) and
    ``fields`` is a length-``n_fields`` tuple of flat ``(L,)`` int arrays giving the
    move components (e.g. ``(i, j)`` for 2-opt, ``(i, j, k, recon)`` for 3-opt) for
    each candidate. Returns ``(best_delta, best_fields)`` where ``best_fields`` is the
    field tuple of the global best candidate across all blocks.
    """

    def step(b, carry):
        best_delta, best_fields = carry
        delta, fields = score_block(b)
        idx = jnp.argmin(delta)
        d = delta[idx]
        better = d < best_delta
        new_fields = tuple(
            jnp.where(better, f[idx], bf) for f, bf in zip(fields, best_fields)
        )
        return jnp.where(better, d, best_delta), new_fields

    init = (jnp.float32(jnp.inf), tuple(jnp.int32(0) for _ in range(n_fields)))
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
