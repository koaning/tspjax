"""Windowed 2-opt local search.

2-opt improves a tour by reversing a segment ``tour[i..j]`` — this removes the two
boundary edges and reconnects them the other way. In a *symmetric* problem the
interior edges of the segment just flip direction and cancel, so only the four
boundary edges change and a move's delta is an O(1) computation.

Which segments get scored is decided by a pluggable *candidate strategy* (see
:mod:`.candidates`): the default :func:`~tspjax.solvers.candidates.all_pairs` is a full
search, :func:`~tspjax.solvers.candidates.windowed` restricts to a positional window,
and :func:`~tspjax.solvers.candidates.nearest` restricts to spatial neighbours. The
shared core in :mod:`._local_search` folds the search in blocks so peak memory never
reaches the full ``(n, n)`` delta grid.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._local_search import _fold_best, _local_search
from .candidates import all_pairs, windowed

#: Anchors of the candidate grid scored at once. Caps peak memory at O(block * width).
_BLOCK = 256


def _apply_swap(tour, move):
    """Reverse ``tour[i..j]`` with pure index math (static shape, jit-safe)."""
    i, j = move
    k = jnp.arange(tour.shape[0])
    in_segment = (k >= i) & (k <= j)
    src = jnp.where(in_segment, i + j - k, k)
    return tour[src]


def two_opt(distances, tour=None, *, window=None, candidates=None, max_steps=10_000):
    """Best-improvement 2-opt to a local minimum (or ``max_steps``).

    ``distances`` is an ``(n, n)`` **symmetric** matrix; ``tour`` an optional ``(n,)``
    start permutation (defaults to ``jnp.arange(n)``). Returns the improved ``(n,)``
    ``int32`` tour.

    The candidate set is chosen by ``candidates``, a strategy from :mod:`.candidates`
    (default :func:`~tspjax.solvers.candidates.all_pairs`). The ``window`` keyword is a
    convenience alias for ``candidates=windowed(window)``; passing both is an error.
    Whatever the strategy, the search folds in blocks so peak memory stays bounded.

    The optimisation runs entirely on device, and the function is ``jax.vmap``-able
    over a leading batch axis on ``tour`` (``distances`` stays fixed). The
    segment-reversal move is only correct for symmetric distances; a host-side guard
    rejects asymmetric matrices.
    """
    D = jnp.asarray(distances, dtype=jnp.float32)
    n = int(D.shape[0])

    # Symmetry guard, host-side: segment-reversal 2-opt is only correct when
    # dist(a, b) == dist(b, a). This reads the matrix on the host, so it can't run
    # under jit (a traced `distances` has no concrete values) — by design, since the
    # whole solver is already a single on-device program; wrap the *call site* in jit
    # if you need to, not this function.
    host = np.asarray(D)
    if not np.allclose(host, host.T):
        raise ValueError(
            "two_opt requires a symmetric distance matrix; got an asymmetric one. "
            "Segment-reversal 2-opt is only correct when dist(a, b) == dist(b, a)."
        )

    if window is not None and candidates is not None:
        raise ValueError(
            "pass either `window` or `candidates`, not both; `window=w` is just an "
            "alias for `candidates=windowed(w)`."
        )

    if tour is None:
        tour = jnp.arange(n)
    tour = jnp.asarray(tour, dtype=jnp.int32)

    if n <= 3:
        return tour  # no non-trivial 2-opt move exists

    if candidates is None:
        candidates = all_pairs if window is None else windowed(window)
    spec = candidates(D, n)

    block = min(_BLOCK, spec.n_anchors)
    n_blocks = (spec.n_anchors + block - 1) // block

    def best_move(cur):
        anchors = spec.anchors(cur)

        def score_block(b):
            rows = b * block + jnp.arange(block, dtype=jnp.int32)
            valid_row = (rows < spec.n_anchors)[:, None]
            a = anchors[jnp.clip(rows, 0, spec.n_anchors - 1)][:, None]  # (block, 1)
            bp = spec.partners(cur, a[:, 0])  # (block, width) partner edge positions
            lo = jnp.minimum(a, bp)
            hi = jnp.maximum(a, bp)
            loc = jnp.clip(lo, 0, n - 1)
            hic = jnp.clip(hi, 0, n - 1)
            # Only the four boundary edges change: reversing cur[lo+1..hi] swaps
            # (cur[lo],cur[lo+1]) & (cur[hi],cur[hi+1]) for (cur[lo],cur[hi]) &
            # (cur[lo+1],cur[hi+1]).
            c_lo, c_lo1 = cur[loc], cur[(loc + 1) % n]
            c_hi, c_hi1 = cur[hic], cur[(hic + 1) % n]
            old = D[c_lo, c_lo1] + D[c_hi, c_hi1]
            new = D[c_lo, c_hi] + D[c_lo1, c_hi1]
            valid = (
                valid_row
                & (bp >= 0)
                & (bp <= n - 1)
                & (a != bp)
                & ~((lo == 0) & (hi == n - 1))  # full reversal is a no-op
            )
            delta = jnp.where(valid, new - old, jnp.inf).reshape(-1)
            i = (loc + 1).reshape(-1)
            j = hic.reshape(-1)
            return delta, (i, j)

        delta, (i, j) = _fold_best(score_block, n_blocks, 2)
        return delta, (i, j)

    return _local_search(D, tour, best_move, _apply_swap, max_steps=max_steps)
