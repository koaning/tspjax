"""Windowed 2-opt local search.

2-opt improves a tour by reversing a segment ``tour[i..j]`` — this removes the two
boundary edges and reconnects them the other way. In a *symmetric* problem the
interior edges of the segment just flip direction and cancel, so only the four
boundary edges change and a move's delta is an O(1) computation.

The candidate set is restricted to a positional *window*: only segments whose
length is at most ``window`` are scored each step. With a curve-ordered start tour
(see :mod:`tspjax.construct`) a positional window doubles as a spatial neighbourhood,
which is where most improving moves live. The shared core in
:mod:`._local_search` folds the search in blocks so peak memory never reaches the
full ``(n, n)`` delta grid.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._local_search import _local_search, _windowed_best

#: Rows of the candidate grid scored at once. Caps peak memory at O(block * window).
_BLOCK = 256


def _apply_swap(tour, move):
    """Reverse ``tour[i..j]`` with pure index math (static shape, jit-safe)."""
    i, j = move
    k = jnp.arange(tour.shape[0])
    in_segment = (k >= i) & (k <= j)
    src = jnp.where(in_segment, i + j - k, k)
    return tour[src]


def two_opt(distances, tour=None, *, window=None, max_steps=10_000):
    """Best-improvement 2-opt to a local minimum (or ``max_steps``).

    ``distances`` is an ``(n, n)`` **symmetric** matrix; ``tour`` an optional
    ``(n,)`` start permutation (defaults to ``jnp.arange(n)``). Each step scores
    only segments of length ``<= window`` (``None`` -> full coverage) and folds the
    search in blocks, so peak memory stays bounded regardless of ``n``. Returns the
    improved ``(n,)`` ``int32`` tour.

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

    if tour is None:
        tour = jnp.arange(n)
    tour = jnp.asarray(tour, dtype=jnp.int32)

    window = (n - 1) if window is None else min(int(window), n - 1)
    if n <= 3 or window < 1:
        return tour  # no non-trivial 2-opt move exists

    block = min(_BLOCK, n)
    offsets = jnp.arange(1, window + 1, dtype=jnp.int32)  # segment lengths j - i

    def best_move(cur):
        def score_block(outer_start):
            i = (outer_start + jnp.arange(block, dtype=jnp.int32))[:, None]  # (block,1)
            j = i + offsets[None, :]                                        # (block,window)
            valid = (i > 0) & (j <= n - 1)
            # Clip gather indices into range; invalid entries are masked to inf below.
            pi = jnp.clip(i - 1, 0, n - 1)
            ci = jnp.clip(i, 0, n - 1)
            cj = jnp.clip(j, 0, n - 1)
            prev_i, city_i = cur[pi], cur[ci]
            city_j, next_j = cur[cj], cur[(cj + 1) % n]
            old = D[prev_i, city_i] + D[city_j, next_j]
            new = D[prev_i, city_j] + D[city_i, next_j]
            return jnp.where(valid, new - old, jnp.inf)

        delta, outer, inner = _windowed_best(score_block, n, window, block=block)
        i = outer
        j = i + 1 + inner  # offsets start at 1, so column `inner` is length inner+1
        return delta, (i, j)

    return _local_search(D, tour, best_move, _apply_swap, max_steps=max_steps)
