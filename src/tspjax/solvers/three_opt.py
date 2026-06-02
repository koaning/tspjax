"""Windowed 3-opt local search.

3-opt removes three edges and reconnects the three resulting segments in one of the
seven non-identity ways. With the tour split into a fixed part and two reorderable
segments ``B`` and ``C``, three of the seven reconnections are plain segment
reversals (each equivalent to a single 2-opt move) and the other four genuinely swap
the order of ``B`` and ``C`` — moves no 2-opt can make. Scoring all seven makes this a
full 3-opt: it subsumes 2-opt and converges to a true 3-opt local optimum.

Like 2-opt, the candidate set is restricted to a positional *window*: only segments
``B`` and ``C`` of length ``<= window`` are scored each step. With a curve-ordered
start tour (see :mod:`tspjax.construct`) that positional window doubles as a spatial
neighbourhood. The shared core in :mod:`._local_search` folds the search in blocks so
peak memory is ``O(block * window^2 * 7)``, never the full ``O(n^3)`` candidate grid —
this is what makes 3-opt feasible at all. Per-step work is ``O(n * window^2)``, so pass
a small ``window`` for large ``n``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._local_search import _local_search, _windowed_best

#: Rows of the candidate grid scored at once. Caps peak memory at O(block * window^2 * 7).
_BLOCK = 256

#: Number of non-identity reconnections of the two reorderable segments.
_N_RECON = 7

# Per-reconnection layout of the rebuilt middle ``[i, k-1]`` as two concatenated
# segments G1 then G2, each ``B`` or ``C`` and forward or reversed. Row ``r`` is
# reconnection ``r`` (see the module docstring / score table below). Used by
# ``_apply_three_opt``; the rows must match the edge sets scored in ``best_move``.
_FIRST_IS_C = jnp.array([0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)  # recon 3-6 put C first
_REV_FIRST = jnp.array([1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)
_REV_SECOND = jnp.array([0, 1, 1, 0, 0, 1, 1], dtype=jnp.int32)


def _apply_three_opt(tour, move):
    """Rebuild ``tour`` for a 3-opt move with pure index math (static shape, jit-safe).

    ``move`` is ``(i, j, k, recon)``. Positions outside the middle ``[i, k-1]`` map to
    themselves; the middle is two concatenated segments (G1 then G2), each ``B`` =
    ``tour[i..j-1]`` or ``C`` = ``tour[j..k-1]``, forward or reversed, selected from
    the ``_FIRST_IS_C`` / ``_REV_FIRST`` / ``_REV_SECOND`` lookup tables by ``recon``.
    """
    i, j, k, recon = move
    n = tour.shape[0]

    len_b = j - i
    len_c = k - j
    first_is_c = _FIRST_IS_C[recon]
    rev_first = _REV_FIRST[recon]
    rev_second = _REV_SECOND[recon]

    len_first = jnp.where(first_is_c, len_c, len_b)
    len_second = jnp.where(first_is_c, len_b, len_c)
    start_first = jnp.where(first_is_c, j, i)
    start_second = jnp.where(first_is_c, i, j)

    p = jnp.arange(n)
    in_middle = (p >= i) & (p < k)
    local = p - i
    in_g1 = local < len_first

    # Index within the owning segment, then mirror it when that segment is reversed.
    t1 = local
    src_g1 = start_first + jnp.where(rev_first, len_first - 1 - t1, t1)
    t2 = local - len_first
    src_g2 = start_second + jnp.where(rev_second, len_second - 1 - t2, t2)

    src_middle = jnp.where(in_g1, src_g1, src_g2)
    src = jnp.where(in_middle, src_middle, p)
    return tour[src]


def three_opt(distances, tour=None, *, window=None, max_steps=10_000):
    """Best-improvement 3-opt to a local minimum (or ``max_steps``).

    ``distances`` is an ``(n, n)`` **symmetric** matrix; ``tour`` an optional ``(n,)``
    start permutation (defaults to ``jnp.arange(n)``). Each step scores only segments
    of length ``<= window`` (``None`` -> full coverage) and folds the search in blocks,
    so peak memory stays bounded regardless of ``n``. Returns the improved ``(n,)``
    ``int32`` tour.

    All seven non-identity reconnections are scored, so this is a full 3-opt: it
    subsumes 2-opt and reaches a true 3-opt local optimum. Per-step work is
    ``O(n * window^2)`` — pass a small ``window`` for large ``n``.

    The optimisation runs entirely on device, and the function is ``jax.vmap``-able
    over a leading batch axis on ``tour`` (``distances`` stays fixed). The
    segment-reversal reconnections are only correct for symmetric distances; a
    host-side guard rejects asymmetric matrices.
    """
    D = jnp.asarray(distances, dtype=jnp.float32)
    n = int(D.shape[0])

    # Symmetry guard, host-side: the reversal reconnections are only correct when
    # dist(a, b) == dist(b, a). This reads the matrix on the host, so it can't run
    # under jit (a traced `distances` has no concrete values) — by design, since the
    # whole solver is already a single on-device program; wrap the *call site* in jit
    # if you need to, not this function.
    host = np.asarray(D)
    if not np.allclose(host, host.T):
        raise ValueError(
            "three_opt requires a symmetric distance matrix; got an asymmetric one. "
            "Segment-reversal reconnections are only correct when "
            "dist(a, b) == dist(b, a)."
        )

    if tour is None:
        tour = jnp.arange(n)
    tour = jnp.asarray(tour, dtype=jnp.int32)

    window = (n - 1) if window is None else min(int(window), n - 1)
    if n < 5 or window < 1:
        return tour  # no non-trivial 3-opt move exists

    block = min(_BLOCK, n)
    offsets = jnp.arange(window, dtype=jnp.int32)  # segment-length offsets d1, d2

    def best_move(cur):
        def score_block(outer_start):
            # i indexes segment B's start; (d1, d2) choose the lengths of B and C.
            i = (outer_start + jnp.arange(block, dtype=jnp.int32))[:, None, None]  # (b,1,1)
            d1 = offsets[None, :, None]  # (1,window,1)
            d2 = offsets[None, None, :]  # (1,1,window)
            j = i + 1 + d1
            k = j + 1 + d2
            valid = (i >= 1) & (j <= n - 1) & (k <= n)

            # Clip gather indices into range; invalid entries are masked to inf below.
            a = cur[jnp.clip(i - 1, 0, n - 1)]
            b = cur[jnp.clip(i, 0, n - 1)]
            c = cur[jnp.clip(j - 1, 0, n - 1)]
            d = cur[jnp.clip(j, 0, n - 1)]
            e = cur[jnp.clip(k - 1, 0, n - 1)]
            f = cur[jnp.clip(k % n, 0, n - 1)]

            Dab, Dcd, Def = D[a, b], D[c, d], D[e, f]
            old = Dab + Dcd + Def
            # One delta per reconnection; stacked on a trailing axis of size 7.
            deltas = jnp.stack(
                [
                    D[a, c] + D[b, d] + Def,  # 0: B' C
                    Dab + D[c, e] + D[d, f],  # 1: B  C'
                    D[a, c] + D[b, e] + D[d, f],  # 2: B' C'
                    D[a, d] + D[e, b] + D[c, f],  # 3: C  B
                    D[a, e] + D[d, b] + D[c, f],  # 4: C' B
                    D[a, d] + D[e, c] + D[b, f],  # 5: C  B'
                    D[a, e] + D[d, c] + D[b, f],  # 6: C' B'
                ],
                axis=-1,
            ) - old[..., None]
            return jnp.where(valid[..., None], deltas, jnp.inf)

        inner = window * window * _N_RECON
        delta, outer, flat = _windowed_best(score_block, n, inner, block=block)
        recon = flat % _N_RECON
        rest = flat // _N_RECON
        d2 = rest % window
        d1 = rest // window
        i = outer
        j = i + 1 + d1
        k = j + 1 + d2
        return delta, (i, j, k, recon)

    return _local_search(D, tour, best_move, _apply_three_opt, max_steps=max_steps)
