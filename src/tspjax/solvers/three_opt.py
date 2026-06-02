"""Windowed 3-opt local search.

3-opt removes three edges and reconnects the three resulting segments in one of the
seven non-identity ways. With the tour split into a fixed part and two reorderable
segments ``B`` and ``C``, three of the seven reconnections are plain segment
reversals (each equivalent to a single 2-opt move) and the other four genuinely swap
the order of ``B`` and ``C`` — moves no 2-opt can make. Scoring all seven makes this a
full 3-opt: it subsumes 2-opt and converges to a true 3-opt local optimum.

Which triples get scored is decided by a pluggable *candidate strategy* (see
:mod:`.candidates`): a 3-opt move cuts three edges ``e1 < e2 < e3``, so the strategy
picks the second cut from the first's partners and the third from the second's. The
default :func:`~tspjax.solvers.candidates.all_pairs` is a full search;
:func:`~tspjax.solvers.candidates.windowed` keeps both segments short (the cheap
positional neighbourhood best suited to 3-opt). The shared core in
:mod:`._local_search` folds the search in blocks so peak memory is
``O(block * width^2 * 7)``, never the full ``O(n^3)`` grid — this is what makes 3-opt
feasible at all. Per-step work is ``O(n * width^2)``, so use a small window for large
``n``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._local_search import _fold_best, _local_search
from .candidates import all_pairs, windowed

#: Anchors of the candidate grid scored at once. Caps peak memory at O(block * width^2 * 7).
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


def three_opt(distances, tour=None, *, window=None, candidates=None, max_steps=10_000):
    """Best-improvement 3-opt to a local minimum (or ``max_steps``).

    ``distances`` is an ``(n, n)`` **symmetric** matrix; ``tour`` an optional ``(n,)``
    start permutation (defaults to ``jnp.arange(n)``). Returns the improved ``(n,)``
    ``int32`` tour.

    The candidate set is chosen by ``candidates``, a strategy from :mod:`.candidates`
    (default :func:`~tspjax.solvers.candidates.all_pairs`). The ``window`` keyword is a
    convenience alias for ``candidates=windowed(window)``; passing both is an error. A
    3-opt move cuts edges ``e1 < e2 < e3``, with ``e2`` drawn from ``e1``'s partners and
    ``e3`` from ``e2``'s — so the positional strategies (``all_pairs``, ``windowed``)
    map cleanly; the spatial ones are 2-opt-oriented.

    All seven non-identity reconnections are scored, so this is a full 3-opt: it
    subsumes 2-opt and reaches a true 3-opt local optimum. Per-step work is
    ``O(n * width^2)`` — use a small window for large ``n``.

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

    if window is not None and candidates is not None:
        raise ValueError(
            "pass either `window` or `candidates`, not both; `window=w` is just an "
            "alias for `candidates=windowed(w)`."
        )

    if tour is None:
        tour = jnp.arange(n)
    tour = jnp.asarray(tour, dtype=jnp.int32)

    if n < 5:
        return tour  # no non-trivial 3-opt move exists

    if candidates is None:
        candidates = all_pairs if window is None else windowed(window)
    spec = candidates(D, n)

    block = min(_BLOCK, spec.n_anchors)
    n_blocks = (spec.n_anchors + block - 1) // block
    m = spec.width

    def best_move(cur):
        anchors = spec.anchors(cur)

        def score_block(b):
            rows = b * block + jnp.arange(block, dtype=jnp.int32)
            valid_row = (rows < spec.n_anchors)[:, None, None]
            e1 = anchors[jnp.clip(rows, 0, spec.n_anchors - 1)]  # (block,) first cut
            e2 = spec.partners(cur, e1)  # (block, m) second-cut candidates
            e3 = spec.partners(cur, e2.reshape(-1)).reshape(block, m, m)  # (block, m, m)
            E1 = e1[:, None, None]
            E2 = e2[:, :, None]
            E3 = e3
            valid = valid_row & (E1 >= 0) & (E1 < E2) & (E2 < E3) & (E3 <= n - 1)

            # Clip gather indices into range; invalid entries are masked to inf below.
            # Removed edges sit at e1, e2, e3: (a,b), (c,d), (e,f).
            c1, c2, c3 = (jnp.clip(x, 0, n - 1) for x in (E1, E2, E3))
            a, b = cur[c1], cur[(c1 + 1) % n]
            c, d = cur[c2], cur[(c2 + 1) % n]
            e, f = cur[c3], cur[(c3 + 1) % n]

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
            delta = jnp.where(valid[..., None], deltas, jnp.inf).reshape(-1)
            # Move fields: cut positions become segment starts (i, j, k) = (e+1) and
            # the reconnection index, broadcast over the 7-wide recon axis.
            shape = (block, m, m, _N_RECON)
            i = jnp.broadcast_to((c1 + 1)[..., None], shape).reshape(-1)
            j = jnp.broadcast_to((c2 + 1)[..., None], shape).reshape(-1)
            k = jnp.broadcast_to((c3 + 1)[..., None], shape).reshape(-1)
            recon = jnp.broadcast_to(
                jnp.arange(_N_RECON, dtype=jnp.int32), shape
            ).reshape(-1)
            return delta, (i, j, k, recon)

        delta, (i, j, k, recon) = _fold_best(score_block, n_blocks, 4)
        return delta, (i, j, k, recon)

    return _local_search(D, tour, best_move, _apply_three_opt, max_steps=max_steps)
