"""Tour *perturbations*: random kicks that knock a tour out of a local optimum.

The improvers in :mod:`tspjax.solvers` (``two_opt``, ``three_opt``) drive a tour to a
*local* minimum and stop — by construction no small move improves it any further. To
keep searching you *kick* the tour: apply a random structural change the improver
can't immediately undo, then re-optimise and keep the result only if it's better.
That kick → improve → accept loop is **iterated local search**; these functions are
the kick step (compose them with the solvers yourself, e.g. in a notebook).

* :func:`double_bridge` — the classic 4-opt kick. Cut the tour at three points into
  ``A·B·C·D`` and reconnect as ``A·C·B·D``. A single 2-opt move can't reverse it,
  which is exactly why it's the canonical iterated-local-search perturbation.
* :func:`random_reversal` — reverse a random contiguous segment (a random 2-opt move).
* :func:`random_shuffle` — randomly permute a random contiguous block of the tour.

Like the constructors and solvers, these take **raw arrays** (a ``(n,)`` tour and a
``jax.random`` key), never a ``Problem``, and use pure index math with static shapes —
so each is ``jax.jit``-able and ``jax.vmap``-able over a leading batch axis on the
tour and the key. Position 0 is left anchored, matching ``two_opt``'s convention, so a
kick is a genuine structural change relative to the cycle's reference point.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["double_bridge", "random_reversal", "random_shuffle"]


def _reverse_segment(tour, i, j):
    """Reverse ``tour[i..j]`` inclusive with pure index math (static shape, jit-safe)."""
    k = jnp.arange(tour.shape[0])
    in_seg = (k >= i) & (k <= j)
    src = jnp.where(in_seg, i + j - k, k)
    return tour[src]


def _sample_segment(key, n, max_len):
    """Sample ``(i, j)`` with ``1 <= i``, ``i < j <= n-1``, length in ``[2, max_len]``."""
    L = jnp.minimum(max_len, n - 1)
    k_len, k_pos = jax.random.split(key)
    length = jax.random.randint(k_len, (), 2, L + 1)       # [2, L]
    i = jax.random.randint(k_pos, (), 1, n - length + 1)   # i in [1, n-length]
    j = i + length - 1
    return i, j


def double_bridge(tour, key):
    """Double-bridge 4-opt kick: split into ``A·B·C·D`` and reconnect as ``A·C·B·D``.

    Samples three distinct interior cut points ``0 < p1 < p2 < p3 < n`` and rebuilds
    the tour by swapping the two middle segments. Returns the kicked ``(n,)`` ``int32``
    tour. For ``n < 4`` (no room for three distinct cuts) the tour is returned
    unchanged. Pure index math, so ``jax.jit`` / ``jax.vmap``-able over a batch of
    ``key`` values (``tour`` fixed, or both batched).
    """
    n = int(tour.shape[0])
    tour = jnp.asarray(tour, dtype=jnp.int32)
    if n < 4:
        return tour  # no room for three distinct interior cut points

    cuts = jax.random.choice(key, jnp.arange(1, n), shape=(3,), replace=False)
    p1, p2, p3 = jnp.sort(cuts)
    len_c = p3 - p2

    q = jnp.arange(n)
    b_c = p1            # output start of segment C
    b_b = p1 + len_c    # output start of segment B
    src = q                                                    # A ([0,p1)) and D ([p3,n)) stay put
    src = jnp.where((q >= b_c) & (q < b_b), p2 + (q - b_c), src)  # C region -> [p2, p3)
    src = jnp.where((q >= b_b) & (q < p3), p1 + (q - b_b), src)   # B region -> [p1, p2)
    return tour[src].astype(jnp.int32)


def random_reversal(tour, key, *, max_len=None):
    """Reverse a random contiguous segment (a random 2-opt move).

    Samples a segment ``tour[i..j]`` (``i >= 1``, length in ``[2, max_len]``) and
    reverses it. ``max_len`` (``None`` -> ``n-1``) caps the kick strength. Returns the
    ``(n,)`` ``int32`` tour; for ``n < 3`` it is returned unchanged. ``jax.jit`` /
    ``jax.vmap``-able over a batch of keys.
    """
    n = int(tour.shape[0])
    tour = jnp.asarray(tour, dtype=jnp.int32)
    if n < 3:
        return tour
    L = (n - 1) if max_len is None else min(int(max_len), n - 1)
    i, j = _sample_segment(key, n, L)
    return _reverse_segment(tour, i, j).astype(jnp.int32)


def random_shuffle(tour, key, *, max_len=None):
    """Randomly permute a random contiguous block of the tour.

    Samples a window ``tour[i..j]`` (``i >= 1``, length in ``[2, max_len]``) and
    scrambles the cities inside it, leaving everything outside fixed. ``max_len``
    (``None`` -> ``n-1``) caps the kick strength. Returns the ``(n,)`` ``int32`` tour;
    for ``n < 3`` it is returned unchanged.

    Implemented with an argsort over per-position sort keys: outside the window each
    key equals the position itself, while inside it lies in ``[i, j+1)``. Since every
    inside key is strictly between neighbours ``i-1`` and ``j+1``, ``argsort`` drops
    the window's cities back into exactly slots ``[i, j]`` in random order and leaves
    every outside position put — a valid permutation with static shape, so the move is
    ``jax.jit`` / ``jax.vmap``-able over a batch of keys.
    """
    n = int(tour.shape[0])
    tour = jnp.asarray(tour, dtype=jnp.int32)
    if n < 3:
        return tour
    L = (n - 1) if max_len is None else min(int(max_len), n - 1)
    k_seg, k_perm = jax.random.split(key)
    i, j = _sample_segment(k_seg, n, L)
    p = jnp.arange(n)
    in_win = (p >= i) & (p <= j)
    u = jax.random.uniform(k_perm, (n,))
    keys = jnp.where(in_win, i + u * (j - i + 1), p.astype(jnp.float32))
    src = jnp.argsort(keys)
    return tour[src].astype(jnp.int32)
