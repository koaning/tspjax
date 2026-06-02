"""Candidate-selection strategies for the local-search solvers.

A local-search step looks at a set of candidate moves, scores each, and applies the
best improving one. A *strategy* decides which moves are even in that set — the single
biggest lever on both speed and quality. Rather than bake the choice in behind flags,
the solvers take a strategy as a plain function, so the neighbourhood is pluggable and
users can write their own without a class hierarchy.

A move is anchored on **edge positions**: edge ``p`` is the tour edge
``cur[p] -> cur[(p + 1) % n]``. A strategy pairs an *anchor* edge with one or more
*partner* edges; the solver supplies the move maths around that pairing (2-opt scores
the pair, 3-opt chains a second partner off the first). A strategy is a callable
``(D, n) -> Candidates`` — it gets the distance matrix (so it can precompute things
like a nearest-neighbour table) and the tour size, and returns:

* ``anchors(cur) -> (n_anchors,)`` — the edge positions to anchor moves on.
* ``partners(cur, pos) -> (len(pos), width)`` — for each position in ``pos``, the
  partner edge positions to try. Called on the anchors, and again on the first
  partners for 3-opt's second cut, so it must accept an arbitrary ``(L,)`` of
  positions.

Everything returned is pure JAX with static shapes, so a strategy compiles and fuses
into the solver's single on-device program — passing one in costs nothing at runtime.

Shipped strategies:

* :func:`all_pairs` — every edge pair (full search). The default.
* :func:`windowed` — partners within ``w`` positions in the tour ordering. A *positional*
  window: cheap, but it can only repair crossings whose edges are within ``w`` of each
  other in the tour.
* :func:`nearest` — partners drawn from each city's ``k`` nearest cities (derived from
  ``D``). A *spatial* neighbourhood: a 2-opt move only helps by creating a short edge,
  and short edges connect near neighbours, so this keeps almost every useful move at a
  fraction of the cost. Best suited to 2-opt.
* :func:`longest_edge` — anchor only on the ``k`` longest edges, pair with everything.
  Targets the worst edges directly; cheap, but a weak neighbourhood on its own (crossing
  edges are not always long). 2-opt-oriented.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp
from jax import lax


class Candidates(NamedTuple):
    """A resolved candidate strategy (see module docstring)."""

    anchors: Callable  # (cur) -> (n_anchors,) int32 anchor edge positions
    partners: Callable  # (cur, pos:(L,)) -> (L, width) int32 partner edge positions
    width: int  # partners per anchor (static)
    n_anchors: int  # number of anchors (static)


def all_pairs(D, n):
    """Every edge pair — full, exhaustive search. The default neighbourhood."""
    idx = jnp.arange(n, dtype=jnp.int32)

    def anchors(cur):
        return idx

    def partners(cur, pos):
        return jnp.broadcast_to(idx[None, :], (pos.shape[0], n))

    return Candidates(anchors, partners, n, n)


def windowed(w):
    """Partners within ``w`` edge positions ahead in the tour ordering (positional)."""

    def build(D, n):
        ww = max(1, min(int(w), n - 1))
        off = jnp.arange(1, ww + 1, dtype=jnp.int32)
        idx = jnp.arange(n, dtype=jnp.int32)

        def anchors(cur):
            return idx

        def partners(cur, pos):
            return pos[:, None] + off[None, :]  # forward; out-of-range masked by solver

        return Candidates(anchors, partners, ww, n)

    return build


def nearest(k):
    """Partners drawn from each city's ``k`` nearest cities (spatial; from ``D``)."""

    def build(D, n):
        kk = max(1, min(int(k), n - 1))
        # Column 0 of top_k(-D) is the city itself (distance 0); drop it.
        nn = lax.top_k(-D, kk + 1)[1][:, 1:].astype(jnp.int32)  # (n, kk) nearest cities
        idx = jnp.arange(n, dtype=jnp.int32)

        def anchors(cur):
            return idx

        def partners(cur, pos):
            inv = jnp.zeros(n, jnp.int32).at[cur].set(idx)  # city -> tour position
            return inv[nn[cur[pos]]]  # (L, kk) positions of pos's cities' nearest cities

        return Candidates(anchors, partners, kk, n)

    return build


def longest_edge(k):
    """Anchor only on the ``k`` longest edges of the current tour; pair with all edges."""

    def build(D, n):
        kk = max(1, min(int(k), n))
        idx = jnp.arange(n, dtype=jnp.int32)

        def anchors(cur):
            edge = D[cur, jnp.roll(cur, -1)]
            return lax.top_k(edge, kk)[1].astype(jnp.int32)  # (kk,) longest-edge positions

        def partners(cur, pos):
            return jnp.broadcast_to(idx[None, :], (pos.shape[0], n))

        return Candidates(anchors, partners, n, kk)

    return build
