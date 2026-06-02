"""Iterated local search: orchestrate improvers and kicks into one search.

A single local-search pass (``two_opt``, ``three_opt``) drives a tour to a *local*
optimum and stops. :func:`iterated_local_search` keeps going: improve once, then
repeatedly **perturb** the current tour (a kick from :mod:`tspjax.perturb`) and
**re-improve**, keeping a candidate whenever it beats the current tour. The best tour
ever seen is tracked separately, so a kick that leads nowhere is never lost.

Unlike the kernels in this package, this is a **host-level driver**: it loops in
Python and calls the on-device improvers each iteration (they keep their own
host-side symmetry guards, so they can't be traced inside a single ``jit``). That's
the right altitude for a metaheuristic — the heavy per-iteration work still happens
on device inside ``improve``; only the cheap accept/track bookkeeping is on the host.

The improver is *yours to compose*. ``improve`` is any ``tour -> tour`` callable, so
you decide the schedule by closing over the solvers::

    from tspjax.solvers import two_opt, three_opt, iterated_local_search

    D = p.distances
    improve = lambda t: three_opt(D, two_opt(D, t, window=20), window=10)
    best, best_len, history = iterated_local_search(
        D, hilbert_tour(p.coords), improve, jax.random.PRNGKey(0), steps=100
    )

``perturb`` defaults to :func:`tspjax.perturb.double_bridge` (the canonical kick) but
takes any ``(tour, key) -> tour``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..perturb import double_bridge


def _tour_length(D, tour):
    """Closed-cycle length of ``tour`` under distance matrix ``D`` (scalar float32)."""
    return jnp.sum(D[tour, jnp.roll(tour, -1)])


def iterated_local_search(distances, tour, improve, key, *, perturb=double_bridge, steps=50):
    """Iterated local search: improve, then loop ``perturb -> improve``, keeping the best.

    ``distances`` is an ``(n, n)`` matrix and ``tour`` a ``(n,)`` start permutation.
    ``improve(tour) -> tour`` is the local-search step (compose ``two_opt`` /
    ``three_opt`` yourself, closing over ``distances``); ``perturb(tour, key) -> tour``
    is the random kick (defaults to :func:`~tspjax.perturb.double_bridge`). Runs
    ``steps`` kick/improve iterations off a single ``jax.random`` ``key``.

    Acceptance is *better-walk*: the current tour advances only to a strictly shorter
    candidate, while ``best`` records the shortest tour seen across all iterations.

    Returns ``(best_tour, best_length, history)``:

    * ``best_tour`` — the shortest ``(n,)`` ``int32`` tour found.
    * ``best_length`` — its closed-cycle length (scalar ``float32``).
    * ``history`` — a ``(steps + 1,)`` ``float32`` array of lengths: ``history[0]`` is
      the initial local optimum, ``history[k]`` (``k >= 1``) the length of the ``k``-th
      perturb/improve candidate. Plot it to see the search explore;
      ``jnp.minimum.accumulate(history)`` is the best-so-far curve, and
      ``best_length == history.min()``.
    """
    D = jnp.asarray(distances, dtype=jnp.float32)
    tour = jnp.asarray(tour, dtype=jnp.int32)

    current = improve(tour)
    cur_len = _tour_length(D, current)
    best, best_len = current, cur_len
    history = [cur_len]

    for _ in range(int(steps)):
        key, sub = jax.random.split(key)
        candidate = improve(perturb(current, sub))
        cand_len = _tour_length(D, candidate)
        history.append(cand_len)

        if cand_len < cur_len:
            current, cur_len = candidate, cand_len
        if cand_len < best_len:
            best, best_len = candidate, cand_len

    return best, best_len, jnp.stack(history).astype(jnp.float32)
