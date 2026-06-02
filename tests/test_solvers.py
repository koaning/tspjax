import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tspjax
from tspjax import MatrixProblem
from tspjax.solvers import (
    all_pairs,
    iterated_local_search,
    longest_edge,
    nearest,
    three_opt,
    two_opt,
    windowed,
)
from tspjax.solvers.candidates import Candidates


def _length(D, tour):
    import jax.numpy as _jnp

    return float(_jnp.sum(D[tour, _jnp.roll(tour, -1)]))


def _random_symmetric(n, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff**2).sum(-1)).astype(np.float32)
    return jnp.asarray(D)


def _is_permutation(tour, n):
    return sorted(int(x) for x in np.asarray(tour)) == list(range(n))


def _has_improving_swap(D, tour):
    """True if any full-window 2-opt swap (i>0, i<j) strictly improves the tour."""
    D = np.asarray(D)
    t = np.asarray(tour)
    n = len(t)
    for i in range(1, n):
        for j in range(i + 1, n):
            pi, ci, cj = t[i - 1], t[i], t[j]
            nj = t[(j + 1) % n]
            delta = (D[pi, cj] + D[ci, nj]) - (D[pi, ci] + D[cj, nj])
            if delta < -1e-4:
                return True
    return False


def test_two_opt_returns_valid_permutation_and_never_worse():
    p = tspjax.load("berlin52")
    start = jnp.arange(p.dimension)
    out = two_opt(p.distances, start)
    assert _is_permutation(out, p.dimension)
    assert float(p.tour_length(out)) <= float(p.tour_length(start)) + 1e-3


def test_two_opt_reaches_local_minimum():
    D = _random_symmetric(25, seed=1)
    out = two_opt(D, jnp.arange(25))
    assert _is_permutation(out, 25)
    # No improving 2-opt move should remain.
    assert not _has_improving_swap(D, out)
    # And running again changes nothing (idempotent at the local minimum).
    again = two_opt(D, out)
    assert np.array_equal(np.asarray(out), np.asarray(again))


def test_two_opt_default_start_is_identity():
    D = _random_symmetric(20, seed=2)
    explicit = two_opt(D, jnp.arange(20))
    default = two_opt(D)
    assert np.array_equal(np.asarray(explicit), np.asarray(default))


def test_two_opt_on_integer_matrix_instance():
    p = tspjax.load("gr17")  # explicit symmetric matrix, integer weights
    out = two_opt(p.distances, jnp.arange(p.dimension))
    assert _is_permutation(out, p.dimension)
    assert float(p.tour_length(out)) < float(p.tour_length(jnp.arange(p.dimension)))


def test_two_opt_rejects_asymmetric_matrix():
    D = jnp.array([[0.0, 1.0, 9.0], [9.0, 0.0, 1.0], [1.0, 9.0, 0.0]])
    p = MatrixProblem("asym", D)
    assert not p.is_symmetric
    with pytest.raises(ValueError, match="symmetric"):
        two_opt(p.distances, jnp.arange(3))


def test_two_opt_window_no_worse_than_narrow():
    D = _random_symmetric(40, seed=3)
    full = two_opt(D, jnp.arange(40), window=None)
    narrow = two_opt(D, jnp.arange(40), window=5)
    full_len = float(jnp.sum(D[full, jnp.roll(full, -1)]))
    narrow_len = float(jnp.sum(D[narrow, jnp.roll(narrow, -1)]))
    assert full_len <= narrow_len + 1e-3


def test_two_opt_is_vmappable_over_tours():
    D = _random_symmetric(30, seed=5)
    key = jax.random.PRNGKey(0)
    starts = jnp.stack(
        [jax.random.permutation(jax.random.fold_in(key, i), 30) for i in range(4)]
    )
    outs = jax.vmap(two_opt, in_axes=(None, 0))(D, starts)
    assert outs.shape == (4, 30)
    for k in range(4):
        assert _is_permutation(outs[k], 30)


def test_two_opt_bounded_memory_large_window_small():
    # Larger instance with a small window: must complete without materialising an
    # (n, n) delta grid (the blocked fold keeps peak memory ~O(block * window)).
    D = _random_symmetric(1500, seed=6)
    out = two_opt(D, jnp.arange(1500), window=8, max_steps=200)
    assert _is_permutation(out, 1500)


def _has_improving_3opt(D, tour):
    """True if any full-window 3-opt move (all 7 reconnections) strictly improves.

    Mirrors the on-device delta table in ``three_opt`` exactly, so it doubles as the
    oracle that the scored deltas and the applied reconnections agree.
    """
    D = np.asarray(D)
    t = np.asarray(tour)
    n = len(t)
    for i in range(1, n):
        for j in range(i + 1, n):
            for k in range(j + 1, n + 1):
                a, b = t[i - 1], t[i]
                c, d = t[j - 1], t[j]
                e, f = t[k - 1], t[k % n]
                old = D[a, b] + D[c, d] + D[e, f]
                news = (
                    D[a, c] + D[b, d] + D[e, f],  # 0: B' C
                    D[a, b] + D[c, e] + D[d, f],  # 1: B  C'
                    D[a, c] + D[b, e] + D[d, f],  # 2: B' C'
                    D[a, d] + D[e, b] + D[c, f],  # 3: C  B
                    D[a, e] + D[d, b] + D[c, f],  # 4: C' B
                    D[a, d] + D[e, c] + D[b, f],  # 5: C  B'
                    D[a, e] + D[d, c] + D[b, f],  # 6: C' B'
                )
                if min(news) < old - 1e-4:
                    return True
    return False


def test_three_opt_returns_valid_permutation_and_never_worse():
    p = tspjax.load("berlin52")
    start = jnp.arange(p.dimension)
    out = three_opt(p.distances, start)
    assert _is_permutation(out, p.dimension)
    assert float(p.tour_length(out)) <= float(p.tour_length(start)) + 1e-3


def test_three_opt_reaches_local_minimum():
    D = _random_symmetric(10, seed=1)
    out = three_opt(D, jnp.arange(10))
    assert _is_permutation(out, 10)
    # No improving 3-opt move should remain (validates the delta table vs. the
    # reconnection actually applied).
    assert not _has_improving_3opt(D, out)
    # And running again changes nothing (idempotent at the local minimum).
    again = three_opt(D, out)
    assert np.array_equal(np.asarray(out), np.asarray(again))


def test_three_opt_default_start_is_identity():
    D = _random_symmetric(20, seed=2)
    explicit = three_opt(D, jnp.arange(20))
    default = three_opt(D)
    assert np.array_equal(np.asarray(explicit), np.asarray(default))


def test_three_opt_subsumes_two_opt():
    # Full 3-opt includes the segment-reversal reconnections, so its local optimum is
    # no worse than 2-opt's from the same start.
    D = _random_symmetric(30, seed=4)
    start = jnp.arange(30)
    two = float(jnp.sum(D[two_opt(D, start), jnp.roll(two_opt(D, start), -1)]))
    three = three_opt(D, start)
    three_len = float(jnp.sum(D[three, jnp.roll(three, -1)]))
    assert three_len <= two + 1e-3


def test_three_opt_rejects_asymmetric_matrix():
    D = jnp.array([[0.0, 1.0, 9.0, 2.0], [9.0, 0.0, 1.0, 3.0],
                   [1.0, 9.0, 0.0, 4.0], [2.0, 3.0, 4.0, 0.0]])
    p = MatrixProblem("asym", D)
    assert not p.is_symmetric
    with pytest.raises(ValueError, match="symmetric"):
        three_opt(p.distances, jnp.arange(4))


def test_three_opt_window_no_worse_than_narrow():
    D = _random_symmetric(40, seed=3)
    full = three_opt(D, jnp.arange(40), window=None)
    narrow = three_opt(D, jnp.arange(40), window=5)
    full_len = float(jnp.sum(D[full, jnp.roll(full, -1)]))
    narrow_len = float(jnp.sum(D[narrow, jnp.roll(narrow, -1)]))
    assert full_len <= narrow_len + 1e-3


def test_three_opt_is_vmappable_over_tours():
    D = _random_symmetric(30, seed=5)
    key = jax.random.PRNGKey(0)
    starts = jnp.stack(
        [jax.random.permutation(jax.random.fold_in(key, i), 30) for i in range(4)]
    )
    outs = jax.vmap(three_opt, in_axes=(None, 0))(D, starts)
    assert outs.shape == (4, 30)
    for k in range(4):
        assert _is_permutation(outs[k], 30)


def test_three_opt_bounded_memory_large_window_small():
    # Large instance, small window: must complete without materialising an O(n^3)
    # delta grid (the blocked fold keeps peak memory ~O(block * window^2 * 7)).
    D = _random_symmetric(1500, seed=6)
    out = three_opt(D, jnp.arange(1500), window=6, max_steps=50)
    assert _is_permutation(out, 1500)


# --- candidate strategies ---------------------------------------------------------


def test_window_keyword_is_windowed_alias():
    # `window=w` must be exactly `candidates=windowed(w)`.
    D = _random_symmetric(40, seed=3)
    start = jnp.arange(40)
    via_kw = two_opt(D, start, window=7)
    via_strat = two_opt(D, start, candidates=windowed(7))
    assert np.array_equal(np.asarray(via_kw), np.asarray(via_strat))


def test_default_candidates_is_all_pairs():
    D = _random_symmetric(30, seed=2)
    start = jnp.arange(30)
    assert np.array_equal(
        np.asarray(two_opt(D, start)), np.asarray(two_opt(D, start, candidates=all_pairs))
    )


def test_window_and_candidates_together_is_error():
    D = _random_symmetric(10, seed=0)
    with pytest.raises(ValueError, match="not both"):
        two_opt(D, jnp.arange(10), window=3, candidates=nearest(4))
    with pytest.raises(ValueError, match="not both"):
        three_opt(D, jnp.arange(10), window=3, candidates=windowed(4))


@pytest.mark.parametrize("strategy", [all_pairs, windowed(6), nearest(8), longest_edge(8)])
def test_two_opt_strategies_valid_and_never_worse(strategy):
    D = _random_symmetric(60, seed=7)
    start = jnp.arange(60)
    out = two_opt(D, start, candidates=strategy)
    assert _is_permutation(out, 60)
    assert _length(D, out) <= _length(D, start) + 1e-3


def test_nearest_is_vmappable_over_tours():
    D = _random_symmetric(30, seed=5)
    key = jax.random.PRNGKey(0)
    starts = jnp.stack(
        [jax.random.permutation(jax.random.fold_in(key, i), 30) for i in range(4)]
    )
    outs = jax.vmap(lambda t: two_opt(D, t, candidates=nearest(6)))(starts)
    assert outs.shape == (4, 30)
    for k in range(4):
        assert _is_permutation(outs[k], 30)


def test_custom_strategy_function_plugs_in():
    # A user-supplied strategy: anchor every edge, pair only with the next edge (a
    # degenerate window of 1). Must compile and run like a shipped one.
    def next_only(D, n):
        idx = jnp.arange(n, dtype=jnp.int32)
        return Candidates(
            anchors=lambda cur: idx,
            partners=lambda cur, pos: pos[:, None] + jnp.int32(1),
            width=1,
            n_anchors=n,
        )

    D = _random_symmetric(25, seed=9)
    start = jnp.arange(25)
    out = two_opt(D, start, candidates=next_only)
    assert _is_permutation(out, 25)
    assert _length(D, out) <= _length(D, start) + 1e-3


def test_three_opt_nearest_valid_and_never_worse():
    D = _random_symmetric(40, seed=8)
    start = jnp.arange(40)
    out = three_opt(D, start, candidates=nearest(6))
    assert _is_permutation(out, 40)
    assert _length(D, out) <= _length(D, start) + 1e-3


def test_ils_returns_valid_tour_and_consistent_history():
    D = _random_symmetric(30, seed=2)
    start = jnp.arange(30)
    improve = lambda t: two_opt(D, t, window=8)  # noqa: E731
    best, best_len, history = iterated_local_search(
        D, start, improve, jax.random.PRNGKey(0), steps=12
    )
    assert _is_permutation(best, 30)
    # best_length is the true length of best_tour ...
    assert float(best_len) == pytest.approx(_length(D, best), abs=1e-3)
    # ... and the minimum of the recorded history.
    assert history.shape == (13,)
    assert float(jnp.min(history)) == pytest.approx(float(best_len), abs=1e-3)
    # history[0] is the initial local optimum (improve(start), no perturbation yet).
    assert float(history[0]) == pytest.approx(_length(D, improve(start)), abs=1e-3)


def test_ils_never_worse_than_a_single_improve_pass():
    D = _random_symmetric(40, seed=4)
    start = jnp.arange(40)
    improve = lambda t: two_opt(D, t, window=10)  # noqa: E731
    plain = _length(D, improve(start))
    _, best_len, _ = iterated_local_search(
        D, start, improve, jax.random.PRNGKey(1), steps=15
    )
    # ILS tracks the best ever seen, and step 0 is the plain improve result.
    assert float(best_len) <= plain + 1e-3


def test_ils_is_deterministic_given_key():
    D = _random_symmetric(30, seed=6)
    start = jnp.arange(30)
    improve = lambda t: two_opt(D, t, window=8)  # noqa: E731
    a = iterated_local_search(D, start, improve, jax.random.PRNGKey(7), steps=8)
    b = iterated_local_search(D, start, improve, jax.random.PRNGKey(7), steps=8)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert np.array_equal(np.asarray(a[2]), np.asarray(b[2]))


def test_ils_composes_two_opt_then_three_opt():
    # The whole point: the caller composes the improver schedule.
    p = tspjax.load("berlin52")
    D = p.distances
    start = jnp.arange(p.dimension)
    improve = lambda t: three_opt(D, two_opt(D, t, window=12), window=6)  # noqa: E731
    best, best_len, history = iterated_local_search(
        D, start, improve, jax.random.PRNGKey(0), steps=6
    )
    assert _is_permutation(best, p.dimension)
    assert float(best_len) <= float(history[0]) + 1e-3
