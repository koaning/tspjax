import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tspjax
from tspjax import MatrixProblem
from tspjax.solvers import two_opt


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
