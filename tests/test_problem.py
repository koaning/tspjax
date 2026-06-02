import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tspjax
from tspjax import FunctionProblem, MatrixProblem


def test_tour_length_matches_manual_sum():
    p = tspjax.load("berlin52")
    tour = jnp.arange(p.dimension)
    D = p.distances
    nxt = jnp.roll(tour, -1)
    expected = sum(float(D[i, j]) for i, j in zip(np.asarray(tour), np.asarray(nxt)))
    assert float(p.tour_length(tour)) == pytest.approx(expected)


def test_tour_length_is_jittable_and_vmappable():
    p = tspjax.load("berlin52")
    fast = jax.jit(p.tour_length)
    key = jax.random.PRNGKey(0)
    tours = jnp.stack([jax.random.permutation(jax.random.fold_in(key, i), p.dimension)
                       for i in range(8)])
    lengths = jax.vmap(fast)(tours)
    assert lengths.shape == (8,)
    assert jnp.all(lengths > 0)


def test_matrix_problem_asymmetric_direction_matters():
    D = jnp.array([[0.0, 1.0, 9.0], [9.0, 0.0, 1.0], [1.0, 9.0, 0.0]])
    p = MatrixProblem("asym", D)
    assert not p.is_symmetric
    forward = p.tour_length(jnp.array([0, 1, 2]))   # 1 + 1 + 1
    backward = p.tour_length(jnp.array([0, 2, 1]))  # 9 + 9 + 9
    assert float(forward) == pytest.approx(3.0)
    assert float(backward) == pytest.approx(27.0)


def test_function_problem_from_pairwise():
    coords = jnp.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])

    def dist(i, j):
        d = coords[i] - coords[j]
        return jnp.sqrt(jnp.sum(d**2))

    p = FunctionProblem.from_pairwise("triangle", 3, dist, is_symmetric=True)
    length = p.tour_length(jnp.array([0, 1, 2]))  # 3 + 4 + 5
    assert float(length) == pytest.approx(12.0)


def test_repr_smoke():
    p = tspjax.load("berlin52")
    assert "berlin52" in repr(p)
