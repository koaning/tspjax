import jax.numpy as jnp
import pytest

import tspjax
from tspjax import EuclideanProblem, MatrixProblem


def test_list_problems_complete():
    names = tspjax.list_problems()
    assert len(names) == 131  # 111 symmetric + 19 ATSP + santa2018
    assert "berlin52" in names
    assert "gr17" in names  # EXPLICIT symmetric instance
    assert "br17" in names  # ATSP instance
    assert "santa2018" in names  # bundled non-TSPLIB instance
    assert names == tuple(sorted(names))


def test_load_santa_by_name():
    # santa2018 loads through the same entry point as everything else.
    import jax.numpy as jnp

    from tspjax import FunctionProblem

    p = tspjax.load("santa2018")
    assert isinstance(p, FunctionProblem)
    assert p.dimension == 197769
    assert float(p.tour_length(jnp.arange(p.dimension))) > 0


def test_load_atsp_is_asymmetric():
    p = tspjax.load("ftv33")
    assert isinstance(p, MatrixProblem)
    assert p.dimension == 34
    assert not p.is_symmetric  # genuine asymmetric instance
    assert p.best_known == 1286
    assert p.distances.shape == (34, 34)


def test_atsp_direction_changes_cost():
    p = tspjax.load("ftv33")
    fwd = float(p.tour_length(jnp.arange(p.dimension)))
    rev = float(p.tour_length(jnp.arange(p.dimension)[::-1]))
    # Asymmetric: forward and reversed traversals generally differ.
    assert fwd != rev


def test_load_euclidean_basic():
    p = tspjax.load("berlin52")
    assert isinstance(p, EuclideanProblem)
    assert p.dimension == 52
    assert p.best_known == 7542
    assert p.is_symmetric
    assert p.coords.shape == (52, 2)
    assert p.distances.shape == (52, 52)


def test_load_explicit_matrix():
    p = tspjax.load("gr17")
    assert isinstance(p, MatrixProblem)
    assert p.dimension == 17
    assert p.distances.shape == (17, 17)
    assert p.is_symmetric
    # Diagonal is zero, matrix is symmetric.
    assert jnp.allclose(jnp.diag(p.distances), 0.0)
    assert jnp.allclose(p.distances, p.distances.T)


def test_unknown_problem_raises():
    with pytest.raises(FileNotFoundError):
        tspjax.load("does_not_exist")


def test_all_problems_load():
    # Every bundled file should parse without error and have a sane shape.
    # A dense (n, n) matrix is infeasible for the largest instances (pla85900
    # would be ~59 GB), so only materialize distances for modest sizes; bigger
    # ones just need to parse and expose coordinates.
    MATERIALIZE_BELOW = 2000
    for name in tspjax.list_problems():
        p = tspjax.load(name)
        assert p.dimension > 0
        if p.dimension < MATERIALIZE_BELOW:
            D = p.distances
            if D is not None:
                assert D.shape == (p.dimension, p.dimension)
                assert bool(jnp.isfinite(D).all())  # forces real materialization
        else:
            assert p.coords is not None or p.distances is not None
