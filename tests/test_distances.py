import jax.numpy as jnp
import pytest

import tspjax
from tspjax import distances


def test_euc_2d_known_values():
    # 3-4-5 right triangle; EUC_2D rounds to nearest integer.
    coords = jnp.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    D = distances.euclidean_2d(coords)
    assert float(D[0, 1]) == 3.0
    assert float(D[1, 2]) == 4.0
    assert float(D[0, 2]) == 5.0


def test_euc_2d_half_distances_round_up():
    coords = jnp.array([[0.0, 0.0], [2.5, 0.0], [4.5, 0.0]])
    D = distances.euclidean_2d(coords)
    assert float(D[0, 1]) == 3.0
    assert float(D[0, 2]) == 5.0


def test_ceil_2d_rounds_up():
    coords = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # sqrt(2) ~ 1.41 -> 2
    D = distances.ceil_2d(coords)
    assert float(D[0, 1]) == 2.0


def test_geo_burma14_reasonable():
    # GEO distances should be positive integers off the diagonal.
    p = tspjax.load("burma14")
    assert p.edge_weight_type == "GEO"
    D = p.distances
    off = D[~jnp.eye(p.dimension, dtype=bool)]
    assert jnp.all(off > 0)
    assert jnp.allclose(D, jnp.round(D))  # integer-valued
    # The GEO formula's "+1.0" term would make self-distance 1; we zero it.
    assert jnp.allclose(jnp.diag(D), 0.0)


def test_att_is_integer_valued():
    p = tspjax.load("att48")
    assert p.edge_weight_type == "ATT"
    assert jnp.allclose(p.distances, jnp.round(p.distances))


def test_unsupported_type_raises():
    with pytest.raises(ValueError):
        distances.distance_matrix(jnp.zeros((3, 2)), "BOGUS_2D")
