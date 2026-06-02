import math

import jax.numpy as jnp
import numpy as np
import pytest

import tspjax
from tspjax import FunctionProblem, santa


def test_sieve():
    p = santa.sieve(12)
    assert list(np.flatnonzero(p)) == [2, 3, 5, 7, 11]
    assert not p[0] and not p[1]


def test_prime_path_cost_hand_computed():
    # Cities on a line at x = 0..9. Identity tour [0..9] as a closed cycle:
    #   steps 1-9 are unit edges (no penalty), step 10 is the 9->0 return (dist 9).
    # City 9 is not prime, so step 10 gets the 1.1 penalty: 9 + 9*1.1 = 18.9.
    coords = jnp.array([[float(i), 0.0] for i in range(10)])
    cost = santa.prime_path_cost(coords)
    assert float(cost(jnp.arange(10))) == pytest.approx(18.9, rel=1e-5)


def _reference_cost(coords, path, is_prime):
    """Literal transcription of the Kaggle reference scoring function."""
    total, prev, step = 0.0, path[0], 1
    for city in path[1:]:
        d = math.dist(coords[city], coords[prev])
        total += d * (1 + 0.1 * ((step % 10 == 0) * int(not is_prime[prev])))
        prev, step = city, step + 1
    return total


def test_matches_kaggle_reference():
    rng = np.random.default_rng(0)
    n = 53
    coords = rng.uniform(0, 1000, size=(n, 2))
    tour = np.concatenate([[0], rng.permutation(np.arange(1, n))])  # start at city 0
    is_prime = santa.sieve(n)

    ours = float(santa.prime_path_cost(jnp.asarray(coords))(jnp.asarray(tour)))
    ref = _reference_cost(coords, list(tour) + [0], is_prime)  # append return-to-pole
    assert ours == pytest.approx(ref, rel=1e-4)


def test_load_bundled():
    p = santa.load()
    assert isinstance(p, FunctionProblem)
    assert p.dimension == 197769
    assert not p.is_symmetric
    assert p.coords.shape == (197769, 2)
    # Cost of the "dumbest path" (visit cities in id order) is finite and positive.
    length = float(p.tour_length(jnp.arange(p.dimension)))
    assert length > 0 and math.isfinite(length)


def test_load_custom_csv(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text("CityId,X,Y\n0,0,0\n1,1,0\n2,1,1\n")
    p = santa.load(csv)
    assert p.dimension == 3
    assert float(p.tour_length(jnp.array([0, 1, 2]))) == pytest.approx(1 + 1 + math.sqrt(2))


def test_santa_exposed_on_package():
    assert hasattr(tspjax, "santa")
