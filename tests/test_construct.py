import jax.numpy as jnp
import numpy as np

import tspjax
from tspjax.construct import hilbert_tour, moore_tour, morton_tour
from tspjax.solvers import two_opt


def _is_permutation(tour, n):
    return sorted(int(x) for x in np.asarray(tour)) == list(range(n))


def _tour_length(coords, tour):
    c = np.asarray(coords)[np.asarray(tour)]
    nxt = np.roll(c, -1, axis=0)
    return float(np.sqrt(((c - nxt) ** 2).sum(-1)).sum())


def _full_grid(bits):
    g = 1 << bits
    xs, ys = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    return jnp.asarray(np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32))


def _ordered_points(coords, tour):
    return np.asarray(coords)[np.asarray(tour)]


def test_constructors_return_valid_permutations():
    coords = tspjax.load("berlin52").coords
    n = coords.shape[0]
    for fn in (hilbert_tour, morton_tour, moore_tour):
        assert _is_permutation(fn(coords), n)
    assert _is_permutation(morton_tour(coords, order="yx"), n)


def test_constructors_beat_identity_tour():
    coords = tspjax.load("berlin52").coords
    n = coords.shape[0]
    identity_len = _tour_length(coords, jnp.arange(n))
    for fn in (hilbert_tour, morton_tour, moore_tour):
        assert _tour_length(coords, fn(coords)) < identity_len


def test_morton_order_flag_changes_tour():
    coords = tspjax.load("berlin52").coords
    xy = np.asarray(morton_tour(coords, order="xy"))
    yx = np.asarray(morton_tour(coords, order="yx"))
    assert not np.array_equal(xy, yx)


def test_hilbert_is_a_space_filling_curve():
    # Over a full grid, consecutive Hilbert cells are grid-adjacent (Manhattan == 1).
    bits = 3
    coords = _full_grid(bits)
    pts = _ordered_points(coords, hilbert_tour(coords, bits=bits))
    steps = np.abs(np.diff(pts, axis=0)).sum(-1)
    assert np.all(steps == 1)


def test_moore_is_a_closed_space_filling_curve():
    bits = 3
    coords = _full_grid(bits)
    pts = _ordered_points(coords, moore_tour(coords, bits=bits))
    steps = np.abs(np.diff(pts, axis=0)).sum(-1)
    assert np.all(steps == 1)  # consecutive cells adjacent
    closing = np.abs(pts[-1] - pts[0]).sum()
    assert closing == 1  # ...and the loop closes


def test_morton_has_jumps_unlike_hilbert():
    bits = 3
    coords = _full_grid(bits)
    pts = _ordered_points(coords, morton_tour(coords, bits=bits))
    steps = np.abs(np.diff(pts, axis=0)).sum(-1)
    assert np.any(steps > 1)  # Z-order is not adjacency-preserving


def test_curve_start_helps_windowed_two_opt():
    p = tspjax.load("berlin52")
    n = p.dimension
    D = p.distances
    identity_out = two_opt(D, jnp.arange(n), window=4)
    hilbert_out = two_opt(D, hilbert_tour(p.coords), window=4)
    id_len = float(p.tour_length(identity_out))
    hi_len = float(p.tour_length(hilbert_out))
    assert hi_len < id_len
