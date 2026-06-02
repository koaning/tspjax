import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tspjax.perturb import double_bridge, random_reversal, random_shuffle


def _is_permutation(tour, n):
    return sorted(int(x) for x in np.asarray(tour)) == list(range(n))


def _changed_positions(a, b):
    """Indices where tours a and b differ."""
    return np.flatnonzero(np.asarray(a) != np.asarray(b))


# A move callable plus how to invoke it with a key (some take only (tour, key)).
MOVES = [double_bridge, random_reversal, random_shuffle]


@pytest.mark.parametrize("move", MOVES)
def test_move_returns_valid_permutation(move):
    n = 52
    out = move(jnp.arange(n), jax.random.PRNGKey(0))
    assert out.dtype == jnp.int32
    assert _is_permutation(out, n)


def test_double_bridge_changes_tour():
    n = 50
    start = jnp.arange(n)
    base = jax.random.PRNGKey(7)
    for i in range(8):
        out = double_bridge(start, jax.random.fold_in(base, i))
        assert _is_permutation(out, n)
        assert not np.array_equal(np.asarray(out), np.asarray(start))


def test_random_reversal_matches_numpy_reference():
    n = 40
    start = jnp.arange(n)
    key = jax.random.PRNGKey(3)
    out = random_reversal(start, key)

    # Replicate _sample_segment's sampling with the same key/split.
    L = n - 1
    k_len, k_pos = jax.random.split(key)
    length = int(jax.random.randint(k_len, (), 2, L + 1))
    i = int(jax.random.randint(k_pos, (), 1, n - length + 1))
    j = i + length - 1

    t = np.arange(n)
    ref = np.concatenate([t[:i], t[i : j + 1][::-1], t[j + 1 :]])
    assert np.array_equal(np.asarray(out), ref)


def test_random_shuffle_confined_to_window():
    n = 60
    start = jnp.arange(n)
    base = jax.random.PRNGKey(11)
    for k in range(8):
        key = jax.random.fold_in(base, k)
        out = np.asarray(random_shuffle(start, key))
        assert _is_permutation(out, n)

        # Replicate _sample_segment's sampling (random_shuffle splits the key first).
        L = n - 1
        k_seg, _ = jax.random.split(key)
        k_len, k_pos = jax.random.split(k_seg)
        length = int(jax.random.randint(k_len, (), 2, L + 1))
        i = int(jax.random.randint(k_pos, (), 1, n - length + 1))
        j = i + length - 1

        t = np.arange(n)
        # Everything outside the window is untouched; the window is a permutation of itself.
        assert np.array_equal(out[:i], t[:i])
        assert np.array_equal(out[j + 1 :], t[j + 1 :])
        assert sorted(out[i : j + 1]) == sorted(t[i : j + 1])


@pytest.mark.parametrize("move", MOVES)
def test_move_is_deterministic_given_key(move):
    n = 45
    key = jax.random.PRNGKey(123)
    a = move(jnp.arange(n), key)
    b = move(jnp.arange(n), key)
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("move", MOVES)
def test_move_differs_across_keys(move):
    n = 80
    start = jnp.arange(n)
    a = move(start, jax.random.PRNGKey(0))
    b = move(start, jax.random.PRNGKey(1))
    assert not np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("move", MOVES)
def test_move_is_jittable(move):
    n = 30
    out = jax.jit(move)(jnp.arange(n), jax.random.PRNGKey(2))
    assert _is_permutation(out, n)


@pytest.mark.parametrize("move", MOVES)
def test_move_is_vmappable_over_keys(move):
    n = 30
    keys = jnp.stack([jax.random.PRNGKey(i) for i in range(4)])
    outs = jax.vmap(move, in_axes=(None, 0))(jnp.arange(n), keys)
    assert outs.shape == (4, n)
    for k in range(4):
        assert _is_permutation(outs[k], n)


@pytest.mark.parametrize("move", [random_reversal, random_shuffle])
def test_max_len_bounds_segment(move):
    n = 100
    start = jnp.arange(n)
    base = jax.random.PRNGKey(5)
    max_len = 6
    for k in range(12):
        out = move(start, jax.random.fold_in(base, k), max_len=max_len)
        changed = _changed_positions(start, out)
        assert changed.size <= max_len


@pytest.mark.parametrize("move", [random_reversal, random_shuffle])
def test_small_n_is_noop_reversal_shuffle(move):
    for n in (0, 1, 2):
        start = jnp.arange(n)
        out = move(start, jax.random.PRNGKey(0))
        assert np.array_equal(np.asarray(out), np.asarray(start))


def test_small_n_is_noop_double_bridge():
    for n in (0, 1, 2, 3):
        start = jnp.arange(n)
        out = double_bridge(start, jax.random.PRNGKey(0))
        assert np.array_equal(np.asarray(out), np.asarray(start))
