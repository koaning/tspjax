"""Tour *constructors*: turn coordinates into a good initial tour.

These are the companions to the *improvers* in :mod:`tspjax.solvers`. Each maps an
``(n, 2)`` coordinate array to a ``(n,)`` permutation by ordering the cities along a
space-filling curve — quantise the coordinates onto a ``2**bits`` grid, compute each
city's index along the curve, and ``argsort``. The result is an ``O(n log n)`` tour
that's typically a few tens of percent above optimal: a far better starting point for
:func:`tspjax.solvers.two_opt` than the identity tour.

A curve-ordered tour also places spatially-near cities near each other *along the
route*, so a cheap positional window in ``two_opt`` behaves like a spatial
neighbourhood — which is where the improving moves live.

* :func:`hilbert_tour` — Hilbert curve. Strong locality, no Z-order jumps.
* :func:`morton_tour`  — Morton / Z-order. ``order`` picks which coordinate supplies
  the high bit of each interleaved pair (``"xy"`` vs the transposed ``"yx"``).
* :func:`moore_tour`   — Moore curve, the *closed-loop* Hilbert variant; apt because a
  TSP tour is a cycle (its two ends are grid-adjacent).
"""

from __future__ import annotations

import jax.numpy as jnp

_DEFAULT_BITS = 16


def _quantize(coords, bits: int):
    """Map ``(n, 2)`` float coords onto an integer ``2**bits`` grid, per axis."""
    coords = jnp.asarray(coords, dtype=jnp.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2); got shape {coords.shape}")
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = jnp.where(hi > lo, hi - lo, 1.0)
    scale = (1 << bits) - 1
    q = jnp.floor((coords - lo) / span * scale + 0.5).astype(jnp.uint32)
    return jnp.clip(q, 0, scale)


def _part1by1(v, bits: int):
    """Spread the low ``bits`` of ``v`` so a zero sits between each original bit."""
    v = v.astype(jnp.uint32)
    out = jnp.zeros_like(v)
    for i in range(bits):
        out = out | (((v >> i) & jnp.uint32(1)) << (2 * i))
    return out


def _hilbert_d(x, y, bits: int):
    """Hilbert distance ``d`` for each ``(x, y)`` on a ``2**bits`` grid (uint32)."""
    x = x.astype(jnp.uint32)
    y = y.astype(jnp.uint32)
    d = jnp.zeros_like(x)
    nm1 = jnp.uint32((1 << bits) - 1)
    s = jnp.uint32(1 << (bits - 1))
    for _ in range(bits):
        rx = ((x & s) > 0).astype(jnp.uint32)
        ry = ((y & s) > 0).astype(jnp.uint32)
        d = d + s * s * ((jnp.uint32(3) * rx) ^ ry)
        # rot(N, x, y, rx, ry): reflect when rx == 1, then swap, both only if ry == 0.
        ry0 = ry == 0
        flip = ry0 & (rx == 1)
        x = jnp.where(flip, nm1 - x, x)
        y = jnp.where(flip, nm1 - y, y)
        x, y = jnp.where(ry0, y, x), jnp.where(ry0, x, y)
        s = s >> jnp.uint32(1)
    return d


def hilbert_tour(coords, *, bits: int = _DEFAULT_BITS):
    """Order cities along a Hilbert curve. Returns a ``(n,)`` ``int32`` tour."""
    q = _quantize(coords, bits)
    d = _hilbert_d(q[:, 0], q[:, 1], bits)
    return jnp.argsort(d).astype(jnp.int32)


def morton_tour(coords, *, order: str = "xy", bits: int = _DEFAULT_BITS):
    """Order cities along a Morton (Z-order) curve. Returns a ``(n,)`` ``int32`` tour.

    ``order="xy"`` gives x the high bit of each interleaved pair; ``"yx"`` the
    transpose. Both are valid Z-orders and generally produce different tours.
    """
    q = _quantize(coords, bits)
    x, y = q[:, 0], q[:, 1]
    if order == "yx":
        x, y = y, x
    elif order != "xy":
        raise ValueError(f"order must be 'xy' or 'yx'; got {order!r}")
    code = (_part1by1(x, bits) << jnp.uint32(1)) | _part1by1(y, bits)
    return jnp.argsort(code).astype(jnp.int32)


def moore_tour(coords, *, bits: int = _DEFAULT_BITS):
    """Order cities along a Moore curve (closed-loop Hilbert). ``(n,)`` ``int32`` tour.

    The Moore curve of order ``k`` is four order-``(k-1)`` Hilbert curves, one per
    quadrant, oriented so the whole path is a closed loop — its first and last cells
    are grid-adjacent, which suits a TSP cycle.
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")
    q = _quantize(coords, bits)
    x, y = q[:, 0], q[:, 1]
    m = bits - 1
    M = jnp.uint32(1 << m)
    half = M * M

    qx = (x >= M).astype(jnp.uint32)
    qy = (y >= M).astype(jnp.uint32)
    lx = x % M
    ly = y % M
    mm1 = M - jnp.uint32(1)

    # Visit quadrants up the left column then down the right: BL -> TL -> TR -> BR.
    # Left quadrants run a Hilbert curve rotated so it ascends; right quadrants run
    # the mirror so it descends, closing the loop along the bottom edge.
    left = qx == 0
    # Rotate local coords into the standard Hilbert frame (entry (0,0), exit (M-1,0)).
    # Left column ascends: rotate 90deg CW -> (ux, uy) = (ly, mm1 - lx).
    # Right column descends: rotate 90deg CCW -> (ux, uy) = (mm1 - ly, lx).
    ux = jnp.where(left, ly, mm1 - ly)
    uy = jnp.where(left, mm1 - lx, lx)
    sub = _hilbert_d(ux, uy, m) if m > 0 else jnp.zeros_like(x)

    rank = jnp.where(
        left,
        qy,                  # BL -> 0, TL -> 1
        jnp.uint32(3) - qy,  # TR -> 2, BR -> 3
    )
    # With both column quadrants rotated the same way, their endpoints already meet
    # across the shared edge (BL.exit adjacent to TL.entry, etc.) — no reversal needed,
    # and BR.exit closes back to BL.entry along the bottom.
    d = rank * half + sub
    return jnp.argsort(d).astype(jnp.int32)
