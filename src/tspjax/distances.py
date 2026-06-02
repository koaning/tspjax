"""JAX distance kernels for the coordinate-based TSPLIB edge-weight types.

Each kernel maps a ``(n, d)`` coordinate array to a dense ``(n, n)`` distance
matrix, applying the integer rounding rule mandated by the TSPLIB spec so that
tour lengths match the published optimal values. Matrices are returned as
``float32`` (rounded to whole numbers) — convenient for GPU math while staying
numerically equal to the integer weights.
"""

from __future__ import annotations

import jax.numpy as jnp

# Earth radius and PI exactly as specified in the TSPLIB documentation, so GEO
# distances reproduce the reference values bit-for-bit.
_GEO_PI = 3.141592
_GEO_RRR = 6378.388


def _pairwise_deltas(coords: jnp.ndarray) -> jnp.ndarray:
    """Return the ``(n, n, d)`` array of coordinate differences."""
    return coords[:, None, :] - coords[None, :, :]


def euclidean_2d(coords: jnp.ndarray) -> jnp.ndarray:
    """EUC_2D / EUC_3D: rounded Euclidean distance (``nint``)."""
    d = jnp.sqrt(jnp.sum(_pairwise_deltas(coords) ** 2, axis=-1))
    return jnp.floor(d + 0.5).astype(jnp.float32)


def ceil_2d(coords: jnp.ndarray) -> jnp.ndarray:
    """CEIL_2D: Euclidean distance rounded up."""
    d = jnp.sqrt(jnp.sum(_pairwise_deltas(coords) ** 2, axis=-1))
    return jnp.ceil(d).astype(jnp.float32)


def att(coords: jnp.ndarray) -> jnp.ndarray:
    """ATT: the pseudo-Euclidean distance used by the att48/att532 instances."""
    sq = jnp.sum(_pairwise_deltas(coords) ** 2, axis=-1)
    rij = jnp.sqrt(sq / 10.0)
    tij = jnp.round(rij)
    # If the rounded value undershoots, bump it up by one (TSPLIB rule).
    dij = jnp.where(tij < rij, tij + 1.0, tij)
    return dij.astype(jnp.float32)


def _geo_radians(coords: jnp.ndarray) -> jnp.ndarray:
    """Convert ``DDD.MM`` degrees-and-minutes coordinates to radians."""
    deg = jnp.trunc(coords)
    minutes = coords - deg
    return _GEO_PI * (deg + 5.0 * minutes / 3.0) / 180.0


def geo(coords: jnp.ndarray) -> jnp.ndarray:
    """GEO: great-circle distance on the TSPLIB reference sphere."""
    rad = _geo_radians(coords)
    lat, lon = rad[:, 0], rad[:, 1]
    q1 = jnp.cos(lon[:, None] - lon[None, :])
    q2 = jnp.cos(lat[:, None] - lat[None, :])
    q3 = jnp.cos(lat[:, None] + lat[None, :])
    inner = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    inner = jnp.clip(inner, -1.0, 1.0)  # guard acos against tiny overshoots
    d = _GEO_RRR * jnp.arccos(inner) + 1.0
    return jnp.trunc(d).astype(jnp.float32)


# Maps EDGE_WEIGHT_TYPE -> kernel. EUC_3D reuses the 2D kernel (it just sums an
# extra coordinate axis), MAX_2D/MAN_2D could be added here later.
_KERNELS = {
    "EUC_2D": euclidean_2d,
    "EUC_3D": euclidean_2d,
    "CEIL_2D": ceil_2d,
    "ATT": att,
    "GEO": geo,
}


def distance_matrix(coords: jnp.ndarray, edge_weight_type: str) -> jnp.ndarray:
    """Build the ``(n, n)`` distance matrix for a coordinate edge-weight type."""
    if edge_weight_type not in _KERNELS:
        raise ValueError(
            f"No distance kernel for EDGE_WEIGHT_TYPE {edge_weight_type!r}; "
            f"supported: {sorted(_KERNELS)}"
        )
    kernel = _KERNELS[edge_weight_type]
    D = kernel(jnp.asarray(coords, dtype=jnp.float32))
    # Force exact zeros on the diagonal: the GEO formula's "+1.0" term otherwise
    # yields a self-distance of 1. Harmless for tour length (tours never visit
    # i->i) but wrong if anyone reads the diagonal.
    return D - jnp.diag(jnp.diagonal(D))


def supported_types() -> list[str]:
    """Coordinate edge-weight types with a distance kernel."""
    return sorted(_KERNELS)
