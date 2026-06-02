"""Matplotlib tour plotting.

Kept free of jax/Problem imports so it can be reused with raw coordinate arrays.
Returns the matplotlib ``Axes``, which renders inline in marimo and Jupyter.
"""

from __future__ import annotations

import numpy as np


def plot_tour(coords, tour=None, ax=None, title=None, **scatter_kwargs):
    """Scatter the cities and, if ``tour`` is given, draw the closed route.

    Args:
        coords: ``(n, 2)`` array of city coordinates.
        tour: optional ``(n,)`` permutation of ``0..n-1`` to draw as a cycle.
        ax: optional matplotlib ``Axes`` to draw into; created if omitted.
        title: optional plot title.

    Returns:
        The matplotlib ``Axes`` used.
    """
    import matplotlib.pyplot as plt

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must be (n, 2); got shape {coords.shape}")

    if ax is None:
        _, ax = plt.subplots()

    if tour is not None:
        order = np.asarray(tour).astype(int)
        loop = np.concatenate([order, order[:1]])  # close the cycle
        ax.plot(coords[loop, 0], coords[loop, 1], "-", color="tab:blue", lw=1, zorder=1)

    scatter_kwargs.setdefault("s", 12)
    scatter_kwargs.setdefault("color", "tab:red")
    ax.scatter(coords[:, 0], coords[:, 1], zorder=2, **scatter_kwargs)
    ax.set_aspect("equal", adjustable="datalim")
    if title:
        ax.set_title(title)
    return ax
