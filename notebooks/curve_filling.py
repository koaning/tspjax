# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "wigglystuff>=0.5",
#     # tspjax isn't on PyPI yet — install it from git:
#     #   "tspjax @ git+https://github.com/koaning/tsplib.git",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # Space-filling curves, by eye

        A **space-filling curve** threads a single 1-D path through every cell of a 2-D
        grid. `tspjax`'s tour constructors use that path as a heuristic tour: quantise
        the cities onto a `2**bits`-per-axis grid, read off each city's position *along*
        the curve, and `argsort`. The tour is just *"visit the cities in curve order."*

        Different curves trade off **locality** differently — how well points that are
        close *along the curve* stay close *in the plane*. Drag the **rank window** below
        to highlight a contiguous stretch of the curve and watch where those points land:

        * **Hilbert / Moore** keep the window in one tight blob — no long jumps.
        * **Morton** (Z-order) periodically *teleports* across the plane: the window
          splinters into far-apart pieces.

        The **resolution** slider is the `bits` argument the constructors take. Drop it
        low and cities collapse into shared grid cells (ties broken arbitrarily), coarsening
        the path; raise it and the curve resolves every city. The **Moore** curve is a
        closed loop, which is why the rank window wraps seamlessly — like a TSP tour.
        """
    )
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import tspjax
    from tspjax.construct import hilbert_tour, moore_tour, morton_tour
    from wigglystuff import CircularRangeSlider

    # Each curve maps to a constructor call and a signature colour (shared by the plot
    # and the circular slider). Morton needs its `order` flag baked in here.
    CURVES = {
        "hilbert": (lambda c, bits: hilbert_tour(c, bits=bits), "#4c78a8"),
        "morton (x,y)": (lambda c, bits: morton_tour(c, order="xy", bits=bits), "#f58518"),
        "morton (y,x)": (lambda c, bits: morton_tour(c, order="yx", bits=bits), "#e45756"),
        "moore": (lambda c, bits: moore_tour(c, bits=bits), "#54a24b"),
    }

    # The default "instance": a dense, fixed random cloud. Dense enough that the curve's
    # shape — and Morton's jumps — read clearly. Seeded so it's stable across re-runs.
    RANDOM = "random (dense)"
    RANDOM_COORDS = np.random.default_rng(0).random((15000, 2))
    return CURVES, CircularRangeSlider, RANDOM, RANDOM_COORDS, np, plt, tspjax


@app.cell(hide_code=True)
def _(CURVES, RANDOM, mo):
    # Default to the dense random cloud; the rest are small/medium symmetric TSPLIB
    # instances, which plot cleanly and build instantly.
    problems = [
        RANDOM,
        "berlin52", "eil51", "st70", "eil76", "pr76", "rd100",
        "kroA100", "ch150", "tsp225", "a280", "lin318",
    ]
    problem_picker = mo.ui.dropdown(options=problems, value=RANDOM, label="Instance")
    curve_picker = mo.ui.dropdown(
        options=list(CURVES), value="hilbert", label="Curve"
    )
    bits_slider = mo.ui.slider(
        start=2, stop=16, value=8, label="Resolution (bits → 2**bits grid)",
        show_value=True,
    )

    mo.hstack([problem_picker, curve_picker, bits_slider], justify="start")
    return bits_slider, curve_picker, problem_picker


@app.cell(hide_code=True)
def _(
    CURVES,
    RANDOM,
    RANDOM_COORDS,
    bits_slider,
    curve_picker,
    np,
    problem_picker,
    tspjax,
):
    if problem_picker.value == RANDOM:
        coords = RANDOM_COORDS
        name = f"random · {coords.shape[0]} pts"
    else:
        _problem = tspjax.load(problem_picker.value)
        coords = np.asarray(_problem.coords)
        name = _problem.name
    n = coords.shape[0]

    _constructor, color = CURVES[curve_picker.value]
    tour = np.asarray(_constructor(coords, int(bits_slider.value)))
    ordered = coords[tour]  # cities in curve order, shape (n, 2)
    return color, n, name, ordered


@app.cell(hide_code=True)
def _(CircularRangeSlider, color, mo, n):
    # Sized to the tour and recreated whenever `n` (or the colour) changes. The value is
    # a (lo, hi) pair that may wrap (hi < lo) — matching the Moore curve's closed loop.
    rank_window = mo.ui.anywidget(
        CircularRangeSlider(
            start=0, stop=n - 1, step=1, value=(0, max(1, n // 4)),
            size=240, color=color, label="Rank window",
        )
    )
    return (rank_window,)


@app.cell(hide_code=True)
def _(color, curve_picker, mo, n, name, ordered, plt, rank_window):
    lo, hi = (int(v) for v in rank_window.value["value"])
    # Contiguous slice along the curve; wraps around the end when hi < lo.
    idx = list(range(lo, n)) + list(range(0, hi)) if hi < lo else list(range(lo, hi))
    wrapped = hi < lo

    fig, ax = plt.subplots(figsize=(6, 6))
    # Full curve in faint grey, closed into a cycle like a tour.
    _loop = list(range(n)) + [0]
    ax.plot(ordered[_loop, 0], ordered[_loop, 1], "-", color="0.8", lw=1, zorder=1)
    ax.scatter(ordered[:, 0], ordered[:, 1], s=10, color="0.7", zorder=2)
    # Highlight the selected rank window in the curve's colour.
    if len(idx) >= 1:
        _w = ordered[idx]
        ax.plot(_w[:, 0], _w[:, 1], "-", color=color, lw=2, zorder=3)
        ax.scatter(_w[:, 0], _w[:, 1], s=22, color=color, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"{name} · {curve_picker.value} · ranks {lo}–{hi}"
        f"{' (wraps)' if wrapped else ''} of {n}"
    )
    fig.tight_layout()

    # Circular rank slider to the left of the map.
    mo.hstack([rank_window, fig], justify="start", align="center")
    return


if __name__ == "__main__":
    app.run()
