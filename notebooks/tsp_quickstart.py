import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # tspjax quickstart

        Load a small TSPLIB instance, build a starting tour with a space-filling
        curve, and polish it with 2-opt or 3-opt — all small enough to run on a laptop
        CPU. Dropdowns below pick the instance, the optimiser, the candidate
        neighbourhood (full / window / nearest / longest-edge), and the JAX backend
        (CPU, or GPU when one is present).
        """
    )
    return (mo,)


@app.cell
def _():
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    import tspjax
    from tspjax.construct import hilbert_tour, moore_tour, morton_tour
    from tspjax.solvers import (
        all_pairs,
        longest_edge,
        nearest,
        three_opt,
        two_opt,
        windowed,
    )

    return (
        all_pairs,
        hilbert_tour,
        jax,
        longest_edge,
        moore_tour,
        morton_tour,
        nearest,
        plt,
        three_opt,
        time,
        tspjax,
        two_opt,
        windowed,
    )


@app.cell(hide_code=True)
def _(jax, mo):
    # Small, symmetric, 2D instances — fast on CPU and plottable.
    problems = [
        "berlin52", "eil51", "st70", "eil76", "pr76", "rd100",
        "kroA100", "ch150", "tsp225", "a280", "lin318",
    ]
    problem_picker = mo.ui.dropdown(options=problems, value="berlin52", label="Problem")

    # Which local-search improver to run. "2-opt → 3-opt" warm-starts the (pricier)
    # 3-opt from a fast 2-opt local optimum.
    algos = ["2-opt", "3-opt", "2-opt → 3-opt"]
    algo_picker = mo.ui.dropdown(options=algos, value="2-opt", label="Algorithm")

    # CPU is always available; `default_backend()` reports the accelerator (gpu/tpu)
    # only when one is present — no exceptions, so the toggle degrades cleanly to a
    # single "cpu" option on a laptop.
    _accel = jax.default_backend()
    available_backends = ["cpu"] + ([_accel] if _accel != "cpu" else [])
    backend_picker = mo.ui.dropdown(
        options=available_backends,
        value=available_backends[0],
        label="JAX backend",
    )

    # Candidate neighbourhood: which moves the solver even considers each step.
    neighborhood_picker = mo.ui.dropdown(
        options=["full", "window", "nearest", "longest edge"],
        value="full",
        label="Neighborhood",
    )

    mo.vstack(
        [
            mo.hstack(
                [problem_picker, algo_picker, neighborhood_picker, backend_picker],
                justify="start",
            ),
            mo.md(
                "Pick an instance, an algorithm, and a backend "
                f"(detected: `{available_backends}`). Everything below re-runs on "
                "your choices."
            ),
        ]
    )
    return algo_picker, backend_picker, neighborhood_picker, problem_picker


@app.cell(hide_code=True)
def _(backend_picker, jax, mo, problem_picker, tspjax):
    device = jax.devices(backend_picker.value)[0]
    problem = tspjax.load(problem_picker.value)

    mo.md(
        f"""
        **Instance:** `{problem.name}` — {problem.dimension} cities,
        optimal tour length **{problem.best_known}**.
        Running on **`{device}`**.
        """
    )
    return device, problem


@app.cell(hide_code=True)
def _(mo, neighborhood_picker, problem):
    # Sizes the chosen neighbourhood: max segment length for "window", or k (candidates
    # per city) for "nearest" / "longest edge". Coarse `step` + `debounce` keep the
    # number of distinct XLA recompiles down (each size is a new candidate-grid shape).
    _n = problem.dimension
    size_slider = mo.ui.slider(
        start=2,
        stop=_n - 1,
        value=min(8, _n - 1),
        step=max(1, (_n - 1) // 25),
        label="Neighborhood size (window / k)",
        debounce=True,
        show_value=True,
    )
    (
        mo.md(
            "*Full search — every edge pair is a candidate. "
            "Pick another **Neighborhood** above to cap it.*"
        )
        if neighborhood_picker.value == "full"
        else size_slider
    )
    return (size_slider,)


@app.cell(hide_code=True)
def _(hilbert_tour, jax, moore_tour, morton_tour, problem):
    # Construction heuristics (space-filling curves) plus a random baseline. Morton
    # comes in two flavours depending on which axis supplies the high interleave bit.
    starts = {
        "hilbert": hilbert_tour(problem.coords),
        "morton (x,y)": morton_tour(problem.coords, order="xy"),
        "morton (y,x)": morton_tour(problem.coords, order="yx"),
        "moore": moore_tour(problem.coords),
        "random": jax.random.permutation(jax.random.PRNGKey(0), problem.dimension),
    }
    return (starts,)


@app.cell(hide_code=True)
def _(
    algo_picker,
    all_pairs,
    device,
    jax,
    longest_edge,
    mo,
    nearest,
    neighborhood_picker,
    problem,
    size_slider,
    starts,
    three_opt,
    time,
    two_opt,
    windowed,
):
    D = jax.device_put(problem.distances, device)

    # Build the candidate strategy from the neighbourhood dropdown + size slider.
    _nb = neighborhood_picker.value
    _k = int(size_slider.value)
    if _nb == "window":
        cand = windowed(_k)
    elif _nb == "nearest":
        cand = nearest(_k)
    elif _nb == "longest edge":
        cand = longest_edge(_k)
    else:
        cand = all_pairs

    # Dispatch the dropdown choice to a solver. "2-opt → 3-opt" feeds the 2-opt local
    # optimum into 3-opt as a (cheaper) warm start.
    def _solve(D, start):
        if algo_picker.value == "2-opt":
            return two_opt(D, start, candidates=cand)
        if algo_picker.value == "3-opt":
            return three_opt(D, start, candidates=cand)
        return three_opt(D, two_opt(D, start, candidates=cand), candidates=cand)

    results = {}
    for _name, _tour in starts.items():
        _start = jax.device_put(_tour, device)
        # warm up XLA compilation on this device, then time a clean run
        jax.block_until_ready(_solve(D, _start))
        _t0 = time.perf_counter()
        _opt = _solve(D, _start)
        jax.block_until_ready(_opt)
        _ms = (time.perf_counter() - _t0) * 1000
        results[_name] = {
            "tour": _opt,
            "start_length": float(problem.tour_length(_tour)),
            "length": float(problem.tour_length(_opt)),
            "ms": _ms,
        }

    _rows = "\n".join(
        f"| {name} | {r['start_length']:,.0f} | {r['length']:,.0f} | "
        f"{(r['length'] / problem.best_known - 1) * 100:+.2f}% | {r['ms']:.1f} |"
        for name, r in results.items()
    )
    _nlabel = _nb if _nb == "full" else f"{_nb} {_k}"
    mo.md(
    f"""
    ### {algo_picker.value} ({_nlabel}) — optimum {problem.best_known:,}

    | start | start length | final length | gap to optimum | wall time (ms) |
    |---|---|---|---|---|
    {_rows}
    """
    )
    return (results,)


@app.cell(hide_code=True)
def _(algo_picker, plt, problem, results, starts):
    # One row per starting heuristic: its initial tour (left) vs the optimised one
    # (right) — the two-column before/after, for every row of the table.
    _items = list(results.items())
    fig, _axes = plt.subplots(len(_items), 2, figsize=(8, 4 * len(_items)))
    _axes = _axes.reshape(len(_items), 2)
    for (_name, _r), (_ax0, _ax1) in zip(_items, _axes):
        problem.plot(tour=starts[_name], ax=_ax0)
        _ax0.set_title(f"{_name} start — {_r['start_length']:,.0f}", fontsize=9)
        problem.plot(tour=_r["tour"], ax=_ax1)
        _gap = (_r["length"] / problem.best_known - 1) * 100
        _ax1.set_title(
            f"after {algo_picker.value} — {_r['length']:,.0f} ({_gap:+.1f}%)",
            fontsize=9,
        )
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
