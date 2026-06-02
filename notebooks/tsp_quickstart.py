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
        curve, and polish it with windowed 2-opt or 3-opt — all small enough to run
        on a laptop CPU. Dropdowns below pick the instance, the optimiser, and the
        JAX backend (CPU, or GPU when one is present).
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
    from tspjax.solvers import three_opt, two_opt

    return (
        hilbert_tour,
        jax,
        moore_tour,
        morton_tour,
        plt,
        three_opt,
        time,
        tspjax,
        two_opt,
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

    # Tick to reveal a window-size slider; unchecked means the full window is used.
    use_window = mo.ui.checkbox(label="Limit window")

    mo.vstack(
        [
            mo.hstack(
                [problem_picker, algo_picker, backend_picker, use_window],
                justify="start",
            ),
            mo.md(
                "Pick an instance, an algorithm, and a backend "
                f"(detected: `{available_backends}`). Everything below re-runs on "
                "your choices."
            ),
        ]
    )
    return algo_picker, backend_picker, problem_picker, use_window


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
def _(mo, problem, use_window):
    # Window slider, sized to the tour. Coarse `step` + `debounce` keep the number of
    # distinct XLA recompiles down (each window value is a new candidate-grid shape).
    _n = problem.dimension
    window_slider = mo.ui.slider(
        start=2,
        stop=_n - 1,
        value=_n // 10,
        step=max(1, (_n - 1) // 25),
        label="Search window (max segment length)",
        debounce=True,
        show_value=True,
    )
    (
        window_slider
        if use_window.value
        else mo.md(
            "*Full window — every city pair is a candidate. "
            "Tick **Limit window** above to cap it.*"
        )
    )
    return (window_slider,)


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
    device,
    jax,
    mo,
    problem,
    starts,
    three_opt,
    time,
    two_opt,
    use_window,
    window_slider,
):
    D = jax.device_put(problem.distances, device)
    window = int(window_slider.value) if use_window.value else None

    # Dispatch the dropdown choice to a solver. "2-opt → 3-opt" feeds the 2-opt local
    # optimum into 3-opt as a (cheaper) warm start.
    def _solve(D, start):
        if algo_picker.value == "2-opt":
            return two_opt(D, start, window=window)
        if algo_picker.value == "3-opt":
            return three_opt(D, start, window=window)
        return three_opt(D, two_opt(D, start, window=window), window=window)

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
    _wlabel = f"window {window}" if window is not None else "full window"
    mo.md(
    f"""
    ### {algo_picker.value} ({_wlabel}) — optimum {problem.best_known:,}

    | start | start length | final length | gap to optimum | wall time (ms) |
    |---|---|---|---|---|
    {_rows}
    """
    )
    return (results,)


@app.cell(hide_code=True)
def _(algo_picker, plt, problem, results, starts):
    # Visualise one constructor: messy curve start vs the polished tour.
    which = "morton (x,y)"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5))
    problem.plot(tour=starts[which], ax=ax_a)
    ax_a.set_title(f"{which} start — {float(problem.tour_length(starts[which])):,.0f}")
    problem.plot(tour=results[which]["tour"], ax=ax_b)
    ax_b.set_title(f"after {algo_picker.value} — {results[which]['length']:,.0f}")
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
