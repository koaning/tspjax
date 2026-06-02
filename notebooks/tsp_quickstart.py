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
        curve, and polish it with windowed 2-opt — all small enough to run on a
        laptop CPU. A backend toggle below lets you flip the same code between CPU
        and GPU when a GPU is present.
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
    from tspjax.solvers import two_opt

    return (
        hilbert_tour,
        jax,
        jnp,
        moore_tour,
        morton_tour,
        plt,
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

    mo.vstack(
        [
            mo.hstack([problem_picker, backend_picker], justify="start"),
            mo.md(
                "Pick an instance and a backend "
                f"(detected: `{available_backends}`). Everything below re-runs on "
                "your choices."
            ),
        ]
    )
    return backend_picker, problem_picker


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
def _(hilbert_tour, jnp, mo, moore_tour, morton_tour, problem):
    # Three construction heuristics: order cities along a space-filling curve.
    starts = {
        "hilbert": hilbert_tour(problem.coords),
        "morton": morton_tour(problem.coords),
        "moore": moore_tour(problem.coords),
    }
    identity_len = float(problem.tour_length(jnp.arange(problem.dimension)))

    _rows = "\n".join(
        f"| {name} | {float(problem.tour_length(tour)):,.0f} |"
        for name, tour in starts.items()
    )

    mo.md(
    f"""
    ### Starting tours

    The identity tour `0,1,2,…` is length **{identity_len:,.0f}**. A curve-ordered
    start is far better before any optimisation runs:

    | constructor | start length |
    |---|---|
    {_rows}
    """
    )
    return (starts,)


@app.cell(hide_code=True)
def _(device, jax, mo, problem, starts, time, two_opt):
    D = jax.device_put(problem.distances, device)

    results = {}
    for _name, _tour in starts.items():
        _start = jax.device_put(_tour, device)
        # warm up XLA compilation on this device, then time a clean run
        jax.block_until_ready(two_opt(D, _start))
        _t0 = time.perf_counter()
        _opt = two_opt(D, _start)
        jax.block_until_ready(_opt)
        _ms = (time.perf_counter() - _t0) * 1000
        results[_name] = {
            "tour": _opt,
            "length": float(problem.tour_length(_opt)),
            "ms": _ms,
        }

    _rows = "\n".join(
        f"| {name} | {r['length']:,.0f} | "
        f"{(r['length'] / problem.best_known - 1) * 100:+.2f}% | {r['ms']:.1f} |"
        for name, r in results.items()
    )
    mo.md(
    f"""
    ### After windowed 2-opt (full window)

    | start | final length | gap to optimum | wall time (ms) |
    |---|---|---|---|
    {_rows}
    """
    )
    return (results,)


@app.cell(hide_code=True)
def _(plt, problem, results, starts):
    # Visualise one constructor: messy curve start vs the polished 2-opt tour.
    which = "hilbert"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5))
    problem.plot(tour=starts[which], ax=ax_a)
    ax_a.set_title(f"{which} start — {float(problem.tour_length(starts[which])):,.0f}")
    problem.plot(tour=results[which]["tour"], ax=ax_b)
    ax_b.set_title(f"after 2-opt — {results[which]['length']:,.0f}")
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
