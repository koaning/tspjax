# Notebooks

Interactive [marimo](https://marimo.io) notebooks demoing `tspjax`.

## `tsp_quickstart.py`

Pick a small TSPLIB instance from a dropdown (all run in well under a second on a
laptop CPU), build starting tours with the space-filling-curve constructors
(`hilbert_tour`, `morton_tour`, `moore_tour`), polish them with windowed `two_opt`
(final length, gap-to-optimum, and wall time per constructor), and plot the
before/after. It's the package-level counterpart to the GPU-benchmark `tsp-demo.py`
at the repo root.

### Run it

```bash
# edit interactively
uv run --extra dev marimo edit notebooks/tsp_quickstart.py

# or run as a read-only app
uv run --extra dev marimo run notebooks/tsp_quickstart.py
```

### Run it elsewhere (molab / a fresh env)

`tspjax` isn't on PyPI yet, so install it straight from GitHub with `uv`:

```bash
uv pip install "git+https://github.com/koaning/tsplib.git"
```

In a [molab](https://marimo.io) notebook, add it as a dependency the same way:

```bash
uv add "git+https://github.com/koaning/tsplib.git"
```

### CPU / GPU toggle

The notebook has a **backend dropdown** near the top. The options are built from
`jax.default_backend()`:

- On a CPU-only machine it lists just `cpu`.
- On a machine with a GPU it also lists `gpu`, and picking it runs the *same* code on
  the GPU (inputs are moved with `jax.device_put`).

To force a backend for a whole process instead (e.g. to benchmark CPU on a GPU box),
set the env var before launching:

```bash
JAX_PLATFORMS=cpu uv run --extra dev marimo run notebooks/tsp_quickstart.py
```
