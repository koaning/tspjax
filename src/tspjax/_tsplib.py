"""Low-level parser for the TSPLIB file format.

This module is intentionally dependency-free (pure Python + numpy): it turns the
raw text of a ``.tsp`` file into a plain :class:`ParsedInstance` of header fields
plus either node coordinates or an explicit weight matrix. Turning that into a
:class:`~tspjax.problem.Problem` (and into jax arrays) happens one layer up, in
:mod:`tspjax.loader`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Header keys we care about; everything else is kept verbatim in ``ParsedInstance.spec``.
_INT_KEYS = {"DIMENSION", "CAPACITY"}


@dataclass
class ParsedInstance:
    """The raw contents of a TSPLIB file, before any jax/distance machinery."""

    name: str
    spec: dict[str, str] = field(default_factory=dict)
    coords: np.ndarray | None = None  # (n, d) float64, in node order
    matrix: np.ndarray | None = None  # (n, n) float64 explicit weights
    display_coords: np.ndarray | None = None  # (n, 2) for plotting only

    @property
    def dimension(self) -> int:
        return int(self.spec["DIMENSION"])

    @property
    def edge_weight_type(self) -> str:
        return self.spec.get("EDGE_WEIGHT_TYPE", "")

    @property
    def comment(self) -> str:
        return self.spec.get("COMMENT", "")


def parse(text: str) -> ParsedInstance:
    """Parse the full text of a TSPLIB ``.tsp`` file."""
    lines = text.splitlines()
    spec: dict[str, str] = {}

    # First pass: read the ``KEY : VALUE`` header up to the first data section.
    section_starts = {
        "NODE_COORD_SECTION",
        "EDGE_WEIGHT_SECTION",
        "DISPLAY_DATA_SECTION",
        "DEPOT_SECTION",
    }
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        head = line.split(":", 1)[0].strip().upper()
        if head in section_starts or head == "EOF":
            i -= 1  # leave the section header for the second pass
            break
        if ":" in line:
            key, value = line.split(":", 1)
            spec[key.strip().upper()] = value.strip()

    name = spec.get("NAME", "").split(":")[0].strip() or "unnamed"
    dimension = int(spec["DIMENSION"])
    inst = ParsedInstance(name=name, spec=spec)

    # Second pass: consume the data sections.
    while i < len(lines):
        header = lines[i].strip()
        i += 1
        if not header:
            continue
        head = header.split(":", 1)[0].strip().upper()
        if head == "EOF":
            break
        if head == "NODE_COORD_SECTION":
            inst.coords, i = _read_coords(lines, i, dimension)
        elif head == "DISPLAY_DATA_SECTION":
            inst.display_coords, i = _read_coords(lines, i, dimension)
        elif head == "EDGE_WEIGHT_SECTION":
            fmt = spec.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").strip().upper()
            inst.matrix, i = _read_matrix(lines, i, dimension, fmt)
        elif head == "DEPOT_SECTION":
            i = _skip_until_terminator(lines, i)

    return inst


def _read_coords(lines: list[str], i: int, n: int) -> tuple[np.ndarray, int]:
    """Read ``n`` ``index x y [z]`` rows; coordinates returned in node order."""
    coords: list[list[float]] = []
    while len(coords) < n and i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        if parts[0].upper() in {"EOF", "-1"}:
            break
        # parts[0] is the 1-based node index; the rest are coordinates.
        coords.append([float(p) for p in parts[1:]])
    return np.asarray(coords, dtype=np.float64), i


def _read_matrix(lines: list[str], i: int, n: int, fmt: str) -> tuple[np.ndarray, int]:
    """Read a flat stream of weights and fold it into an ``(n, n)`` matrix per ``fmt``."""
    values: list[float] = []
    while i < len(lines):
        stripped = lines[i].strip()
        head = stripped.split(":", 1)[0].strip().upper()
        if head in {
            "EOF",
            "DISPLAY_DATA_SECTION",
            "NODE_COORD_SECTION",
            "DEPOT_SECTION",
        }:
            break
        values.extend(float(p) for p in lines[i].split())
        i += 1
    return _fold_matrix(np.asarray(values, dtype=np.float64), n, fmt), i


def _fold_matrix(values: np.ndarray, n: int, fmt: str) -> np.ndarray:
    """Reconstruct a symmetric ``(n, n)`` matrix from a flat weight stream."""
    m = np.zeros((n, n), dtype=np.float64)
    it = iter(values)

    if fmt == "FULL_MATRIX":
        return values.reshape(n, n)
    if fmt == "UPPER_ROW":  # strict upper triangle, row by row (no diagonal)
        for r in range(n):
            for c in range(r + 1, n):
                m[r, c] = m[c, r] = next(it)
    elif fmt == "UPPER_DIAG_ROW":  # upper triangle including diagonal
        for r in range(n):
            for c in range(r, n):
                m[r, c] = m[c, r] = next(it)
    elif fmt == "LOWER_ROW":  # strict lower triangle, row by row (no diagonal)
        for r in range(n):
            for c in range(r):
                m[r, c] = m[c, r] = next(it)
    elif fmt == "LOWER_DIAG_ROW":  # lower triangle including diagonal
        for r in range(n):
            for c in range(r + 1):
                m[r, c] = m[c, r] = next(it)
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt!r}")
    return m


def _skip_until_terminator(lines: list[str], i: int) -> int:
    """Advance past a section terminated by ``-1`` or ``EOF``."""
    while i < len(lines):
        tok = lines[i].strip().split()
        i += 1
        if not tok:
            continue
        if tok[0].upper() == "EOF" or tok[0] == "-1":
            break
    return i


def parse_solutions(text: str) -> dict[str, int]:
    """Parse the ``solutions`` file (``name : length [ (annotation)]``)."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        for token in rest.split():
            try:
                out[name.strip()] = int(token)
                break
            except ValueError:
                continue
    return out
