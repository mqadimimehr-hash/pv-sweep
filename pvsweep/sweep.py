"""Sweep definition, expansion, execution, CSV I/O and plotting helpers."""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

MODES = ("grid", "oat")


def expand_values(spec: Any) -> List[Any]:
    """Expand one parameter specification into a list of values.

    Accepted forms:
      * a list/tuple of explicit values: ``[1, 2, 3]``
      * a scalar (treated as a single value)
      * ``{"linspace": [start, stop, num]}``
      * ``{"logspace": [start_exp, stop_exp, num]}`` (base 10, like numpy)
      * ``{"geomspace": [start, stop, num]}``
      * ``{"range": [start, stop, step]}`` (stop inclusive within tolerance)
    """
    if isinstance(spec, Mapping):
        if len(spec) != 1:
            raise ValueError(f"parameter spec must have exactly one key, got {dict(spec)!r}")
        (kind, args), = spec.items()
        if kind in ("values", "list"):
            return list(args)
        if not isinstance(args, (list, tuple)) or len(args) != 3:
            raise ValueError(f"{kind!r} expects [start, stop, num/step], got {args!r}")
        a, b, c = (float(x) for x in args)
        if kind == "linspace":
            arr = np.linspace(a, b, _count(c))
        elif kind == "logspace":
            arr = np.logspace(a, b, _count(c))
        elif kind == "geomspace":
            arr = np.geomspace(a, b, _count(c))
        elif kind == "range":
            if c == 0 or (b - a) / c < 0:
                raise ValueError(f"invalid range {args!r}")
            n = int(math.floor((b - a) / c + 1e-9)) + 1
            arr = a + c * np.arange(n)
        else:
            raise ValueError(f"unknown spacing {kind!r}; use linspace, logspace, geomspace or range")
        return [float(x) for x in arr]
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("parameter value list is empty")
        return list(spec)
    return [spec]


def _count(x: float) -> int:
    n = int(x)
    if n != x or n < 1:
        raise ValueError(f"number of points must be a positive integer, got {x!r}")
    return n


@dataclass
class Sweep:
    """A parameter sweep.

    Parameters
    ----------
    parameters: mapping of parameter name -> value spec (see :func:`expand_values`).
    mode: ``"grid"`` (full Cartesian product) or ``"oat"`` (one-at-a-time around ``base``).
    base: fixed parameters passed to every run; in OAT mode also the reference point.
    backend: backend configuration, e.g. ``{"type": "single_diode"}``.
    """

    parameters: Dict[str, Any]
    mode: str = "grid"
    base: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=lambda: {"type": "single_diode"})

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not self.parameters:
            raise ValueError("a sweep needs at least one parameter")
        self.values = {k: expand_values(v) for k, v in self.parameters.items()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Sweep":
        unknown = set(d) - {"parameters", "mode", "base", "backend", "name", "description"}
        if unknown:
            raise ValueError(f"unknown sweep keys: {sorted(unknown)}")
        backend = d.get("backend") or {"type": "single_diode"}
        if isinstance(backend, str):
            backend = {"type": backend}
        return cls(parameters=dict(d.get("parameters") or {}), mode=d.get("mode", "grid"),
                   base=dict(d.get("base") or {}), backend=dict(backend))

    def points(self) -> List[Dict[str, Any]]:
        """Return the list of parameter dicts (base merged in) to evaluate."""
        names = list(self.values)
        pts: List[Dict[str, Any]] = []
        if self.mode == "grid":
            for combo in itertools.product(*(self.values[n] for n in names)):
                pts.append({**self.base, **dict(zip(names, combo))})
        else:
            ref = {n: self.base.get(n, self.values[n][0]) for n in names}
            for n in names:
                for v in self.values[n]:
                    pts.append({**self.base, **ref, n: v, "varied": n})
        return pts

    def __len__(self) -> int:
        if self.mode == "grid":
            return int(np.prod([len(v) for v in self.values.values()]))
        return sum(len(v) for v in self.values.values())


def load_sweep(source: Union[str, Path, Mapping[str, Any]]) -> Sweep:
    """Load a :class:`Sweep` from a YAML file path or a dict."""
    if isinstance(source, Mapping):
        return Sweep.from_dict(source)
    import yaml

    with open(source, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ValueError(f"{source}: top level of a sweep file must be a mapping")
    return Sweep.from_dict(data)


def run_sweep(sweep: Union[Sweep, Mapping[str, Any], str, Path], backend: Any = None,
              progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
              stop_on_error: bool = False) -> List[Dict[str, Any]]:
    """Evaluate every point of a sweep and return a list of result rows.

    Each row holds the input parameters, the backend outputs, and a ``status``
    column (``"ok"`` or ``"error"``; failing rows also carry ``error``).
    """
    from .backends import get_backend

    if not isinstance(sweep, Sweep):
        sweep = load_sweep(sweep)
    be = backend if backend is not None else get_backend(sweep.backend)
    pts = sweep.points()
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(pts):
        params = {k: v for k, v in p.items() if k != "varied"}
        row = dict(p)
        try:
            row.update(be(params))
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - recorded per row
            if stop_on_error:
                raise
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if progress:
            progress(i + 1, len(pts), row)
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Union[str, Path]) -> Path:
    """Write result rows to CSV (union of all keys, first-seen order)."""
    cols: List[str] = []
    for r in rows:
        cols.extend(k for k in r if k not in cols)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def read_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a results CSV, converting numeric-looking cells to float."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _num(v: Any) -> Any:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# --------------------------------------------------------------------------- plotting

def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("plotting needs matplotlib: pip install 'pv-sweep[plot]'") from exc
    return plt


def _ok(rows: Iterable[Mapping[str, Any]], *cols: str) -> List[Mapping[str, Any]]:
    return [r for r in rows if r.get("status", "ok") == "ok"
            and all(isinstance(r.get(c), (int, float)) for c in cols)]


def plot_line(rows, x: str, y: str, group: Optional[str] = None, ax=None, logx: bool = False,
              logy: bool = False):
    """1D line plot of ``y`` vs ``x``; one line per distinct value of ``group``."""
    plt = _plt()
    rows = _ok(rows, x, y)
    if not rows:
        raise ValueError(f"no numeric data for columns {x!r} and {y!r}")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.6))
    groups: Dict[Any, List[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get(group) if group else None, []).append(r)
    for g, rs in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        rs = sorted(rs, key=lambda r: r[x])
        ax.plot([r[x] for r in rs], [r[y] for r in rs], marker="o", ms=3,
                label=None if group is None else f"{group}={g:g}" if isinstance(g, float) else f"{group}={g}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if group:
        ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return ax


def plot_heatmap(rows, x: str, y: str, z: str, ax=None, cmap: str = "viridis"):
    """2D heatmap of ``z`` over the (``x``, ``y``) grid. Duplicate cells are averaged."""
    plt = _plt()
    rows = _ok(rows, x, y, z)
    if not rows:
        raise ValueError(f"no numeric data for columns {x!r}, {y!r}, {z!r}")
    xs = sorted({r[x] for r in rows})
    ys = sorted({r[y] for r in rows})
    total = np.zeros((len(ys), len(xs)))
    count = np.zeros_like(total)
    for r in rows:
        i, j = ys.index(r[y]), xs.index(r[x])
        total[i, j] += r[z]
        count[i, j] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(xs)), [f"{v:.3g}" for v in xs], rotation=45)
    ax.set_yticks(range(len(ys)), [f"{v:.3g}" for v in ys])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.figure.colorbar(im, ax=ax, label=z)
    return ax
