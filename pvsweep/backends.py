"""Simulation backends.

A backend is any callable ``backend(params: dict) -> dict`` returning output
quantities. Two are built in:

* :class:`SingleDiodeBackend` - analytic single-diode model, works out of the box.
* :class:`CommandBackend` - wraps any command-line simulator via subprocess.
"""

from __future__ import annotations

import csv
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np

K_B = 1.380649e-23      # J/K
Q_E = 1.602176634e-19   # C


# --------------------------------------------------------------------------- single diode

def single_diode(Jph: float = 35.0, J0: float = 1e-9, n: float = 1.3, Rs: float = 1.0,
                 Rsh: float = 1e3, T: float = 300.0, Pin: float = 100.0,
                 points: int = 4000) -> Dict[str, float]:
    """Evaluate the single-diode model under illumination.

    ``J = Jph - J0 [exp((V + J Rs) / (n Vt)) - 1] - (V + J Rs) / Rsh``

    Units: current densities (``Jph``, ``J0``) in mA/cm^2, resistances
    (``Rs``, ``Rsh``) in ohm*cm^2, temperature ``T`` in K, incident power
    ``Pin`` in mW/cm^2. ``J0`` is used as given (no built-in temperature
    scaling). Returns Voc [V], Jsc [mA/cm^2], FF [%], PCE [%], Vmpp, Jmpp, Pmax.

    The implicit equation is solved exactly by parametrising on the junction
    voltage ``Vd = V + J Rs``: for each ``Vd`` the current is explicit and
    ``V = Vd - J Rs``.
    """
    for name, val in (("J0", J0), ("n", n), ("T", T), ("Rsh", Rsh), ("Pin", Pin)):
        if not val > 0:
            raise ValueError(f"{name} must be > 0, got {val!r}")
    if Rs < 0 or Jph < 0:
        raise ValueError("Rs and Jph must be >= 0")
    vt = K_B * T / Q_E
    nvt = n * vt
    vd_max = nvt * np.log(Jph / J0 + 1.0) + 5 * nvt
    vd = np.linspace(0.0, vd_max, int(points))
    j = Jph - J0 * np.expm1(vd / nvt) - vd / Rsh          # mA/cm^2
    v = vd - j * Rs * 1e-3                                  # V (mA -> A)

    # V(Vd) is monotonic increasing, so interpolation is well defined.
    jsc = float(np.interp(0.0, v, j))
    if j[-1] > 0:
        raise RuntimeError("J did not cross zero; increase the voltage range")
    idx = np.nonzero(j <= 0)[0][0]
    voc = float(np.interp(0.0, [j[idx], j[idx - 1]], [v[idx], v[idx - 1]])) if idx > 0 else float(v[0])
    p = v * j
    mask = (v >= 0) & (j >= 0)
    if jsc <= 0 or voc <= 0 or not mask.any():
        return {"Voc": max(voc, 0.0), "Jsc": max(jsc, 0.0), "FF": 0.0, "PCE": 0.0,
                "Vmpp": 0.0, "Jmpp": 0.0, "Pmax": 0.0}
    k = int(np.argmax(np.where(mask, p, -np.inf)))
    pmax = float(p[k])
    return {
        "Voc": voc,
        "Jsc": jsc,
        "FF": 100.0 * pmax / (voc * jsc),
        "PCE": 100.0 * pmax / Pin,
        "Vmpp": float(v[k]),
        "Jmpp": float(j[k]),
        "Pmax": pmax,
    }


class SingleDiodeBackend:
    """Backend wrapping :func:`single_diode`.

    Parameter names are matched case-insensitively with a few aliases
    (``temperature`` -> ``T``, ``n_ideality`` -> ``n``). Sweep parameters that
    are not model inputs raise an error unless ``ignore_unknown=True``.
    """

    ALIASES = {"jph": "Jph", "j0": "J0", "n": "n", "ideality": "n", "n_ideality": "n",
               "rs": "Rs", "rsh": "Rsh", "t": "T", "temperature": "T", "pin": "Pin"}

    def __init__(self, ignore_unknown: bool = False, **defaults: Any) -> None:
        self.ignore_unknown = ignore_unknown
        self.defaults = self._map(defaults)

    def _map(self, params: Mapping[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, val in params.items():
            key = self.ALIASES.get(k.lower())
            if key is None:
                if self.ignore_unknown:
                    continue
                raise KeyError(f"single_diode backend does not know parameter {k!r} "
                               f"(known: Jph, J0, n, Rs, Rsh, T/temperature, Pin)")
            out[key] = float(val)
        return out

    def __call__(self, params: Mapping[str, Any]) -> Dict[str, float]:
        return single_diode(**{**self.defaults, **self._map(params)})


# --------------------------------------------------------------------------- command

class CommandBackend:
    """Run an external command per sweep point and parse its results.

    Parameters
    ----------
    command:
        Template string (or list of template tokens) rendered with
        :meth:`str.format`. Available fields: every sweep parameter, plus
        ``{output}`` (result file path), ``{workdir}`` (per-run directory),
        ``{index}`` and ``{python}`` (the current interpreter). A string is
        split with :func:`shlex.split` *before* formatting, so parameter values
        are never re-interpreted by a shell.
    result_file:
        Template for the file to parse (default ``{workdir}/result.txt``). Use
        ``"-"`` to parse the command's stdout instead.
    regex:
        Mapping output name -> regex with one capture group, searched in the
        result text (last match wins).
    csv_columns:
        Mapping output name -> CSV column name; values are taken from the
        last data row (or ``csv_row`` index).
    timeout:
        Per-run timeout in seconds.
    workdir:
        Base directory for per-run folders (a temp dir if omitted).
    input_template / input_file:
        Optional text template rendered with the same fields and written to
        ``input_file`` (relative to the run dir) before the command runs.
    """

    def __init__(self, command: Union[str, Sequence[str]], result_file: Optional[str] = None,
                 regex: Optional[Mapping[str, str]] = None,
                 csv_columns: Optional[Mapping[str, str]] = None, csv_row: int = -1,
                 timeout: Optional[float] = None, workdir: Optional[str] = None,
                 input_template: Optional[str] = None, input_file: str = "input.txt",
                 env: Optional[Mapping[str, str]] = None, cwd_is_workdir: bool = True) -> None:
        if not regex and not csv_columns:
            raise ValueError("CommandBackend needs `regex` or `csv_columns` to parse results")
        if regex and csv_columns:
            raise ValueError("use either `regex` or `csv_columns`, not both")
        self.tokens = shlex.split(command) if isinstance(command, str) else list(command)
        self.result_file = result_file or "{workdir}/result.txt"
        self.regex = {k: re.compile(p) for k, p in (regex or {}).items()}
        for k, p in self.regex.items():
            if p.groups < 1:
                raise ValueError(f"regex for {k!r} needs a capture group")
        self.csv_columns = dict(csv_columns or {})
        self.csv_row = csv_row
        self.timeout = timeout
        self.base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="pvsweep-"))
        self.input_template = input_template
        self.input_file = input_file
        self.env = dict(env) if env else None
        self.cwd_is_workdir = cwd_is_workdir
        self._count = 0

    def render(self, params: Mapping[str, Any], index: int) -> Dict[str, Any]:
        run_dir = self.base / f"run_{index:05d}"
        fields = {**params, "workdir": str(run_dir), "index": index, "python": sys.executable}
        out = self.result_file.format(**fields) if self.result_file != "-" else "-"
        fields["output"] = out
        argv = [t.format(**fields) for t in self.tokens]
        return {"argv": argv, "run_dir": run_dir, "output": out, "fields": fields}

    def __call__(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        r = self.render(params, self._count)
        self._count += 1
        r["run_dir"].mkdir(parents=True, exist_ok=True)
        if self.input_template is not None:
            (r["run_dir"] / self.input_file).write_text(
                self.input_template.format(**r["fields"]), encoding="utf-8")


        env = {**os.environ, **self.env} if self.env else None
        proc = subprocess.run(r["argv"], capture_output=True, text=True, timeout=self.timeout,
                              cwd=str(r["run_dir"]) if self.cwd_is_workdir else None, env=env)
        (r["run_dir"] / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (r["run_dir"] / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-1:] or [""]
            raise RuntimeError(f"command exited with {proc.returncode}: {tail[0]}")
        if r["output"] == "-":
            text = proc.stdout
        else:
            path = Path(r["output"])
            if not path.is_absolute():
                path = r["run_dir"] / path
            if not path.exists():
                raise FileNotFoundError(f"result file not found: {path}")
            text = path.read_text(encoding="utf-8")
        return self.parse(text)

    def parse(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.regex:
            for name, pat in self.regex.items():
                matches = pat.findall(text)
                if not matches:
                    raise ValueError(f"pattern for {name!r} not found in result")
                m = matches[-1]
                out[name] = _num(m[0] if isinstance(m, tuple) else m)
        else:
            rows = list(csv.DictReader(text.splitlines()))
            if not rows:
                raise ValueError("result CSV has no data rows")
            row = rows[self.csv_row]
            for name, col in self.csv_columns.items():
                if col not in row:
                    raise KeyError(f"column {col!r} not in result CSV (have {list(row)})")
                out[name] = _num(row[col])
        return out


def _num(s: str) -> Any:
    try:
        return float(s)
    except (TypeError, ValueError):
        return s.strip() if isinstance(s, str) else s


# --------------------------------------------------------------------------- registry

BACKENDS: Dict[str, Callable[..., Any]] = {
    "single_diode": SingleDiodeBackend,
    "command": CommandBackend,
}


def get_backend(config: Union[str, Mapping[str, Any], None]) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    """Build a backend from ``{"type": name, **options}`` (or just a name)."""
    if config is None:
        config = {"type": "single_diode"}
    if isinstance(config, str):
        config = {"type": config}
    cfg = dict(config)
    kind = cfg.pop("type", "single_diode")
    opts = {**cfg.pop("options", {}), **cfg}
    try:
        cls = BACKENDS[kind]
    except KeyError:
        raise ValueError(f"unknown backend {kind!r}; available: {sorted(BACKENDS)}") from None
    return cls(**opts)
