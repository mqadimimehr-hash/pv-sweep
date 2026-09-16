# pv-sweep

Automate parameter sweeps for solar cell device simulations.

Drift-diffusion studies usually involve a lot of repetitive runs: sweep the
absorber thickness, then the doping, then the defect density, then collect
Voc / Jsc / FF / PCE from each output file into one table. `pv-sweep` handles
that loop. You describe the sweep in YAML (or a Python dict), choose a backend,
and get a CSV of results plus quick plots.

- **Sweep definitions**: explicit lists, `linspace`, `logspace`, `geomspace` or `range`.
- **Two modes**: full grid (Cartesian product) or one-at-a-time (OAT) around a reference point.
- **Pluggable backends**:
  - `single_diode`: a built-in analytic single-diode model, so the package works with no simulator installed.
  - `command`: wraps any command-line simulator. It fills parameter values into a command template, runs the command, and reads results from a file or stdout using a regex or a CSV column mapping.
- **Results**: one CSV row per point, with a `status` column. A failed run is recorded and the sweep keeps going.
- **Plots**: 1D line plots (optionally one line per group) and 2D heatmaps. Plotting needs matplotlib, which is optional.

## Install

```bash
pip install -e .            # core: numpy, pyyaml
pip install -e ".[plot]"    # add matplotlib for plotting
pip install -e ".[dev]"     # add pytest
```

## Quick start

```bash
pvsweep run examples/sweep.yaml -o results.csv
pvsweep plot results.csv --x Rs --y PCE --group Rsh -s pce_vs_rs.png
pvsweep plot results.csv --x Rs --y Rsh --z PCE -s pce_map.png
pvsweep list examples/sweep.yaml     # show the expanded points without running them
```

`pvsweep run` options: `--quiet`, `--fail-fast` (stop at the first error),
`--strict` (exit with code 1 if any point failed).

## Sweep file

```yaml
mode: grid                 # or: oat
base:                      # fixed inputs (also the OAT reference point)
  Jph: 38.0
  J0: 1.0e-8
  n: 1.5
  temperature: 300
parameters:
  Rs:  {linspace: [0.0, 8.0, 5]}
  Rsh: {logspace: [2, 4, 5]}     # 1e2 ... 1e4
  # thickness: [100, 200, 400]   # explicit list
backend:
  type: single_diode
```

In `oat` mode each parameter is varied on its own while the others stay at
their `base` value. If a parameter has no `base` value, its first listed value
is used. OAT rows include a `varied` column naming the parameter that changed.

## Python API

```python
from pvsweep import Sweep, run_sweep, write_csv
from pvsweep.sweep import plot_line

sweep = Sweep({"Jph": {"linspace": [20, 40, 5]}, "Rs": [0.5, 2, 5]},
              base={"J0": 1e-9, "n": 1.3})
rows = run_sweep(sweep)            # list of dicts
write_csv(rows, "results.csv")
plot_line(rows, x="Jph", y="Voc", group="Rs")
```

Any callable `backend(params: dict) -> dict` can be passed as a backend:
`run_sweep(sweep, backend=my_function)`.

## Backends

### `single_diode`

This backend solves the implicit single-diode equation exactly:

```
J = Jph - J0 [exp((V + J Rs) / (n kT/q)) - 1] - (V + J Rs) / Rsh
```

It reports `Voc` [V], `Jsc` [mA/cm²], `FF` [%], `PCE` [%], `Vmpp`, `Jmpp` and `Pmax`.

| parameter | meaning | unit | default |
|---|---|---|---|
| `Jph` | photogenerated current density | mA/cm² | 35 |
| `J0` | diode saturation current density | mA/cm² | 1e-9 |
| `n` (`ideality`) | ideality factor | – | 1.3 |
| `Rs` | series resistance | Ω·cm² | 1 |
| `Rsh` | shunt resistance | Ω·cm² | 1000 |
| `T` (`temperature`) | temperature | K | 300 |
| `Pin` | incident power | mW/cm² | 100 |

`J0` is used exactly as given, with no built-in temperature scaling. This model
is a lumped-circuit approximation for quick studies and teaching. It is not a
drift-diffusion solver. Sweep parameters it does not recognize cause an error,
unless you set `ignore_unknown: true` in the backend options.

### `command`

The `command` backend is meant to wrap any command-line simulator: your own
solver, a Python script, or a batch-capable device simulator.

```yaml
backend:
  type: command
  command: "mysim --thick {thickness} --dop {doping} --out {output}"
  result_file: "{workdir}/summary.csv"   # "-" parses stdout instead
  csv_columns: {Voc: voc, Jsc: jsc, FF: ff, PCE: eff}   # last row is used
  # regex: {PCE: 'Efficiency\s*=\s*([-+0-9.eE]+)'}      # alternative: one capture group
  timeout: 600
  workdir: runs/                          # one run_00000/, run_00001/ ... per point
  # input_template: "thickness = {thickness}\n"  # optional input file written per run
  # input_file: input.txt
```

The following placeholders are available in the command template:

- every sweep and `base` parameter
- `{output}`: the path of the result file
- `{workdir}`: the folder for this run
- `{index}`: the run number
- `{python}`: the current Python interpreter

The command string is split into arguments before the values are filled in, and
it runs without a shell, so parameter values are never interpreted by a shell.
Each run's stdout and stderr are saved in its run folder.

> Adapters for specific simulators (for example SCAPS or other drift-diffusion
> codes) are not included yet. They are planned, and contributions are welcome.

## Development

```bash
pip install -e ".[dev,plot]"
pytest
```

## License

MIT © 2026 Mohammad Ghadimimehr. See [LICENSE](LICENSE).
Repository: <https://github.com/mqadimimehr-hash/pv-sweep>
