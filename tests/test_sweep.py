import sys
from pathlib import Path

import numpy as np
import pytest

from pvsweep import (CommandBackend, SingleDiodeBackend, Sweep, expand_values, get_backend,
                     load_sweep, read_csv, run_sweep, single_diode, write_csv)
from pvsweep.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sweep.yaml"


# ---------------------------------------------------------------- expansion / sweeps

def test_expand_values_forms():
    assert expand_values([1, 2, 3]) == [1, 2, 3]
    assert expand_values(5) == [5]
    assert expand_values({"linspace": [0, 1, 5]}) == pytest.approx([0, 0.25, 0.5, 0.75, 1])
    assert expand_values({"logspace": [1, 3, 3]}) == pytest.approx([10, 100, 1000])
    assert expand_values({"range": [0, 1, 0.25]}) == pytest.approx([0, 0.25, 0.5, 0.75, 1])


@pytest.mark.parametrize("bad", [{"linspace": [0, 1]}, {"linspace": [0, 1, 2.5]},
                                 {"foo": [0, 1, 2]}, [], {"range": [0, 1, -1]}])
def test_expand_values_rejects_bad(bad):
    with pytest.raises(ValueError):
        expand_values(bad)


def test_grid_size_and_base_merge():
    s = Sweep({"a": [1, 2, 3], "b": {"linspace": [0, 1, 4]}}, base={"c": 9})
    pts = s.points()
    assert len(s) == len(pts) == 12
    assert all(p["c"] == 9 for p in pts)
    assert len({(p["a"], p["b"]) for p in pts}) == 12


def test_oat_size_and_reference():
    s = Sweep({"a": [1, 2, 3], "b": [10, 20]}, mode="oat", base={"b": 20})
    pts = s.points()
    assert len(s) == len(pts) == 5
    a_pts = [p for p in pts if p["varied"] == "a"]
    assert [p["a"] for p in a_pts] == [1, 2, 3] and all(p["b"] == 20 for p in a_pts)
    b_pts = [p for p in pts if p["varied"] == "b"]
    assert all(p["a"] == 1 for p in b_pts)  # no base value -> first listed value


def test_invalid_sweep():
    with pytest.raises(ValueError):
        Sweep({"a": [1]}, mode="random")
    with pytest.raises(ValueError):
        Sweep({})
    with pytest.raises(ValueError):
        Sweep.from_dict({"parameters": {"a": [1]}, "typo": 1})


def test_load_example_yaml():
    s = load_sweep(EXAMPLE)
    assert s.mode == "grid" and len(s) == 25


# ---------------------------------------------------------------- single-diode physics

def test_ideal_diode_matches_analytic_voc():
    r = single_diode(Jph=30, J0=1e-9, n=1.0, Rs=0, Rsh=1e12, T=300)
    vt = 1.380649e-23 * 300 / 1.602176634e-19
    assert r["Voc"] == pytest.approx(vt * np.log(30 / 1e-9 + 1), rel=1e-3)
    assert r["Jsc"] == pytest.approx(30, rel=1e-6)
    assert 80 < r["FF"] < 90
    assert r["PCE"] == pytest.approx(r["Voc"] * r["Jsc"] * r["FF"] / 100, rel=1e-6)


def test_voc_and_jsc_rise_with_jph():
    res = [single_diode(Jph=j) for j in (10, 20, 30, 40)]
    assert all(np.diff([r["Voc"] for r in res]) > 0)
    assert all(np.diff([r["Jsc"] for r in res]) > 0)


def test_losses_reduce_ff():
    ffs_rs = [single_diode(Rs=rs)["FF"] for rs in (0, 2, 5, 10)]
    assert all(np.diff(ffs_rs) < 0)
    ffs_rsh = [single_diode(Rsh=rsh)["FF"] for rsh in (50, 200, 1e3, 1e5)]
    assert all(np.diff(ffs_rsh) > 0)


def test_voc_falls_with_j0_and_temperature():
    assert single_diode(J0=1e-8)["Voc"] < single_diode(J0=1e-10)["Voc"]
    # fixed J0: Voc = n kT/q ln(Jph/J0) grows with T; check it responds monotonically
    assert single_diode(T=350)["Voc"] > single_diode(T=300)["Voc"]


def test_dark_cell_gives_zero_output():
    r = single_diode(Jph=0)
    assert r["PCE"] == 0 and r["FF"] == 0


def test_backend_aliases_and_unknown():
    be = SingleDiodeBackend(Jph=30)
    assert be({"temperature": 300, "rs": 1})["Jsc"] == pytest.approx(30, rel=0.05)
    with pytest.raises(KeyError):
        be({"thickness": 100})
    assert "PCE" in SingleDiodeBackend(ignore_unknown=True)({"thickness": 100})


def test_run_sweep_and_csv_roundtrip(tmp_path):
    rows = run_sweep({"parameters": {"Rs": [0, 5]}, "base": {"Jph": 30}})
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert rows[0]["PCE"] > rows[1]["PCE"]
    back = read_csv(write_csv(rows, tmp_path / "r.csv"))
    assert back[1]["PCE"] == pytest.approx(rows[1]["PCE"])


def test_errors_recorded_per_row():
    rows = run_sweep({"parameters": {"n": [1.2, -1]}})
    assert [r["status"] for r in rows] == ["ok", "error"]
    assert "n must be > 0" in rows[1]["error"]
    with pytest.raises(ValueError):
        run_sweep({"parameters": {"n": [-1]}}, stop_on_error=True)


# ---------------------------------------------------------------- command backend

WRITER = ("import sys; t = float(sys.argv[2]); "
          "open(sys.argv[1], 'w').write('Voc = %.4f V\\nPCE = %.3f %%\\n' % (0.5 + t / 1000, t / 20))")


def test_command_backend_regex(tmp_path):
    be = CommandBackend(f'{{python}} -c "{WRITER}" {{output}} {{thickness}}',
                        regex={"Voc": r"Voc = ([\d.]+)", "PCE": r"PCE = ([\d.]+)"},
                        workdir=str(tmp_path))
    rows = run_sweep(Sweep({"thickness": [100, 300]}), backend=be)
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert rows[0]["Voc"] == pytest.approx(0.6) and rows[1]["PCE"] == pytest.approx(15)
    assert (tmp_path / "run_00001" / "result.txt").exists()


def test_command_backend_csv_stdout_via_config(tmp_path):
    code = "import sys; print('x,eff'); print('1,%s' % (2 * float(sys.argv[1])))"
    rows = run_sweep({
        "parameters": {"d": [1.5, 4]},
        "backend": {"type": "command", "command": [sys.executable, "-c", code, "{d}"],
                    "result_file": "-", "csv_columns": {"PCE": "eff"}, "workdir": str(tmp_path)},
    })
    assert [r["PCE"] for r in rows] == [3.0, 8.0]


def test_command_backend_failure_is_recorded(tmp_path):
    be = CommandBackend('{python} -c "import sys; sys.exit(3)"', regex={"a": "(x)"},
                        workdir=str(tmp_path))
    row = run_sweep(Sweep({"p": [1]}), backend=be)[0]
    assert row["status"] == "error" and "exited with 3" in row["error"]


def test_command_backend_requires_parser():
    with pytest.raises(ValueError):
        CommandBackend("echo hi")
    with pytest.raises(ValueError):
        get_backend({"type": "nope"})


# ---------------------------------------------------------------- CLI / plotting

def test_cli_run_and_plot(tmp_path):
    out = tmp_path / "res.csv"
    assert main(["run", str(EXAMPLE), "-o", str(out), "-q"]) == 0
    rows = read_csv(out)
    assert len(rows) == 25 and all(r["status"] == "ok" for r in rows)
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    assert main(["plot", str(out), "--x", "Rs", "--y", "PCE", "--group", "Rsh",
                 "-s", str(tmp_path / "line.png")]) == 0
    assert main(["plot", str(out), "--x", "Rs", "--y", "Rsh", "--z", "PCE",
                 "-s", str(tmp_path / "map.png")]) == 0
    assert (tmp_path / "line.png").stat().st_size > 0
    assert (tmp_path / "map.png").stat().st_size > 0


def test_cli_missing_file(tmp_path):
    assert main(["run", str(tmp_path / "missing.yaml")]) == 2
