"""Command-line interface: ``pvsweep run`` / ``pvsweep plot`` / ``pvsweep list``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .sweep import load_sweep, plot_heatmap, plot_line, read_csv, run_sweep, write_csv


def _cmd_run(args: argparse.Namespace) -> int:
    sweep = load_sweep(args.sweep)
    total = len(sweep)
    print(f"pvsweep: {total} point(s), mode={sweep.mode}, backend={sweep.backend.get('type')}",
          file=sys.stderr)

    def progress(i: int, n: int, row: dict) -> None:
        if not args.quiet:
            tag = "" if row["status"] == "ok" else f"  [error] {row.get('error')}"
            print(f"  [{i}/{n}]{tag}", file=sys.stderr)

    rows = run_sweep(sweep, progress=progress, stop_on_error=args.fail_fast)
    path = write_csv(rows, args.output)
    n_err = sum(r["status"] != "ok" for r in rows)
    print(f"wrote {len(rows)} row(s) to {path} ({n_err} error(s))", file=sys.stderr)
    return 1 if n_err and args.strict else 0


def _cmd_list(args: argparse.Namespace) -> int:
    for p in load_sweep(args.sweep).points():
        print(p)
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    rows = read_csv(args.csv)
    if args.z:
        ax = plot_heatmap(rows, args.x, args.y, args.z)
    else:
        ax = plot_line(rows, args.x, args.y, group=args.group, logx=args.logx, logy=args.logy)
    fig = ax.figure
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=args.dpi)
        print(f"saved {args.save}", file=sys.stderr)
    else:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pvsweep", description="Parameter sweeps for solar cell simulations")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a sweep defined in YAML")
    r.add_argument("sweep", help="sweep YAML file")
    r.add_argument("-o", "--output", default="results.csv", help="output CSV (default: results.csv)")
    r.add_argument("-q", "--quiet", action="store_true", help="no per-point progress")
    r.add_argument("--fail-fast", action="store_true", help="stop at the first failing point")
    r.add_argument("--strict", action="store_true", help="exit 1 if any point failed")
    r.set_defaults(func=_cmd_run)

    ls = sub.add_parser("list", help="print the expanded sweep points without running")
    ls.add_argument("sweep")
    ls.set_defaults(func=_cmd_list)

    pl = sub.add_parser("plot", help="plot a results CSV (needs matplotlib)")
    pl.add_argument("csv")
    pl.add_argument("--x", required=True)
    pl.add_argument("--y", required=True)
    pl.add_argument("--z", help="colour column: draws a 2D heatmap of z over (x, y)")
    pl.add_argument("--group", help="draw one line per value of this column")
    pl.add_argument("--logx", action="store_true")
    pl.add_argument("--logy", action="store_true")
    pl.add_argument("-s", "--save", help="save figure to file instead of showing it")
    pl.add_argument("--dpi", type=int, default=150)
    pl.set_defaults(func=_cmd_plot)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"pvsweep: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
