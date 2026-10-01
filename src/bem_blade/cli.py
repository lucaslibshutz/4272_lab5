"""Command line: ``bem-blade run CONFIG [--set key=value ...]`` and ``bem-blade polar ...``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bem-blade", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the BEM + structural analysis for a config file")
    pr.add_argument("config", help="YAML or JSON config (see examples/)")
    pr.add_argument("-o", "--out", help="output directory (default: output.dir in config)")
    pr.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set operating.U_inf=[6,8,10,12]")
    pr.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    pr.add_argument("-q", "--quiet", action="store_true", help="no console summary")

    pp = sub.add_parser("polar", help="generate a polar table with NeuralFoil (optional extra)")
    pp.add_argument("naca", help="4-digit NACA designation, e.g. 2412")
    pp.add_argument("--re", type=float, nargs="+", required=True, help="Reynolds numbers")
    pp.add_argument("-o", "--out", required=True, help="output CSV path")
    pp.add_argument("--alpha", type=float, nargs=3, default=(-20.0, 30.0, 0.5),
                    metavar=("MIN", "MAX", "STEP"), help="alpha range in degrees")

    args = p.parse_args(argv)
    if args.cmd == "run":
        from .report import run_from_config
        run_from_config(args.config, args.set, args.out, plots=not args.no_plots,
                        quiet=args.quiet)
        return 0
    if args.cmd == "polar":
        import numpy as np
        from .polargen import generate_polar_csv
        lo, hi, step = args.alpha
        path = generate_polar_csv(args.naca, args.re, args.out, np.arange(lo, hi + 1e-9, step))
        print(f"wrote {path}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
