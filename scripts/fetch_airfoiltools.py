"""Download NACA 4412 XFOIL polars from airfoiltools.com and build the bem-blade polar table.

Usage (from the repo root):  uv run python scripts/fetch_airfoiltools.py

Raw CSVs go to data/airfoiltools/ (kept as downloaded); the combined table with columns
``Re, alpha_deg, Cl, Cd`` goes to data/naca4412_airfoiltools.csv. Source polars are XFOIL,
Ncrit = 9, Mach 0, at every Reynolds number airfoiltools publishes for naca4412-il.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

AIRFOIL = "naca4412-il"
RES = (50_000, 100_000, 200_000, 500_000, 1_000_000)
URL = "http://airfoiltools.com/polar/csv?polar=xf-{airfoil}-{re}"   # Ncrit 9 (the -n5 keys are Ncrit 5)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "airfoiltools"
OUT = ROOT / "data" / "naca4412_airfoiltools.csv"


def fetch(re: int) -> str:
    raw = RAW_DIR / f"xf-{AIRFOIL}-{re}.csv"
    if not raw.exists():
        with urllib.request.urlopen(URL.format(airfoil=AIRFOIL, re=re), timeout=30) as r:
            raw.write_text(r.read().decode())
    return raw.read_text()


def parse(text: str) -> list[tuple[float, float, float]]:
    """Rows (alpha_deg, Cl, Cd) from the data block that follows the 'Alpha,Cl,Cd,...' header."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Alpha,"))
    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [(float(r["Alpha"]), float(r["Cl"]), float(r["Cd"])) for r in rows if r["Alpha"]]


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Re", "alpha_deg", "Cl", "Cd"])
        for re in RES:
            rows = parse(fetch(re))
            for a, cl, cd in rows:
                w.writerow([re, f"{a:.3f}", f"{cl:.4f}", f"{cd:.5f}"])
            print(f"Re {re:>9,}: {len(rows):3d} points, alpha [{rows[0][0]:.2f}, {rows[-1][0]:.2f}] deg")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
