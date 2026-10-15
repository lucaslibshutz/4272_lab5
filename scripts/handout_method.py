"""Lab handout method for Blade 7, written independently of bem.py as a cross-check.

Blade-element theory only (write-up §2-§5): U_eff = (Omega r, U_inf), phi = atan2(U_inf, Omega r),
alpha = phi - theta, F_Q' = 1/2 rho U_eff^2 c (C_L sin phi - C_D cos phi). Torque
Q = B * sum over sections of F_Q' r dr for each discretization rule of write-up §4 (left, trapezoid,
midpoint geometry, right), and the freewheeling speed Omega_ss from Q(Omega_ss) = 0 (§5).

Only ``bem_blade.aero.load_polar_table`` is shared with the package.

    uv run python scripts/handout_method.py [--polar CSV] [--blend DEG] [--U 15] [--rpm 3600]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import openpyxl
from scipy.optimize import brentq

import bem_blade  # noqa: F401  (float64)
from bem_blade.aero import load_polar_table

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "data" / "Blade7_geometry.xlsx"
POLAR = ROOT / "data" / "naca4412_airfoiltools.csv"
RHO, MU, B = 1.225, 1.81e-5, 3
RULES = ("left", "trap", "mid", "right")
RPM_GRID = np.arange(1000.0, 60001.0, 250.0)


def blade7_geometry(path: Path = GEOMETRY):
    """r [m], chord [m], theta [rad] at the 20 stations of the Blade 7 sheet."""
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(min_row=2, max_row=21, values_only=True))
    r, theta_deg, c = (np.array(x, float) for x in zip(*rows))
    return r, c, np.deg2rad(theta_deg)


def stations(polar, rpm, r, c, theta, U):
    """Per station: f = dQ/dr per blade [N], alpha [rad], Re, C_L, C_D."""
    Om = rpm * 2 * np.pi / 60
    U_eff = np.hypot(U, Om * r)
    phi = np.arctan2(U, Om * r)
    alpha = phi - theta
    Re = RHO * U_eff * c / MU
    cl, cd = (np.asarray(v) for v in polar.cl_cd(jnp.asarray(alpha), jnp.asarray(Re)))
    f = 0.5 * RHO * U_eff**2 * c * (cl * np.sin(phi) - cd * np.cos(phi)) * r
    return f, alpha, Re, cl, cd


def torque(polar, rpm, rule, geom, U):
    """Rotor torque Q [N m] with discretization ``rule``."""
    r, c, theta = geom
    dr = np.diff(r)
    if rule == "mid":  # interpolate the geometry to the section centres (write-up eq. midpoint)
        mid = lambda x: 0.5 * (x[1:] + x[:-1])
        f = stations(polar, rpm, mid(r), mid(c), mid(theta), U)[0]
        return B * np.sum(f * dr)
    f = stations(polar, rpm, r, c, theta, U)[0]
    seg = {"left": f[:-1], "right": f[1:], "trap": 0.5 * (f[:-1] + f[1:])}[rule]
    return B * np.sum(seg * dr)


def omega_ss_rpm(polar, rule, geom, U):
    """Freewheeling speed [RPM]: last + to - sign change of Q on RPM_GRID, refined by brentq."""
    q = np.array([torque(polar, x, rule, geom, U) for x in RPM_GRID])
    down = np.where((q[:-1] > 0) & (q[1:] <= 0))[0]
    if down.size == 0:
        raise ValueError(f"no + to - sign change of Q in {RPM_GRID[0]:g}-{RPM_GRID[-1]:g} RPM")
    i = down[-1]
    return brentq(lambda x: torque(polar, x, rule, geom, U), RPM_GRID[i], RPM_GRID[i + 1],
                  xtol=0.1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--polar", default=str(POLAR), help="polar table CSV (Re, alpha_deg, Cl, Cd)")
    p.add_argument("--blend", type=float, default=10.0,
                   help="width [deg] of the blend from the table edge to the flat plate")
    p.add_argument("--U", type=float, default=15.0, help="wind speed [m/s]")
    p.add_argument("--rpm", type=float, default=3600.0, help="RPM at which to report Q")
    a = p.parse_args(argv)

    geom = blade7_geometry()
    polar, _ = load_polar_table(a.polar, blend_deg=a.blend)
    print(f"polar: {a.polar} (blend {a.blend:g} deg), U_inf = {a.U:g} m/s")
    print(f"{'rule':6s} {'Q at ' + format(a.rpm, '.0f') + ' RPM [mN m]':>22s} {'Omega_ss [RPM]':>15s}")
    oss = {}
    for rule in RULES:
        oss[rule] = omega_ss_rpm(polar, rule, geom, a.U)
        print(f"{rule:6s} {torque(polar, a.rpm, rule, geom, a.U) * 1e3:22.1f} {oss[rule]:15.0f}")

    r, c, theta = geom
    _, alpha, Re, cl, cd = stations(polar, oss["trap"], r, c, theta, a.U)
    inside = np.asarray(polar.in_table(jnp.asarray(alpha), jnp.asarray(Re)))
    print(f"\nat the trapezoid Omega_ss = {oss['trap']:.0f} RPM:")
    for name, j in (("hub", 0), ("tip", -1)):
        print(f"  {name}: alpha = {np.rad2deg(alpha[j]):6.2f} deg, Re = {Re[j]:.3g}, "
              f"L/D = {cl[j] / cd[j]:6.1f}, in table: {bool(inside[j])}")
    print(f"  stations outside the table: {int((~inside).sum())} of {r.size}")

    print("\nblend sensitivity (trapezoid Omega_ss):")
    for blend in (2.0, 5.0, 10.0, 20.0):
        pol_b, _ = load_polar_table(a.polar, blend_deg=blend)
        print(f"  blend {blend:4.0f} deg: {omega_ss_rpm(pol_b, 'trap', geom, a.U):7.0f} RPM")


if __name__ == "__main__":
    main()
