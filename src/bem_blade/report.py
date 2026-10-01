"""Orchestration and outputs (PLAN.md §8-9): stations CSV, summary JSON, plots, console summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .model import LIMITATIONS, Result, op_label, rpm, run

STATION_COLUMNS = [
    "r", "c", "theta_deg", "phi_deg", "alpha_deg", "a", "ap", "F", "W", "Re", "Cl", "Cd", "cn",
    "ct", "pN", "pT", "V_N", "V_T", "M_flap", "M_edge", "A", "S_tt", "S_nn", "S_tn",
    "sigma_max_t", "sigma_max_t_loc", "sigma_min_c", "sigma_min_c_loc", "stall", "aero_model",
    "bem_region",
]


def eng(x: float, unit: str) -> str:
    """Format with an SI prefix: eng(1994e3, 'N m') -> '1.99 MN m'."""
    for f, pre in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m")):
        if abs(x) >= f:
            return f"{x / f:.3g} {pre}{unit}"
    return f"{x:.3g} {unit}"


def _loc(res: Result, pt: int, case: int) -> str:
    b = res.blade
    grav = "" if not b.include_gravity else (" g+t" if case == 0 else " g-t")
    if b.x_over_c is None:
        return f"pt{pt}{grav}"
    surf = "upper" if b.is_upper[pt] else "lower"
    return f"{surf} x/c={b.x_over_c[pt]:.3f}{grav}"


def station_rows(res: Result, i: int) -> list[dict]:
    o, g = res.out, res.blade.geom
    n = len(g.r)
    rows = []
    for j in range(n):
        rows.append({
            "r": float(g.r[j]), "c": float(g.c[j]), "theta_deg": float(np.rad2deg(g.theta[j])),
            "phi_deg": float(np.rad2deg(o["phi"][i, j])),
            "alpha_deg": float(np.rad2deg(o["alpha"][i, j])),
            "a": o["a"][i, j], "ap": o["ap"][i, j], "F": o["F"][i, j], "W": o["W"][i, j],
            "Re": o["Re"][i, j], "Cl": o["cl"][i, j], "Cd": o["cd"][i, j],
            "cn": o["cn"][i, j], "ct": o["ct"][i, j], "pN": o["pN"][i, j], "pT": o["pT"][i, j],
            "V_N": o["V_N"][i, j], "V_T": o["V_T"][i, j],
            "M_flap": o["M_flap"][i, j], "M_edge": o["M_edge"][i, j],
            "A": float(g.sec.A[j]), "S_tt": float(g.sec.S_tt[j]), "S_nn": float(g.sec.S_nn[j]),
            "S_tn": float(g.sec.S_tn[j]),
            "sigma_max_t": o["sigma_max"][i, j],
            "sigma_max_t_loc": _loc(res, o["sigma_max_pt"][i, j], o["sigma_max_case"][i, j]),
            "sigma_min_c": o["sigma_min"][i, j],
            "sigma_min_c_loc": _loc(res, o["sigma_min_pt"][i, j], o["sigma_min_case"][i, j]),
            "stall": bool(o["stall"][i, j]), "aero_model": str(o["aero_model"][i, j]),
            "bem_region": int(o["region"][i, j]),
        })
    return rows


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                        for k, v in row.items()})


def point_summary(res: Result, i: int) -> dict:
    o, g = res.out, res.blade.geom
    r = np.asarray(g.r)
    mat = res.cfg["structure"]["material"]
    Mf, Me = o["M_flap"][i, :len(r)], o["M_edge"][i, :len(r)]
    Mres = np.hypot(Mf, Me)
    jt, jc = int(np.argmax(o["sigma_max"][i])), int(np.argmin(o["sigma_min"][i]))
    s_t, s_c = float(o["sigma_max"][i, jt]), float(o["sigma_min"][i, jc])
    jf, je, jr = (int(np.argmax(np.abs(x))) for x in (Mf, Me, Mres))
    tip = o["delta_n"][i][:, -1]
    return {
        "U_inf": float(res.U[i]), "Omega": float(res.Omega[i]), "rpm": float(rpm(res)[i]),
        "tsr": float(res.Omega[i] * res.blade.rotor.R / res.U[i]),
        "T": float(o["T"][i]), "Q": float(o["Q"][i]), "P": float(o["P"][i]),
        "CT": float(o["CT"][i]), "CP": float(o["CP"][i]),
        "max_M_flap": {"value": float(Mf[jf]), "r": float(r[jf])},
        "max_M_edge": {"value": float(Me[je]), "r": float(r[je]),
                       "note": "aerodynamic only; gravity enters the stress load cases"},
        "max_M_resultant": {"value": float(Mres[jr]), "r": float(r[jr])},
        "root_M_grav": float(o["M_grav"][i, 0]),
        "root_N_centrifugal": float(o["N"][i, 0]),
        "sigma_max_tension": {"value": s_t, "r": float(r[jt]),
                              "location": _loc(res, o["sigma_max_pt"][i, jt],
                                               o["sigma_max_case"][i, jt])},
        "sigma_max_compression": {"value": s_c, "r": float(r[jc]),
                                  "location": _loc(res, o["sigma_min_pt"][i, jc],
                                                   o["sigma_min_case"][i, jc])},
        "SF_tension": mat["sigma_allow_tension"] / s_t if s_t > 0 else float("inf"),
        "SF_compression": mat["sigma_allow_compression"] / -s_c if s_c < 0 else float("inf"),
        "tip_deflection_n": float(tip[np.argmax(np.abs(tip))]),
    }


def envelope(res: Result) -> list[dict]:
    """Per-station extremes across operating points and which U_inf governs (§8)."""
    o, r = res.out, np.asarray(res.blade.geom.r)
    n = len(r)
    Mf, Me = np.abs(o["M_flap"][:, :n]), np.abs(o["M_edge"][:, :n])
    W = rpm(res)
    rows = []
    for j in range(n):
        kf, ke = np.argmax(Mf[:, j]), np.argmax(Me[:, j])
        kt, kc = np.argmax(o["sigma_max"][:, j]), np.argmin(o["sigma_min"][:, j])
        rows.append({"r": r[j],
                     "max_abs_M_flap": Mf[kf, j], "U_M_flap": res.U[kf], "rpm_M_flap": W[kf],
                     "max_abs_M_edge": Me[ke, j], "U_M_edge": res.U[ke], "rpm_M_edge": W[ke],
                     "sigma_max_t": o["sigma_max"][kt, j], "U_sigma_t": res.U[kt],
                     "rpm_sigma_t": W[kt],
                     "sigma_min_c": o["sigma_min"][kc, j], "U_sigma_c": res.U[kc],
                     "rpm_sigma_c": W[kc]})
    return rows


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    return x


def write_outputs(res: Result, out_dir: str | Path, plots: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = len(res.U) > 1
    points = [point_summary(res, i) for i in range(len(res.U))]

    for i, u in enumerate(res.U):
        name = f"stations_{op_label(res, i, file=True)}.csv" if sweep else "stations.csv"
        _write_csv(out_dir / name, station_rows(res, i), STATION_COLUMNS)

    summary = {"operating_points": points, "re_passes": res.re_passes,
               "warnings": res.warnings, "limitations": list(LIMITATIONS),
               "config": res.cfg}
    if sweep:
        env = envelope(res)
        _write_csv(out_dir / "envelope.csv", env, list(env[0]))
        g = max(range(len(points)), key=lambda k: abs(points[k]["max_M_flap"]["value"]))
        summary["envelope"] = {
            "governing_root_flap": {"U_inf": points[g]["U_inf"], "rpm": points[g]["rpm"]},
            "min_SF_tension": min(p["SF_tension"] for p in points),
            "min_SF_compression": min(p["SF_compression"] for p in points),
        }
        summary["limitations"].append(
            "Envelope uses fixed theta (no pitch control model); operating points are exactly "
            "those listed, with no controller deciding which are reachable.")
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))

    if plots:
        from .plots import make_plots
        make_plots(res, out_dir)
    return out_dir


def run_from_config(source, overrides=None, out_dir=None, plots=None, quiet=False) -> Result:
    """Convenience entry point: load config, run, write outputs, print the summary."""
    cfg = load_config(source, overrides)
    res = run(cfg)
    out = write_outputs(res, out_dir or cfg["output"]["dir"],
                        cfg["output"]["plots"] if plots is None else plots)
    if not quiet:
        from .console import print_summary
        title = Path(source).stem if not isinstance(source, dict) else None
        print_summary(res, title=title, out_dir=out)
    return res
