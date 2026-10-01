"""Config loading, defaults, dotted overrides and validation (PLAN.md §2)."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads YAML 1.2 floats like ``40.0e9`` (YAML 1.1 needs ``40.0e+9``)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$|^[-+]?\.(?:inf|Inf|INF)$"
               r"|^\.(?:nan|NaN|NAN)$"),
    list("-+0123456789."))


def _yaml(text: str):
    return yaml.load(text, Loader=_Loader)


DEFAULTS: dict[str, Any] = {
    "airfoil": {"naca": "2412", "polar_file": None, "alpha_stall_deg": 12.0, "cd0": None,
                "n_points_per_surface": 150},
    "rotor": {"B": 3, "R": None, "r_hub": 0.0, "pitch_deg": 0.0},
    "stations": {"file": None, "r": None, "chord": None, "theta_deg": None},
    "operating": {"U_inf": None, "Omega": None, "rpm": None, "tsr": None},
    "air": {"rho": 1.225, "mu": 1.81e-5},
    "structure": {
        "model": "shell",
        "skin_thickness": None,
        "skin_thickness_frac": None,
        "user_section_props": None,
        "material": {"E": None, "sigma_allow_tension": None, "sigma_allow_compression": None,
                     "density": None},
        "include_gravity": True,
        "include_centrifugal": True,
    },
    "solver": {"re_tol": 0.01, "re_max_passes": 10},
    "output": {"dir": "results", "plots": True},
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_override(cfg: dict, assignment: str) -> None:
    """Apply ``a.b.c=value`` in place; ``value`` is parsed as YAML (numbers, lists, null...)."""
    if "=" not in assignment:
        raise ValueError(f"override must look like key.path=value, got {assignment!r}")
    path, raw = assignment.split("=", 1)
    keys = path.strip().split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _yaml(raw)


def load_config(source: str | Path | dict, overrides: list[str] | None = None) -> dict:
    """Load a YAML/JSON file (or dict), fill defaults, apply overrides, validate."""
    if isinstance(source, dict):
        user = source
        base_dir = Path.cwd()
    else:
        path = Path(source)
        user = _yaml(path.read_text())  # YAML is a superset of JSON
        base_dir = path.parent
    cfg = _merge(DEFAULTS, user)
    for o in overrides or []:
        apply_override(cfg, o)
    for sec, key in (("airfoil", "polar_file"), ("stations", "file")):
        f = cfg[sec][key]
        if f is not None and not Path(f).is_absolute():
            cfg[sec][key] = str((base_dir / f).resolve())
    if cfg["stations"]["file"] is not None:
        cfg["stations"].update(read_stations_file(cfg["stations"]["file"]))
    validate(cfg)
    return cfg


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"config: {msg}")


_COLUMN_ALIASES = {"r": "r", "chord": "chord", "c": "chord", "theta_deg": "theta_deg",
                   "pitch": "theta_deg", "twist": "theta_deg", "theta": "theta_deg"}


def read_stations_file(path: str | Path) -> dict[str, list[float]]:
    """Read r / chord / theta_deg (alias: pitch) columns from a .csv or .xlsx (first sheet).

    The first row is the header; reading stops at the first row whose first cell is not a
    number, so trailing notes in a spreadsheet are ignored. Units: m, m, deg (PLAN.md §1 sign).
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        ws = openpyxl.load_workbook(path, data_only=True, read_only=True).worksheets[0]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    else:
        import csv
        with path.open() as f:
            rows = [r for r in csv.reader(f) if r]
    header = [str(h).strip().lower() if h is not None else "" for h in rows[0]]
    cols = {_COLUMN_ALIASES[h]: i for i, h in enumerate(header) if h in _COLUMN_ALIASES}
    missing = {"r", "chord", "theta_deg"} - set(cols)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}; header was {rows[0]}")
    out: dict[str, list[float]] = {k: [] for k in cols}
    for row in rows[1:]:
        try:
            vals = {k: float(row[i]) for k, i in cols.items()}
        except (TypeError, ValueError, IndexError):
            break
        for k, v in vals.items():
            out[k].append(v)
    return out


def operating_points(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(U, Omega) arrays of equal length. Omega from Omega [rad/s], rpm, or tsr * U / R.

    Scalars broadcast against lists, so ``U_inf: 15, rpm: [1000, 2000]`` sweeps rotor speed.
    """
    op, R = cfg["operating"], cfg["rotor"]["R"]
    U = np.atleast_1d(np.asarray(op["U_inf"], float))
    if op["tsr"] is not None:
        tsr = np.atleast_1d(np.asarray(op["tsr"], float))
        U, tsr = np.broadcast_arrays(U, tsr)
        return np.array(U), tsr * U / R
    Om = (np.asarray(op["Omega"], float) if op["Omega"] is not None
          else np.asarray(op["rpm"], float) * 2 * np.pi / 60)
    U, Om = np.broadcast_arrays(U, np.atleast_1d(Om))
    return np.array(U, float), np.array(Om, float)


def validate(cfg: dict) -> None:
    _need(isinstance(cfg["airfoil"]["naca"], str),
          'airfoil.naca must be a quoted string, e.g. "0012" (unquoted 0012 parses as a number)')
    rot, st, op, s = cfg["rotor"], cfg["stations"], cfg["operating"], cfg["structure"]
    _need(rot["R"] is not None and rot["R"] > 0, "rotor.R must be > 0")
    _need(0 <= rot["r_hub"] < rot["R"], "need 0 <= rotor.r_hub < rotor.R")
    _need(int(rot["B"]) >= 1, "rotor.B must be >= 1")
    for k in ("r", "chord", "theta_deg"):
        _need(st[k] is not None, f"stations.{k} is required")
    r, c, th = (np.asarray(st[k], float) for k in ("r", "chord", "theta_deg"))
    _need(r.ndim == 1 and r.size >= 2, "stations.r needs >= 2 entries")
    _need(r.shape == c.shape == th.shape, "stations.r, chord, theta_deg must have equal length")
    _need(bool(np.all(np.diff(r) > 0)), "stations.r must be strictly increasing")
    _need(rot["r_hub"] <= r[0] and r[-1] <= rot["R"], "need r_hub <= r[0] and r[-1] <= R")
    _need(bool(np.all(c > 0)), "stations.chord must be > 0")

    _need(op["U_inf"] is not None, "operating.U_inf is required")
    _need(sum(op[k] is not None for k in ("Omega", "rpm", "tsr")) == 1,
          "give exactly one of operating.Omega / rpm / tsr")
    try:
        U, Om = operating_points(cfg)
    except ValueError as e:
        raise ValueError("config: operating lists must have equal length (or be scalars)") from e
    _need(bool(np.all(U > 0) and np.all(Om > 0)), "U_inf and Omega must be > 0")

    _need(s["model"] in ("solid", "shell", "user"), "structure.model must be solid|shell|user")
    if s["model"] == "shell":
        _need((s["skin_thickness"] is None) != (s["skin_thickness_frac"] is None),
              "shell model: give exactly one of skin_thickness / skin_thickness_frac")
    if s["model"] == "user":
        u = s["user_section_props"]
        _need(isinstance(u, dict), "user model: structure.user_section_props is required")
        for k in ("A", "S_11", "S_22", "S_12", "points_1", "points_2"):
            _need(k in u, f"user_section_props.{k} is required")
            _need(len(u[k]) == r.size, f"user_section_props.{k} must have one entry per station")
    m = s["material"]
    _need(m["E"] is not None and m["E"] > 0, "structure.material.E is required")
    for k in ("sigma_allow_tension", "sigma_allow_compression"):
        _need(m[k] is not None and m[k] > 0, f"structure.material.{k} is required")
