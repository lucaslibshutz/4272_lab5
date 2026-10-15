"""Assemble a blade from a config and evaluate it at one or many operating points.

``evaluate`` is one jitted function (BEM -> loads -> moments -> stress -> deflection) that is
``vmap``-ed over operating points, so an envelope sweep is a single compiled call per Re pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import naca
from .aero import FallbackPolar, load_polar_table
from .bem import Rotor, blade_element_stations, rotor_totals, solve_stations
from .section import SectionProps, airfoil_sections, user_sections
from .structure import G, bending_stress, curvatures, deflection, integrate_from_tip, \
    shear_and_moment
from .config import operating_points

log = logging.getLogger(__name__)

LIMITATIONS = [
    "Steady, uniform, axial inflow; one rigid blade (no yaw/shear/tower shadow, no aeroelasticity).",
    "Not evaluated: extreme/parked load cases (IEC 50-year gust), dynamic and fatigue loads.",
    "Not evaluated: shell buckling, shear stress, torsion from pitching moment.",
    "No 3D/rotational stall augmentation or dynamic stall.",
]
BLADE_ELEMENT_LIMITATION = (
    "Blade-element model (solver.model = blade_element): no induction (a = a' = 0) and no "
    "tip/hub loss, so each section sees the undisturbed free stream. This overpredicts the angle "
    "of attack, the loads and C_P (which can exceed the Betz limit).")
MODELS = ("bem", "blade_element")


class Geometry(NamedTuple):
    r: jnp.ndarray        # (n,)
    c: jnp.ndarray
    theta: jnp.ndarray    # twist + pitch [rad]
    r_ext: jnp.ndarray    # (n,) or (n+1,) with a zero-load node at R
    sec: SectionProps


class Material(NamedTuple):
    E: float
    density: float        # 0 disables mass terms


@dataclass
class Blade:
    geom: Geometry
    rotor: Rotor
    material: Material
    polar: object
    include_gravity: bool
    include_centrifugal: bool
    model: str                       # "bem" | "blade_element"
    x_over_c: np.ndarray | None      # per contour point (None for user sections)
    is_upper: np.ndarray | None
    warnings: list[str] = field(default_factory=list)


def build_blade(cfg: dict) -> Blade:
    warnings: list[str] = []
    af, rot, st, s = cfg["airfoil"], cfg["rotor"], cfg["stations"], cfg["structure"]
    r = np.asarray(st["r"], float)
    c = np.asarray(st["chord"], float)
    theta = np.deg2rad(np.asarray(st["theta_deg"], float) + rot["pitch_deg"])
    R, r_hub = float(rot["R"]), float(rot["r_hub"])
    model = cfg["solver"]["model"]

    if model == "blade_element":
        # Loads at every given station, trapezoid over those stations only (write-up §4).
        r_ext = r
        if r[-1] < R:
            warnings.append(f"r[-1] = {r[-1]:g} m < R = {R:g} m: blade-element model integrates "
                            "over the given stations only, so the load between r[-1] and R is "
                            "ignored.")
    else:
        r_ext = np.r_[r, R] if r[-1] < R else r
    if r[0] > r_hub:
        warnings.append(f"r[0] = {r[0]:g} m > r_hub = {r_hub:g} m: moments at r[0] are not the "
                        "true root moment (load between hub and r[0] is ignored).")

    # Airfoil + polar
    contour = naca.naca4_contour(af["naca"], int(af["n_points_per_surface"]))
    if af["polar_file"]:
        polar, w = load_polar_table(af["polar_file"])
        warnings += w
    else:
        cd0 = np.nan if af["cd0"] is None else float(af["cd0"])
        polar = FallbackPolar(alpha_L0=naca.zero_lift_angle(af["naca"]),
                              alpha_stall=float(np.deg2rad(af["alpha_stall_deg"])), cd0=cd0)
        warnings.append("Fallback aero (thin-airfoil lift, flat-plate drag, flat-plate post-stall): "
                        "rough model; supply airfoil.polar_file for real polars.")

    # Sections
    x_over_c = is_upper = None
    if s["model"] == "user":
        u = s["user_section_props"]
        frame_chord = u.get("frame", "rotor") == "chord"
        n = r.size
        sec = user_sections(u["A"], u["S_11"], u["S_22"], u["S_12"],
                            np.asarray(u["points_1"], float), np.asarray(u["points_2"], float),
                            u.get("centroid_1", np.zeros(n)), u.get("centroid_2", np.zeros(n)),
                            theta=jnp.asarray(theta) if frame_chord else None)
    else:
        t_skin = None
        if s["model"] == "shell":
            t_skin = (np.full_like(c, s["skin_thickness"]) if s["skin_thickness"] is not None
                      else s["skin_thickness_frac"] * c)
            if np.any(t_skin > 0.05 * c):
                warnings.append("Shell skin thickness exceeds 0.05 c at some stations; "
                                "thin-wall model is questionable there.")
            warnings.append("Uniform thin-shell section: a first-order model. Real blades carry "
                            "most bending in spar caps.")
            t_skin = jnp.asarray(t_skin)
        sec = airfoil_sections(jnp.asarray(contour.xy), jnp.asarray(c), jnp.asarray(theta),
                               s["model"], t_skin)
        x_over_c, is_upper = contour.xy[:, 0], contour.is_upper

    m = s["material"]
    density = float(m["density"] or 0.0)
    want_mass = s["include_gravity"] or s["include_centrifugal"]
    if want_mass and density == 0.0:
        warnings.append("material.density not given: gravity and centrifugal terms disabled.")
    if density > 0 and r_ext.size > r.size:
        warnings.append("Mass per length held constant from r[-1] to R for gravity/centrifugal.")

    geom = Geometry(jnp.asarray(r), jnp.asarray(c), jnp.asarray(theta), jnp.asarray(r_ext), sec)
    rotor = Rotor(B=float(rot["B"]), R=R, r_hub=r_hub, rho=float(cfg["air"]["rho"]),
                  mu=float(cfg["air"]["mu"]))
    return Blade(geom, rotor, Material(float(m["E"]), density), polar,
                 bool(s["include_gravity"]) and density > 0,
                 bool(s["include_centrifugal"]) and density > 0,
                 model, x_over_c, is_upper, warnings)


def _pad_tip(x, n_ext):
    """Extend station array with its tip-node value (zero) if the grid has a tip node."""
    return jnp.concatenate([x, jnp.zeros(n_ext - x.shape[0])])


@partial(jax.jit, static_argnames=("gravity", "centrifugal", "blade_element"))
def evaluate(geom: Geometry, rotor: Rotor, mat: Material, polar, U, Omega, Re,
             gravity: bool, centrifugal: bool, blade_element: bool = False):
    """One operating point. Returns a flat dict of arrays (station axis first where relevant)."""
    n, n_ext = geom.r.shape[0], geom.r_ext.shape[0]
    stations = blade_element_stations if blade_element else solve_stations
    s = stations(geom.r, geom.c, geom.theta, Re, U, Omega, rotor, polar)
    pN, pT = _pad_tip(s["pN"], n_ext), _pad_tip(s["pT"], n_ext)
    V_N, M_flap = shear_and_moment(pN, geom.r_ext)
    V_T, M_edge = shear_and_moment(pT, geom.r_ext)
    out = dict(s, **rotor_totals(geom.r_ext, pN, pT, U, Omega, rotor))

    # Mass terms (m' held at its last-station value out to R)
    m_line = mat.density * geom.sec.A
    m_ext = jnp.concatenate([m_line, jnp.full(n_ext - n, m_line[-1])])
    M_grav = shear_and_moment(m_ext * G, geom.r_ext)[1] if gravity else jnp.zeros(n_ext)
    N = (integrate_from_tip(m_ext * Omega**2 * geom.r_ext, geom.r_ext) if centrifugal
         else jnp.zeros(n_ext))

    # Load cases: gravity along +t and -t (blade horizontal); one case if gravity is off.
    signs = jnp.array([1.0, -1.0]) if gravity else jnp.array([0.0])
    M_t = -M_flap[:n]
    M_n = M_edge[:n][None, :] + signs[:, None] * M_grav[:n][None, :]    # (case, n)
    sec = geom.sec
    sigma = bending_stress(sec.pts_t[None], sec.pts_n[None], M_t[None], M_n, N[:n][None],
                           sec.A, sec.S_tt, sec.S_nn, sec.S_tn)          # (case, n, pts)
    kn, kt = curvatures(M_t[None], M_n, mat.E, sec.S_tt, sec.S_nn, sec.S_tn)
    pad = lambda k: jnp.concatenate([k, jnp.zeros((k.shape[0], n_ext - n))], axis=1)
    d_n = jax.vmap(deflection, in_axes=(0, None))(pad(kn), geom.r_ext)
    d_t = jax.vmap(deflection, in_axes=(0, None))(pad(kt), geom.r_ext)

    flat = sigma.transpose(1, 0, 2).reshape(n, -1)                       # (n, case*pts)
    i_max, i_min = jnp.argmax(flat, axis=1), jnp.argmin(flat, axis=1)
    n_pts = sigma.shape[2]
    out.update(
        pN_ext=pN, pT_ext=pT, V_N=V_N, V_T=V_T, M_flap=M_flap, M_edge=M_edge, M_grav=M_grav, N=N,
        sigma_max=jnp.max(flat, axis=1), sigma_min=jnp.min(flat, axis=1),
        sigma_max_pt=i_max % n_pts, sigma_max_case=i_max // n_pts,
        sigma_min_pt=i_min % n_pts, sigma_min_case=i_min // n_pts,
        delta_n=d_n, delta_t=d_t, signs=signs,
    )
    return out


_evaluate_sweep = jax.vmap(evaluate, in_axes=(None, None, None, None, 0, 0, 0, None, None, None))


@dataclass
class Result:
    cfg: dict
    blade: Blade
    U: np.ndarray            # (n_op,)
    Omega: np.ndarray
    out: dict                # numpy arrays, leading axis n_op
    re_passes: int
    warnings: list[str]
    # structured per-operating-point flags: (kind, op_index, stations r [m]); text is in warnings
    flags: list[tuple[str, int, list[float]]] = field(default_factory=list)
    freewheel: bool = False  # True when Omega is the solved freewheeling speed (Q = 0)


def rpm(res: Result) -> np.ndarray:
    return res.Omega * 60 / (2 * np.pi)


def op_label(res: Result, i: int, file: bool = False) -> str:
    """Name operating point i by whatever varies across the sweep (U, RPM, or both)."""
    vary_u = np.ptp(res.U) > 0
    vary_w = np.ptp(res.Omega) > 0 or not vary_u
    parts = []
    if vary_u:
        parts.append(f"U{res.U[i]:g}" if file else f"U = {res.U[i]:g} m/s")
    if vary_w:
        parts.append(f"rpm{rpm(res)[i]:.0f}" if file else f"{rpm(res)[i]:.0f} RPM")
    return ("_" if file else ", ").join(parts)


def _solve(blade: Blade, U, Om, solver: dict):
    """Evaluate every operating point (vectorized), with the Reynolds outer loop (§5).

    Returns (out, Re, passes, converged). In blade-element mode W does not depend on the
    solution, so the initial Re is already exact and one pass suffices."""
    g = blade.geom
    U, Om = np.atleast_1d(np.asarray(U, float)), np.atleast_1d(np.asarray(Om, float))
    Re = (blade.rotor.rho * np.sqrt(U[:, None] ** 2 + (Om[:, None] * np.asarray(g.r)) ** 2)
          * np.asarray(g.c) / blade.rotor.mu)
    tol, max_passes = solver["re_tol"], int(solver["re_max_passes"])
    for k in range(1, max_passes + 1):
        out = _evaluate_sweep(g, blade.rotor, blade.material, blade.polar, jnp.asarray(U),
                              jnp.asarray(Om), jnp.asarray(Re), blade.include_gravity,
                              blade.include_centrifugal, blade.model == "blade_element")
        Re_new = np.asarray(out["Re_new"])
        change = np.max(np.abs(Re_new / Re - 1.0))
        if change < tol:
            break
        Re = Re_new
    return out, Re, k, change < tol


def rotor_torque(blade: Blade, U: float, Om, solver: dict) -> np.ndarray:
    """Rotor shaft torque Q(Omega) [N m] at wind speed U for each Omega in ``Om``."""
    Om = np.atleast_1d(np.asarray(Om, float))
    out = _solve(blade, np.full_like(Om, U), Om, solver)[0]
    return np.asarray(out["Q"])


def freewheel_omega(blade: Blade, U: float, solver: dict) -> float:
    """Freewheeling speed Omega_ss [rad/s]: the root of Q(Omega) = 0 (write-up §5).

    Scans ``solver.freewheel_rpm`` = [min, max, step] for the last sign change of Q from + to -
    (the rotor speeds up while Q > 0), then bisects that bracket to ``freewheel_tol_rpm``."""
    lo_rpm, hi_rpm, step = (float(v) for v in solver["freewheel_rpm"])
    to_om = 2 * np.pi / 60
    grid = np.arange(lo_rpm, hi_rpm + 0.5 * step, step) * to_om
    Q = rotor_torque(blade, U, grid, solver)
    down = np.where((Q[:-1] > 0) & (Q[1:] <= 0))[0]
    if down.size == 0:
        raise ValueError(f"freewheel: Q(Omega) has no + to - sign change for U = {U:g} m/s in "
                         f"{lo_rpm:g}-{hi_rpm:g} RPM; widen solver.freewheel_rpm.")
    lo, hi = grid[down[-1]], grid[down[-1] + 1]
    tol = float(solver["freewheel_tol_rpm"]) * to_om
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if rotor_torque(blade, U, mid, solver)[0] > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run(cfg: dict) -> Result:
    """Run every operating point in ``cfg`` (vectorized), with the Reynolds outer loop (§5).

    With ``operating.rpm: freewheel`` the rotor speed at each U_inf is first solved from
    Q(Omega) = 0, and the full model is then evaluated there."""
    blade = build_blade(cfg)
    freewheel = cfg["operating"]["rpm"] == "freewheel"
    if freewheel:
        U = np.atleast_1d(np.asarray(cfg["operating"]["U_inf"], float))
        Om = np.array([freewheel_omega(blade, u, cfg["solver"]) for u in U])
    else:
        U, Om = operating_points(cfg)
    out, Re, k, converged = _solve(blade, U, Om, cfg["solver"])
    out = jax.tree_util.tree_map(np.asarray, out)
    out["Re"] = Re
    warnings = list(blade.warnings)
    if blade.model == "blade_element":
        warnings.append(BLADE_ELEMENT_LIMITATION)
    if not converged:
        warnings.append(f"Reynolds loop not converged after {k} passes.")
    res = Result(cfg, blade, U, Om, out, k, warnings, freewheel=freewheel)
    res.warnings += _flags(res)
    return res


def _flags(res: Result) -> list[str]:
    cfg, blade, U, out = res.cfg, res.blade, res.U, res.out
    w = []
    r = np.asarray(blade.geom.r)
    alpha_stall = np.deg2rad(cfg["airfoil"]["alpha_stall_deg"])
    in_table = np.asarray(blade.polar.in_table(jnp.asarray(out["alpha"]), jnp.asarray(out["Re"])))
    solved = out["region"] > 0
    out["stall"] = (np.abs(out["alpha"]) > alpha_stall) & solved
    out["aero_model"] = np.where(blade.polar.kind == "fallback", "fallback",
                                 np.where(in_table, "table", "table-extrapolated"))
    def add(kind, i, rs, text):
        res.flags.append((kind, i, [float(v) for v in np.atleast_1d(rs)]))
        w.append(f"[{op_label(res, i)}] {text}")

    for i in range(len(U)):
        bad = r[out["region"][i] == 0]
        if bad.size:
            add("no_bracket", i, bad, f"no BEM bracket found at r = {bad.round(3).tolist()} "
                "(loads set to 0).")
        odd = r[out["region"][i] >= 2]
        if odd.size:
            add("region", i, odd, f"phi outside (0, 90] deg (unusual state) at r = "
                f"{odd.round(3).tolist()}.")
        resid = r[np.abs(out["res"][i]) > 1e-6]
        if resid.size:
            add("residual", i, resid, f"BEM residual > 1e-6 (likely a pole, not a root) at r = "
                f"{resid.round(3).tolist()}.")
        st = r[out["stall"][i]]
        if st.size:
            add("stall", i, st, f"|alpha| > alpha_stall at r = {st.round(3).tolist()}.")
        ext = r[(out["aero_model"][i] == "table-extrapolated") & solved[i]]
        if ext.size:
            add("extrapolated", i, ext, f"polar extrapolated (alpha or Re outside table) at r = "
                f"{ext.round(3).tolist()}.")
        if out["CP"][i] >= 16 / 27:
            if blade.model == "blade_element":
                add("betz_be", i, [], f"C_P = {out['CP'][i]:.3f} exceeds the Betz limit, as "
                    "expected without induction (blade-element model).")
            else:
                add("betz", i, [], f"C_P = {out['CP'][i]:.3f} exceeds the Betz limit: check "
                    "inputs.")
        tip = np.max(np.abs(out["delta_n"][i][:, -1]))
        if tip > 0.1 * blade.rotor.R:
            add("deflection", i, [], f"tip deflection {tip:.2f} m > 0.1 R: rigid-blade "
                "assumption questionable.")
    return w
