"""Steady BEM solve (PLAN.md §5): Ning (2014) single-residual formulation in phi.

Everything here is pure JAX. A station solve is a scalar function that is ``vmap``-ed over
stations (and, in ``model.py``, again over operating points), then ``jit``-ed.

Root finding: the spec asks for a bracketing method (Brent). Brent's data-dependent control flow
vectorizes poorly, so we use fixed-iteration bisection on the same brackets: equally guaranteed,
branch-free under vmap, and 80 halvings of a <= pi/2 bracket reach float64 resolution.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

EPS = 1e-6
N_BISECT = 80

# Bracket regions, tried in order (§5): 1 = normal, 2 = propeller brake, 3 = phi > 90 deg.
BRACKETS = ((EPS, jnp.pi / 2), (-jnp.pi / 4, -EPS), (jnp.pi / 2, jnp.pi - EPS))


class Rotor(NamedTuple):
    B: float
    R: float
    r_hub: float
    rho: float
    mu: float


def prandtl_F(r, phi, rotor: Rotor):
    s = jnp.abs(jnp.sin(phi))
    f_tip = 2 / jnp.pi * jnp.arccos(jnp.exp(-rotor.B * (rotor.R - r) / (2 * r * s)))
    hub_arg = rotor.B * (r - rotor.r_hub) / (2 * jnp.maximum(rotor.r_hub, 1e-12) * s)
    f_hub = jnp.where(rotor.r_hub > 0, 2 / jnp.pi * jnp.arccos(jnp.exp(-hub_arg)), 1.0)
    return f_tip * f_hub


def station_state(phi, r, c, theta, Re, U, Omega, rotor: Rotor, polar):
    """Everything that follows from phi at one station. Returns a dict incl. the residual."""
    sphi, cphi = jnp.sin(phi), jnp.cos(phi)
    alpha = phi - theta
    cl, cd = polar.cl_cd(alpha, Re)
    cn = cl * cphi + cd * sphi
    ct = cl * sphi - cd * cphi
    F = prandtl_F(r, phi, rotor)
    sigma_p = rotor.B * c / (2 * jnp.pi * r)
    k = sigma_p * cn / (4 * F * sphi**2)
    kp = sigma_p * ct / (4 * F * sphi * cphi)

    # Momentum region + Buhl (2005) high-induction correction (phi > 0).
    g1 = 2 * F * k - (10 / 9 - F)
    g2 = 2 * F * k - F * (4 / 3 - F)
    g3 = 2 * F * k - (25 / 9 - 2 * F)
    a_buhl = jnp.where(jnp.abs(g3) < 1e-6, 1 - 1 / (2 * jnp.sqrt(g2)), (g1 - jnp.sqrt(g2)) / g3)
    a_pos = jnp.where(k <= 2 / 3, k / (1 + k), a_buhl)
    # Propeller-brake region (phi < 0): momentum theory gives a = k/(k-1), so
    # sin(phi)/(1-a) = sin(phi)(1-k). Same branch as Ning (2014) / CCBlade; region 2 is flagged.
    a_neg = jnp.where(k > 1, k / (k - 1), 0.0)
    a = jnp.where(phi > 0, a_pos, a_neg)
    ap = kp / (1 - kp)

    lam_r = Omega * r / U
    res = jnp.where(phi > 0,
                    sphi / (1 - a) - cphi * (1 - kp) / lam_r,
                    sphi * (1 - k) - cphi * (1 - kp) / lam_r)
    return dict(phi=phi, alpha=alpha, cl=cl, cd=cd, cn=cn, ct=ct, F=F, a=a, ap=ap, res=res)


def _bisect(f, lo, hi):
    f_lo = f(lo)

    def body(_, s):
        lo, hi, f_lo = s
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        same = jnp.sign(f_mid) == jnp.sign(f_lo)
        return (jnp.where(same, mid, lo), jnp.where(same, hi, mid), jnp.where(same, f_mid, f_lo))

    lo, hi, _ = jax.lax.fori_loop(0, N_BISECT, body, (lo, hi, f_lo))
    return 0.5 * (lo + hi)


def solve_station(r, c, theta, Re, U, Omega, rotor: Rotor, polar):
    """Solve one station at fixed Re. Returns state dict + W, region (0 = no bracket found)."""
    res = lambda phi: station_state(phi, r, c, theta, Re, U, Omega, rotor, polar)["res"]

    phi, region = jnp.nan, 0
    for i, (lo, hi) in reversed(list(enumerate(BRACKETS, start=1))):  # first valid wins
        ok = res(lo) * res(hi) <= 0
        phi = jnp.where(ok, _bisect(res, lo, hi), phi)
        region = jnp.where(ok, i, region)

    s = station_state(phi, r, c, theta, Re, U, Omega, rotor, polar)
    s["W"] = jnp.sqrt((U * (1 - s["a"])) ** 2 + (Omega * r * (1 + s["ap"])) ** 2)
    s["region"] = region
    return s


def solve_stations(r, c, theta, Re, U, Omega, rotor: Rotor, polar):
    """vmap over stations. Stations at r <= r_hub or r >= R (F = 0) carry zero load."""
    active = (r > rotor.r_hub * (1 + 1e-9)) & (r < rotor.R * (1 - 1e-9))
    r_safe = jnp.clip(r, rotor.r_hub + 1e-6 * rotor.R, rotor.R * (1 - 1e-6))
    s = jax.vmap(solve_station, in_axes=(0, 0, 0, 0, None, None, None, None))(
        r_safe, c, theta, Re, U, Omega, rotor, polar)

    found = active & (s["region"] > 0)
    phi_geo = jnp.arctan2(U, Omega * r)
    zero = jnp.zeros_like(r)
    s["phi"] = jnp.where(found, s["phi"], phi_geo)
    s["alpha"] = s["phi"] - theta
    for key in ("a", "ap", "F", "cn", "ct", "res"):
        s[key] = jnp.where(found, s[key], zero)
    s["W"] = jnp.where(found, s["W"], jnp.hypot(U, Omega * r))
    cl_g, cd_g = polar.cl_cd(s["alpha"], Re)
    s["cl"] = jnp.where(found, s["cl"], cl_g)
    s["cd"] = jnp.where(found, s["cd"], cd_g)
    s["region"] = jnp.where(active, s["region"], -1)  # -1 = hub/tip station, not solved

    q = 0.5 * rotor.rho * s["W"] ** 2
    s["pN"] = q * c * s["cn"]
    s["pT"] = q * c * s["ct"]
    s["Re_new"] = rotor.rho * s["W"] * c / rotor.mu
    return s


def rotor_totals(r_ext, pN_ext, pT_ext, U, Omega, rotor: Rotor):
    """Thrust, torque, power, C_T, C_P from per-blade loads on the tip-extended grid."""
    T = rotor.B * jnp.trapezoid(pN_ext, r_ext)
    Q = rotor.B * jnp.trapezoid(pT_ext * r_ext, r_ext)
    P = Q * Omega
    area = jnp.pi * rotor.R**2
    return dict(T=T, Q=Q, P=P,
                CT=T / (0.5 * rotor.rho * U**2 * area),
                CP=P / (0.5 * rotor.rho * U**3 * area))
