# /// script
# requires-python = ">=3.11"
# dependencies = ["jax", "numpy", "matplotlib"]
# ///
"""MAE 4272 Lab 5 - Blade #7 loads and material requirements (Lucas Libshutz).

Reproduces the numbers in the report: the free-rotation speed, the blade loads, the stress
along the span, and the minimum material properties, without (blade element) and with
(blade element momentum) induction.

HOW TO RUN
  1. Install uv, a small tool that sets up Python for you (one time only):
       Mac / Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
       Windows:      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  2. Put this file and naca4412_airfoiltools.csv in the same folder.
  3. Open a terminal in that folder and run:
       uv run lab5_blade7.py
     The first run downloads what it needs (about a minute). It prints a summary of the
     results, and saves the stress plot as blade7_stress.png next to this file.
"""

from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
POLAR_FILE = HERE / "naca4412_airfoiltools.csv"   # XFOIL, Ncrit = 9, from airfoiltools.com

# Blade #7 geometry sheet (Canvas): r [m], pitch [deg], chord [m]
BLADE7 = np.array([
    [0.0254, 39.4920573333638, 0.07024245987344668],
    [0.03342105263157895, 31.688541477831933, 0.0604595002422356],
    [0.04144210526315789, 25.872064263159608, 0.05242841414500866],
    [0.04946315789473685, 21.45351882048194, 0.04596206070820924],
    [0.05748421052631579, 18.020744185203682, 0.04074772742632171],
    [0.06550526315789473, 15.295086191910652, 0.03650241484550876],
    [0.07352631578947369, 13.087740059645498, 0.033003412970697525],
    [0.08154736842105263, 11.268731492136542, 0.030082947061445115],
    [0.08956842105263157, 9.746735361997278, 0.027615921415814645],
    [0.09758947368421053, 8.456205806883304, 0.025508747847355632],
    [0.10561052631578946, 7.34912897783139, 0.023690762709940506],
    [0.11363157894736842, 6.3896594367421535, 0.022107991959150006],
    [0.12165263157894737, 5.550566788445537, 0.020718701825087546],
    [0.12967368421052633, 4.810835297500706, 0.019490229283951718],
    [0.13769473684210526, 4.154011001596065, 0.018396711376318667],
    [0.14571578947368422, 3.5670425562185843, 0.01741744347642622],
    [0.15373684210526317, 3.0394542677219776, 0.01653567948680117],
    [0.16175789473684213, 2.5627466243192036, 0.01573774504732547],
    [0.16977894736842108, 2.1299552632542356, 0.015012374654565846],
    [0.1778, 1.735322027401453, 0.014350210672250029],
])

U_INF = 15.0                       # wind speed [m/s]
RHO, MU, A_SOUND = 1.225, 1.81e-5, 343.0
N_B = 3
RHO_MAT = 1180.0                   # typical cured SLA resin [kg/m^3]
E_REF = 2.201e9                    # Formlabs Grey Resin V4.1, post-cured [Pa]
G = 9.81
DEFLECTION_LIMIT = 0.1             # allowed tip deflection, fraction of R
FACTOR_OF_SAFETY = 2.0
RPM = 2 * np.pi / 60


def pytree(cls):
    """Frozen dataclass whose fields are all JAX leaves."""
    cls = dataclass(frozen=True)(cls)
    jax.tree_util.register_dataclass(cls, data_fields=list(cls.__dataclass_fields__),
                                     meta_fields=[])
    return cls


@pytree
class Stations:
    r: jax.Array
    c: jax.Array
    theta: jax.Array


@pytree
class Sections:
    A: jax.Array       # (n,)
    J: jax.Array       # (n, 2, 2) second-moment tensor about the centroid, rotor frame
    pts: jax.Array     # (n, k, 2) surface points relative to the centroid, rotor frame


@pytree
class Rotor:
    R: float
    r_hub: float


@pytree
class Polar:
    log_re: jax.Array  # (nRe,)
    alpha: jax.Array   # (nA,) [rad], -180 to 180 deg
    cl: jax.Array      # (nRe, nA)
    cd: jax.Array

    def __call__(self, alpha, Re):
        @jnp.vectorize
        def one(a, re):
            a = (a + jnp.pi) % (2 * jnp.pi) - jnp.pi
            lookup = lambda tab: jnp.interp(
                jnp.log(re), self.log_re, jax.vmap(lambda row: jnp.interp(a, self.alpha, row))(tab))
            return lookup(self.cl), lookup(self.cd)

        return one(alpha, Re)


# --- inputs -----------------------------------------------------------------------------------

def load_polar(path=POLAR_FILE, blend_deg=10.0):
    """Tabulated polars, blended linearly into a flat plate outside the tabulated range."""
    re, a, cl, cd = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    grid_deg = np.arange(-180.0, 180.0 + 1e-9, 0.1)
    grid = np.deg2rad(grid_deg)
    res = np.unique(re)
    cls, cds = [], []
    for x in res:
        o = np.argsort(a[re == x])
        ai, cli, cdi = a[re == x][o], cl[re == x][o], cd[re == x][o]
        f = np.clip(np.maximum(ai[0] - grid_deg, grid_deg - ai[-1]) / blend_deg, 0.0, 1.0)
        cls.append((1 - f) * np.interp(grid_deg, ai, cli) + f * np.sin(2 * grid))
        cds.append((1 - f) * np.interp(grid_deg, ai, cdi) + f * (cdi.min() + 2 * np.sin(grid) ** 2))
    return Polar(jnp.log(res), jnp.asarray(grid), jnp.asarray(cls), jnp.asarray(cds))


def naca4412(n=150):
    """Closed CCW contour, unit chord: TE -> upper surface -> LE -> lower surface."""
    m, p, t = 0.04, 0.4, 0.12
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    psi = np.arctan(np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x)))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3
                  - 0.1036 * x**4)
    upper = np.column_stack([x - yt * np.sin(psi), yc + yt * np.cos(psi)])[::-1]
    lower = np.column_stack([x + yt * np.sin(psi), yc - yt * np.cos(psi)])[1:-1]
    return np.vstack([upper, lower]), np.r_[np.ones(n, bool), np.zeros(n - 2, bool)]


@jax.vmap
def section(xy, c, theta):
    """Area, second-moment tensor and centroid-relative points of the scaled, pitched profile."""
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    p = c * xy @ R.T
    q = jnp.roll(p, -1, axis=0)
    s = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    A = 0.5 * s.sum()
    p_bar = (s[:, None] * (p + q)).sum(0) / (6 * A)
    outer = lambda u, v: u[:, :, None] * v[:, None, :]
    J_O = (s[:, None, None] * (outer(p, p) + outer(q, q)
                               + 0.5 * (outer(p, q) + outer(q, p)))).sum(0) / 12
    return A, J_O - A * jnp.outer(p_bar, p_bar), p - p_bar


# --- aerodynamics -----------------------------------------------------------------------------

def blade_element(st, rotor, polar, Om, Re):
    """No induction: every section sees the undisturbed wind."""
    phi = jnp.arctan2(U_INF, Om * st.r)
    cl, cd = polar(phi - st.theta, Re)
    return phi, jnp.hypot(U_INF, Om * st.r), cl, cd, jnp.zeros_like(st.r), jnp.ones_like(st.r, bool)


BRACKETS = ((1e-6, jnp.pi / 2), (-jnp.pi / 4, -1e-6), (jnp.pi / 2, jnp.pi - 1e-6))


def bisect(f, lo, hi, n=80):
    def step(_, s):
        lo, hi, f_lo = s
        mid = 0.5 * (lo + hi)
        same = jnp.sign(f(mid)) == jnp.sign(f_lo)
        return jnp.where(same, mid, lo), jnp.where(same, hi, mid), jnp.where(same, f(mid), f_lo)

    lo, hi, _ = jax.lax.fori_loop(0, n, step, (lo, hi, f(lo)))
    return 0.5 * (lo + hi)


def bem(st, rotor, polar, Om, Re):
    """Blade element momentum: induction factors a, a' from momentum balance on each annulus
    (Prandtl tip/hub loss, Buhl high-induction correction, single residual in phi)."""

    def state(phi, r, c, theta, re):
        sp, cp = jnp.sin(phi), jnp.cos(phi)
        cl, cd = polar(phi - theta, re)
        cn, ct = cl * cp + cd * sp, cl * sp - cd * cp
        loss = lambda d: 2 / jnp.pi * jnp.arccos(jnp.exp(-N_B * d / (2 * jnp.abs(sp))))
        F = loss((rotor.R - r) / r) * loss((r - rotor.r_hub) / rotor.r_hub)
        sigma = N_B * c / (2 * jnp.pi * r)
        k, kp = sigma * cn / (4 * F * sp**2), sigma * ct / (4 * F * sp * cp)
        g1, g2, g3 = 2 * F * k - (10 / 9 - F), 2 * F * k - F * (4 / 3 - F), 2 * F * k - (25 / 9 - 2 * F)
        a_buhl = jnp.where(jnp.abs(g3) < 1e-6, 1 - 1 / (2 * jnp.sqrt(g2)), (g1 - jnp.sqrt(g2)) / g3)
        a = jnp.where(phi > 0, jnp.where(k <= 2 / 3, k / (1 + k), a_buhl),
                      jnp.where(k > 1, k / (k - 1), 0.0))
        ap = kp / (1 - kp)
        res = jnp.where(phi > 0, sp / (1 - a), sp * (1 - k)) - cp * (1 - kp) * U_INF / (Om * r)
        return res, a, ap, cl, cd

    def one(r, c, theta, re):
        f = lambda phi: state(phi, r, c, theta, re)[0]
        phi, found = jnp.nan, False
        for lo, hi in reversed(BRACKETS):
            ok = f(lo) * f(hi) <= 0
            phi, found = jnp.where(ok, bisect(f, lo, hi), phi), found | ok
        _, a, ap, cl, cd = state(phi, r, c, theta, re)
        return phi, jnp.hypot(U_INF * (1 - a), Om * r * (1 + ap)), cl, cd, a, found

    r_safe = jnp.clip(st.r, rotor.r_hub + 1e-6 * rotor.R, rotor.R * (1 - 1e-6))
    phi, W, cl, cd, a, found = jax.vmap(one)(r_safe, st.c, st.theta, Re)
    on = found & (st.r > rotor.r_hub * (1 + 1e-9)) & (st.r < rotor.R * (1 - 1e-9))
    return phi, jnp.where(on, W, jnp.hypot(U_INF, Om * st.r)), cl, cd, jnp.where(on, a, 0.0), on


def _loads(st, rotor, polar, Om, Re, model):
    phi, W, cl, cd, a, on = model(st, rotor, polar, Om, Re)
    q = 0.5 * RHO * W**2 * st.c
    p_Q = jnp.where(on, q * (cl * jnp.sin(phi) - cd * jnp.cos(phi)), 0.0)
    p_T = jnp.where(on, q * (cl * jnp.cos(phi) + cd * jnp.sin(phi)), 0.0)
    return dict(p_Q=p_Q, p_T=p_T, W=W, a=a, alpha=phi - st.theta)


def _batched(model):
    f = lambda st, rotor, polar, Om, Re: _loads(st, rotor, polar, Om, Re, model)
    return jax.jit(jax.vmap(f, in_axes=(None, None, None, 0, 0)))


_LOADS = {m: _batched(m) for m in (blade_element, bem)}


def loads(st, rotor, polar, Om, model):
    """Sectional loads at each rotor speed in Om, iterating on the Reynolds number."""
    Om = jnp.atleast_1d(jnp.asarray(Om, float))
    Re = RHO * jnp.hypot(U_INF, Om[:, None] * st.r) * st.c / MU
    for _ in range(10):
        out = _LOADS[model](st, rotor, polar, Om, Re)
        Re_new = RHO * out["W"] * st.c / MU
        if jnp.max(jnp.abs(Re_new / Re - 1)) < 0.01:
            break
        Re = Re_new
    return out


def trapezoid(r):
    dr = jnp.diff(r)
    return 0.5 * (jnp.r_[dr, 0.0] + jnp.r_[0.0, dr])


def torque(st, rotor, polar, Om, model, w=None):
    w = trapezoid(st.r) if w is None else w
    return np.asarray(N_B * (loads(st, rotor, polar, Om, model)["p_Q"] * st.r) @ w)


def free_rotation_speed(st, rotor, polar, model, w=None):
    """Root of Q(Omega) = 0: scan for the last + to - sign change, then bisect to 1 RPM."""
    grid = np.arange(1000.0, 60000.0 + 125.0, 250.0) * RPM
    Q = torque(st, rotor, polar, grid, model, w)
    i = np.where((Q[:-1] > 0) & (Q[1:] <= 0))[0][-1]
    lo, hi = grid[i], grid[i + 1]
    while hi - lo > RPM:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if torque(st, rotor, polar, mid, model, w)[0] > 0 else (lo, mid)
    return 0.5 * (lo + hi)


# --- structure --------------------------------------------------------------------------------

def from_tip(y, r):
    seg = 0.5 * (y[1:] + y[:-1]) * jnp.diff(r)
    return jnp.r_[jnp.cumsum(seg[::-1])[::-1], 0.0]


def from_root(y, r):
    seg = 0.5 * (y[1:] + y[:-1]) * jnp.diff(r)
    return jnp.r_[0.0, jnp.cumsum(seg)]


def moment(p, r):
    """M(r0) = int_{r0}^R (r - r0) p dr, as shear then moment from the tip inward."""
    return from_tip(from_tip(p, r), r)


@jax.jit
def stress_and_deflection(st, sec, p_Q, p_T, Om):
    """Normal stress at every surface point and tip deflection, for gravity acting along +/- the
    edgewise direction (the blade pointing horizontally, either side of the hub)."""
    M_x, M_y = moment(p_T, st.r), moment(p_Q, st.r)
    m_line = RHO_MAT * sec.A
    N = from_tip(m_line * Om**2 * st.r, st.r)
    M_grav = moment(m_line * G, st.r)
    J_inv = jnp.linalg.inv(sec.J)

    def case(sign):
        M_e = M_y + sign * M_grav
        g = jnp.einsum("nij,nj->ni", J_inv, jnp.stack([M_e, -M_x], -1))
        sigma = N[:, None] / sec.A[:, None] + jnp.einsum("nki,ni->nk", sec.pts, g)
        kappa = jnp.einsum("nij,nj->ni", J_inv, jnp.stack([-M_e, M_x], -1)) / E_REF
        delta_tip = from_root(from_root(kappa[:, 1], st.r), st.r)[-1]
        return sigma, delta_tip

    sigma, delta_tip = jax.vmap(case)(jnp.array([1.0, -1.0]))
    return dict(sigma=sigma, delta_tip=jnp.max(jnp.abs(delta_tip)), M_x=M_x, M_y=M_y, N=N,
                M_grav=M_grav)


def analyze(st, sec, rotor, polar, model):
    Om = free_rotation_speed(st, rotor, polar, model)
    aero = jax.tree_util.tree_map(lambda x: x[0], loads(st, rotor, polar, Om, model))
    s = stress_and_deflection(st, sec, aero["p_Q"], aero["p_T"], Om)
    sigma = np.asarray(s["sigma"])
    tension = np.unravel_index(sigma.argmax(), sigma.shape)
    compression = np.unravel_index(sigma.argmin(), sigma.shape)
    return dict(
        Om=Om, aero=aero, thrust=float(N_B * trapezoid(st.r) @ aero["p_T"]),
        N_root=float(s["N"][0]), M_x_root=float(s["M_x"][0]), M_y_root=float(s["M_y"][0]),
        M_grav_root=float(s["M_grav"][0]),
        sigma_t=sigma.max(), at_t=tension[1:], sigma_c=-sigma.min(), at_c=compression[1:],
        sigma_max_span=sigma.max(axis=(0, 2)), sigma_min_span=sigma.min(axis=(0, 2)),
        delta_tip=float(s["delta_tip"]),
        E_min=E_REF * float(s["delta_tip"]) / (DEFLECTION_LIMIT * rotor.R),
    )


# --- report -----------------------------------------------------------------------------------

def main():
    r, pitch_deg, c = BLADE7.T
    st = Stations(jnp.asarray(r), jnp.asarray(c), jnp.deg2rad(jnp.asarray(pitch_deg)))
    rotor = Rotor(R=float(r[-1]), r_hub=float(r[0]))
    xy, is_upper = naca4412()
    A, J, pts = section(jnp.broadcast_to(jnp.asarray(xy), (r.size, *xy.shape)), st.c, st.theta)
    sec = Sections(A, J, pts)
    polar = load_polar()

    print("Blade #7, NACA 4412, 3 blades, U = 15 m/s, free rotation (no load, no friction)\n")
    be, bm = analyze(st, sec, rotor, polar, blade_element), analyze(st, sec, rotor, polar, bem)

    def where(res, key):
        j, k = res[key]
        side = "upper (downwind)" if is_upper[k] else "lower (upwind)"
        return f"r = {r[j] * 1e3:.1f} mm, {side} surface, x/c = {xy[k, 0]:.2f}"

    rows = [
        ("Free-rotation speed [RPM]", lambda x: x["Om"] / RPM, "{:.0f}"),
        ("Rotor thrust [N]", lambda x: x["thrust"], "{:.1f}"),
        ("Root centrifugal tension N [N]", lambda x: x["N_root"], "{:.0f}"),
        ("Root flapwise moment M_x [N m]", lambda x: x["M_x_root"], "{:.2f}"),
        ("Root edgewise moment M_y [N m]", lambda x: x["M_y_root"], "{:.2g}"),
        ("Required tensile strength [MPa]", lambda x: x["sigma_t"] / 1e6, "{:.1f}"),
        ("Required compressive strength [MPa]", lambda x: x["sigma_c"] / 1e6, "{:.1f}"),
        ("Tip deflection at E = 2.2 GPa [mm]", lambda x: x["delta_tip"] * 1e3, "{:.1f}"),
        ("Minimum modulus, tip <= 0.1 R [GPa]", lambda x: x["E_min"] / 1e9, "{:.1f}"),
        ("Tip Mach number", lambda x: float(x["aero"]["W"][-1]) / A_SOUND, "{:.2f}"),
        ("Mean axial induction a", lambda x: float(jnp.mean(x["aero"]["a"][1:-1])), "{:.2f}"),
    ]
    print(f"{'':38s}{'blade element':>15s}{'with induction':>17s}")
    for name, get, fmt in rows:
        print(f"{name:38s}{fmt.format(get(be)):>15s}{fmt.format(get(bm)):>17s}")
    print(f"\nMax tension, blade element:      {where(be, 'at_t')}")
    print(f"Max compression, blade element:  {where(be, 'at_c')}")
    print(f"Gravity moment at the root:      {be['M_grav_root']:.3f} N m")

    print(f"\nRecommended (with induction, factor of safety {FACTOR_OF_SAFETY:g}):")
    print(f"  ultimate tensile strength >= {FACTOR_OF_SAFETY * bm['sigma_t'] / 1e6:.0f} MPa, "
          f"tensile modulus >= {bm['E_min'] / 1e9:.1f} GPa")

    print("\nChecks quoted in the report (blade element):")
    sweep = np.arange(1000.0, be["Om"] / RPM, 250.0) * RPM
    C_P = torque(st, rotor, polar, sweep, blade_element) * sweep / (
        0.5 * RHO * U_INF**3 * np.pi * rotor.R**2)
    print(f"  peak power coefficient C_P = {C_P.max():.2f} at {sweep[C_P.argmax()] / RPM:.0f} RPM")
    dr = jnp.diff(st.r)
    for name, w in (("left", jnp.r_[dr, 0.0]), ("right", jnp.r_[0.0, dr])):
        Om = free_rotation_speed(st, rotor, polar, blade_element, w)
        print(f"  {name:5s} rule: free-rotation speed {Om / RPM:.0f} RPM")
    mid = jax.tree_util.tree_map(lambda x: 0.5 * (x[1:] + x[:-1]), st)
    Om = free_rotation_speed(mid, rotor, polar, blade_element, dr)
    print(f"  midpoint rule: free-rotation speed {Om / RPM:.0f} RPM")
    for blend in (2.0, 5.0, 10.0, 20.0):
        Om = free_rotation_speed(st, rotor, load_polar(blend_deg=blend), blade_element)
        print(f"  polar blend width {blend:2.0f} deg: free-rotation speed {Om / RPM:.0f} RPM")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for res, label, color in ((be, "blade element", "C0"), (bm, "with induction (BEM)", "C1")):
        ax.plot(r * 1e3, res["sigma_max_span"] / 1e6, color=color, label=f"max. tension, {label}")
        ax.plot(r * 1e3, res["sigma_min_span"] / 1e6, "--", color=color,
                label=f"max. compression, {label}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="r [mm]", ylabel="normal stress [MPa]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "blade7_stress.png", dpi=200)
    print(f"\nSaved {HERE / 'blade7_stress.png'}")


if __name__ == "__main__":
    main()
