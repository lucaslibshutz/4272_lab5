# /// script
# requires-python = ">=3.11"
# dependencies = ["jax", "numpy", "matplotlib"]
# ///
"""MAE 4272 design project - blade sweep for maximum power at minimum material (NACA 4412).

Reuses lab5_blade7.py (polar, BEM with induction, section properties). Every candidate blade is
    twist   theta(r) = phi_Schmitz(r; lambda_d) - alpha_d,  phi_Schmitz = (2/3) atan(R / (lambda_d r))
    chord   c(r)     = linear taper c_root -> c_tip
on 20 stations from the hub face (r = 1 in) to the tip (r = 7 in, blade length 6 in).

Objective (epsilon-constraint): the most shaft power max_Omega Q(Omega) Omega at U = 15 m/s
(BEM, induction on), then, among blades within POWER_FRACTION of the best, the least material.
Constraints (pre-print checklist): blade length <= 6.00 in; axial extent of the blade
< 2.00 in (less AXIAL_MARGIN for the attachment); thinnest section (12 % c at the tip) >= 0.025 in.

Run:  uv run blade_optimize.py   ->  blade_parameters_optimized.csv (Fusion parameter format)
GPU (NVIDIA, CUDA 12):  uv run --with "jax[cuda12]" blade_optimize.py
Outputs land next to this file: blade_parameters_optimized.csv, blade_sweep_results.csv (every
design), blade_sweep_pareto.png.
"""

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np

import lab5_blade7 as L

IN = 0.0254
R_HUB, R_TIP, N_ST = 1.0 * IN, 7.0 * IN, 20
AXIAL_LIMIT, AXIAL_MARGIN = 2.0 * IN, 0.10 * IN
MIN_FEATURE = 0.025 * IN
POWER_FRACTION = 0.95
UTS_GREY = 47e6                       # Formlabs Grey V4.1 post-cured, for the strength check

GRID = dict(
    lam=np.arange(3.0, 9.01, 0.5),            # design tip-speed ratio
    alpha=np.arange(2.0, 10.01, 1.0),         # design angle of attack [deg]
    c_root=np.arange(10.0, 70.01, 5.0) / 1e3,  # [m]; 70 mm = Blade 7 root (known to fit the hub)
    c_tip=np.arange(6.0, 30.01, 3.0) / 1e3,    # [m]
)
TSR_GRID = np.arange(1.0, 14.01, 0.25)        # operating tip-speed ratios for P(Omega)
BATCH = 512


@L.pytree
class GridPolar:
    """lab5_blade7.Polar on its uniform 0.1 deg alpha grid: index directly instead of searching."""
    log_re: jax.Array
    a0: jax.Array
    da: jax.Array
    cl: jax.Array
    cd: jax.Array

    @classmethod
    def of(cls, p):
        return cls(p.log_re, p.alpha[0], p.alpha[1] - p.alpha[0], p.cl, p.cd)

    def __call__(self, alpha, Re):
        @jnp.vectorize
        def one(a, re):
            a = (a + jnp.pi) % (2 * jnp.pi) - jnp.pi
            t = (a - self.a0) / self.da
            i = jnp.clip(jnp.floor(t).astype(int), 0, self.cl.shape[1] - 2)
            w = t - i
            row = lambda tab: tab[:, i] * (1 - w) + tab[:, i + 1] * w
            lookup = lambda tab: jnp.interp(jnp.log(re), self.log_re, row(tab))
            return lookup(self.cl), lookup(self.cd)

        return one(alpha, Re)


R_ST = jnp.linspace(R_HUB, R_TIP, N_ST)
W_TRAP = L.trapezoid(R_ST)
ROTOR = L.Rotor(R=R_TIP, r_hub=R_HUB)
XY, _ = L.naca4412()
XY_QC = jnp.asarray(XY - np.array([0.25, 0.0]))  # sections stacked on the quarter chord
_q = np.roll(XY, -1, axis=0)
A_UNIT = 0.5 * np.sum(XY[:, 0] * _q[:, 1] - _q[:, 0] * XY[:, 1])  # unit-chord area
P_UNIT = np.sum(np.hypot(*(_q - XY).T))                             # unit-chord perimeter
OM_GRID = jnp.asarray(TSR_GRID) * L.U_INF / R_TIP


def geometry(lam, alpha_deg, c_root, c_tip):
    theta = (2 / 3) * jnp.arctan(R_TIP / (lam * R_ST)) - jnp.deg2rad(alpha_deg)
    c = c_root + (c_tip - c_root) * (R_ST - R_HUB) / (R_TIP - R_HUB)
    return c, theta


def evaluate(d, polar):
    """One blade: power curve over TSR_GRID (BEM, Re iterated), geometry and material."""
    c, theta = geometry(*d)
    st = L.Stations(R_ST, c, theta)
    Re = L.RHO * jnp.hypot(L.U_INF, OM_GRID[:, None] * R_ST) * c / L.MU
    f = jax.vmap(lambda om, re: L._loads(st, ROTOR, polar, om, re, L.bem))
    for _ in range(3):
        out = f(OM_GRID, Re)
        Re = L.RHO * out["W"] * c / L.MU
    Q = L.N_B * (out["p_Q"] * R_ST) @ W_TRAP
    P = Q * OM_GRID
    k = jnp.argmax(P)
    # axial (rotor-axis) span of all sections about the quarter-chord stacking line
    y = c[:, None] * (jnp.sin(theta)[:, None] * XY_QC[:, 0] + jnp.cos(theta)[:, None] * XY_QC[:, 1])
    vol = W_TRAP @ (A_UNIT * c**2)
    area = W_TRAP @ (P_UNIT * c) + A_UNIT * c[-1] ** 2       # sides + tip cap
    return dict(P_max=P[k], Q_opt=Q[k], tsr_opt=jnp.asarray(TSR_GRID)[k], ext=y.max() - y.min(),
                vol=vol, area=area, c_min=c.min())


@jax.jit
def sweep(designs, polar):
    """All blades in one compiled call; lax.map over chunks bounds memory."""
    n = designs.shape[0]
    pad = -n % BATCH
    d = jnp.concatenate([designs, jnp.repeat(designs[-1:], pad, 0)]).reshape(-1, BATCH, 4)
    out = jax.lax.map(jax.vmap(lambda x: evaluate(x, polar)), d)
    return jax.tree_util.tree_map(lambda x: x.reshape(-1)[:n], out)


def main():
    polar = GridPolar.of(L.load_polar())
    designs = np.array([d for d in itertools.product(*GRID.values()) if d[3] <= d[2]])
    t0 = time.time()
    out = jax.tree_util.tree_map(np.asarray, sweep(jnp.asarray(designs), polar))
    print(f"{len(designs)} designs, BEM sweep incl. compile: {time.time() - t0:.1f} s")
    P_max, Q_opt, tsr_opt, ext, vol, area = (out[k] for k in
                                             ("P_max", "Q_opt", "tsr_opt", "ext", "vol", "area"))
    feasible = (ext <= AXIAL_LIMIT - AXIAL_MARGIN) & (0.12 * out["c_min"] >= MIN_FEATURE)
    print(f"{feasible.sum()} geometrically feasible")

    P_best = np.where(feasible, P_max, -np.inf).max()
    ok = feasible & (P_max >= POWER_FRACTION * P_best)
    i_best = int(np.argmax(np.where(feasible, P_max, -np.inf)))
    i_pick = int(np.flatnonzero(ok)[np.argmin(vol[ok])])

    np.savetxt(L.HERE / "blade_sweep_results.csv",
               np.column_stack([designs * [1, 1, 1e3, 1e3], ext / IN, vol * 1e6,
                                area * 1e4, P_max, Q_opt, tsr_opt, feasible]),
               delimiter=",", fmt="%.6g",
               header="lambda_d,alpha_d_deg,c_root_mm,c_tip_mm,axial_extent_in,volume_cm3,"
                      "area_cm2,P_max_W,Q_at_Pmax_Nm,tsr_at_Pmax,feasible", comments="")

    A_disk = np.pi * R_TIP**2
    P_wind = 0.5 * L.RHO * L.U_INF**3 * A_disk
    print(f"\n{'':22s}{'max power':>12s}{'picked':>12s}")
    rows = [("lambda_d", lambda i: designs[i][0], "{:.1f}"),
            ("alpha_d [deg]", lambda i: designs[i][1], "{:.0f}"),
            ("c_root [mm]", lambda i: designs[i][2] * 1e3, "{:.0f}"),
            ("c_tip [mm]", lambda i: designs[i][3] * 1e3, "{:.0f}"),
            ("P_max [W]", lambda i: P_max[i], "{:.1f}"),
            ("C_P", lambda i: P_max[i] / P_wind, "{:.3f}"),
            ("Q at P_max [N m]", lambda i: Q_opt[i], "{:.3f}"),
            ("TSR at P_max", lambda i: tsr_opt[i], "{:.2f}"),
            ("RPM at P_max", lambda i: tsr_opt[i] * L.U_INF / R_TIP / L.RPM, "{:.0f}"),
            ("volume/blade [cm3]", lambda i: vol[i] * 1e6, "{:.2f}"),
            ("area/blade [cm2]", lambda i: area[i] * 1e4, "{:.1f}"),
            ("axial extent [in]", lambda i: ext[i] / IN, "{:.2f}")]
    for name, get, fmt in rows:
        print(f"{name:22s}{fmt.format(get(i_best)):>12s}{fmt.format(get(i_pick)):>12s}")

    # strength / stiffness at freewheel (worst case: generator disconnected), with induction
    c, theta = (np.asarray(x) for x in geometry(*designs[i_pick]))
    r = np.asarray(R_ST)
    st = L.Stations(R_ST, jnp.asarray(c), jnp.asarray(theta))
    A, J, pts = L.section(jnp.broadcast_to(jnp.asarray(XY), (N_ST, *XY.shape)), st.c, st.theta)
    res = L.analyze(st, L.Sections(A, J, pts), ROTOR, L.load_polar(), L.bem)
    print(f"\nPicked blade at freewheel (BEM): {res['Om'] / L.RPM:.0f} RPM, "
          f"sigma_t = {res['sigma_t'] / 1e6:.1f} MPa (FoS {UTS_GREY / res['sigma_t']:.2f} on "
          f"Grey V4.1 UTS), tip deflection {res['delta_tip'] * 1e3:.1f} mm")

    # Fusion parameter table, same layout as blade_parameters.csv (lengths in mm, z from root)
    lines = ["Name,Unit,Expression,Value,Comments,Favorite"]
    for j in range(N_ST):
        n = f"{j + 1:02d}"
        cm, qm, pd, zm = c[j] * 1e3, c[j] * 1e3 / 4, np.rad2deg(theta[j]), (r[j] - R_HUB) * 1e3
        lines += [f"c{n},,{cm:.4f},{cm:.4f},Station {j + 1} chord,0",
                  f"q{n},mm,{qm:.4f},{qm:.4f},Station {j + 1} quarter chord offset,0",
                  f"p{n},deg,{pd:.5f},{pd:.5f},Station {j + 1} pitch angle,0",
                  f"z{n},mm,{zm:.4f},{zm:.4f},Station {j + 1} spanwise position,0"]
    with open(L.HERE / "blade_parameters_optimized.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote blade_parameters_optimized.csv and blade_sweep_results.csv")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(vol[feasible] * 1e6, P_max[feasible], s=4, alpha=0.4, label="feasible")
    ax.scatter(vol[~feasible] * 1e6, P_max[~feasible], s=4, alpha=0.2, c="0.6",
               label="violates extent/feature")
    ax.axhline(POWER_FRACTION * P_best, color="k", lw=0.8, ls="--")
    ax.plot(vol[i_pick] * 1e6, P_max[i_pick], "r*", ms=14, label="picked")
    ax.set(xlabel="volume per blade [cm$^3$]", ylabel="max shaft power [W]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(L.HERE / "blade_sweep_pareto.png", dpi=200)


if __name__ == "__main__":
    main()
