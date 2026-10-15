# bem-blade

Steady blade-element-momentum (BEM) analysis of a wind-turbine blade that goes all the way to
**spanwise bending moments, maximum normal stress and safety factors**. It takes a NACA
4-digit airfoil, spanwise chord/twist arrays and material properties. The theory and
sign conventions are in [`PLAN.md`](PLAN.md); section numbers (§) below refer to it.

Two aerodynamic models are available through `solver.model`:

- **`bem`** (the default): blade element momentum theory with axial/tangential induction, Prandtl tip/hub
  loss and the Buhl correction (PLAN.md §5).
- **`blade_element`**: the lab handout's model, which Blade 7 uses. It has no induction and no
  tip/hub loss: each section sees U_eff = (Ωr, U∞), φ = atan2(U∞, Ωr), and every station carries
  load. The rotor torque comes from the trapezoid rule over the given stations. The theory is
  derived in the lab write-up, [`../lab5_writeup`](../lab5_writeup) (§2–§5).

Either model can solve for the **freewheeling speed** Ω_ss, the root of Q(Ω) = 0, with
`operating.rpm: freewheel`.

- **JAX throughout.** One jitted function runs the BEM solve, loads, moments, stress and
  deflection. It is `vmap`-ed over stations and over operating points, so a 21-point RPM sweep
  is a single compiled call per Reynolds pass. All of it runs in float64.
- **Config-driven.** One YAML file holds everything, and any value can be overridden from the
  command line with `--set key.path=value`.
- **Real-data ready.** Blade geometry can be read straight from a `.xlsx` or `.csv`. Polars can
  come from a table (XFOIL polars from airfoiltools.com, or generated with NeuralFoil), or from the
  fallback model in §4.2–4.3.

## Install

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync                      # core: jax, numpy, scipy, matplotlib, pyyaml, openpyxl
uv sync --extra neuralfoil   # optional: polar generation with NeuralFoil
```

## Quick start (Blade 7)

```bash
uv run bem-blade run examples/blade7.yaml
```

This analyses the 3-bladed, 7 in (0.1778 m) SLA-printed Blade 7 rotor:

- **Airfoil:** NACA 4412, with XFOIL polars from airfoiltools.com (Ncrit 9, Re 5×10⁴–10⁶), as in
  the lab handout.
- **Material:** Formlabs Grey Resin V4.1, post-cured.
- **Model:** blade-element theory as in the lab handout (`solver.model: blade_element`).
- **Operating point:** U∞ = 15 m/s at the freewheeling speed (`operating.rpm: freewheel`),
  which comes out at Ω_ss ≈ 17,574 RPM. Loads, moments and stresses are evaluated there.

Results go to `results/blade7/`.

The lab handout's method is also implemented independently of the package, in
`scripts/handout_method.py`. It prints the write-up §4 table: Q at 3600 RPM and Ω_ss for the
left, trapezoid, midpoint and right rules. It also prints the hub/tip state at Ω_ss and how
sensitive Ω_ss is to the post-stall blend width:

```bash
uv run python scripts/handout_method.py            # --blend DEG, --U, --rpm, --polar CSV
```

Other examples:

```bash
uv run bem-blade run examples/blade_40m.yaml   # generic 40 m blade, single point, fallback aero
uv run bem-blade run examples/sweep_40m.yaml   # same blade, U_inf sweep at constant TSR
```

### Common tweaks without editing files

```bash
# single rotor speed, or an RPM sweep, instead of the freewheeling speed
uv run bem-blade run examples/blade7.yaml --set operating.rpm=3600
uv run bem-blade run examples/blade7.yaml --set "operating.rpm=[1000,2000,3000,3600]"
# freewheeling speed at several tunnel speeds
uv run bem-blade run examples/blade7.yaml --set "operating.U_inf=[10,15]"
# momentum (BEM) model instead of the handout's blade-element model
uv run bem-blade run examples/blade7.yaml --set solver.model=bem
# different tunnel speed and a coarser sweep
uv run bem-blade run examples/blade7.yaml --set operating.U_inf=10 --set "operating.rpm=[1500,2500,3500]"
# constant TSR instead of RPM (exactly one of Omega / rpm / tsr may be set)
uv run bem-blade run examples/blade7.yaml --set operating.rpm=null --set operating.tsr=4.47
# collective pitch offset, fallback aero, other output dir, no plots
uv run bem-blade run examples/blade7.yaml --set rotor.pitch_deg=2 --set airfoil.polar_file=null -o results/test --no-plots
```

`--set` values are parsed as YAML, so `null`, lists `[a,b]` and `1e-5` all work.

### New blade geometry

Point `stations.file` at a spreadsheet or CSV. It needs a header row with `r`, `chord`, and
`pitch` (or `theta_deg`/`twist`). Reading stops at the first row that isn't numeric, so notes
below the table are ignored. Units are m, m and deg. The angle uses the §1 convention: positive
when the leading edge is rotated upwind. This matches the "pitch" definition in the Blade 7
sheet.

```yaml
stations:
  file: ../data/Blade8_geometry.xlsx
```

Remember to set `rotor.R` (tip radius) and `rotor.r_hub`.

### New polar table

The Blade 7 table comes from airfoiltools.com. To re-download it (raw CSVs are kept in
`data/airfoiltools/`):

```bash
uv run python scripts/fetch_airfoiltools.py   # -> data/naca4412_airfoiltools.csv
```

Or generate one with NeuralFoil (the earlier Blade 7 polar, still in `data/`):

```bash
uv run bem-blade polar 4412 --re 5e3 1e4 2e4 5e4 1e5 2e5 3e5 --alpha -20 25 0.5 -o data/naca4412_neuralfoil.csv
```

Any CSV with columns `Re, alpha_deg, Cl, Cd` works, e.g. XFOIL output or wind-tunnel data.
Interpolation is linear in α and linear in log Re. Outside the tabulated α range, the table is
extended with the §4.3 flat-plate model, and every station that uses the extension is flagged.

## Config reference

| key | meaning |
|---|---|
| `airfoil.naca` | 4-digit NACA code, **quoted** (`"0012"`) |
| `airfoil.polar_file` | polar CSV (path relative to the config file); `null` = fallback model §4.2–4.3 |
| `airfoil.alpha_stall_deg` | stall angle for the fallback blend and the stall flag |
| `airfoil.cd0` | fallback drag floor; `null` = flat-plate estimate 2·0.074/Re^0.2 |
| `rotor.B`, `rotor.R`, `rotor.r_hub` | blade count, tip radius [m], hub radius [m] |
| `rotor.pitch_deg` | collective pitch added to every station's θ |
| `stations.file` *or* `stations.r/chord/theta_deg` | blade geometry (strictly increasing r, r_hub ≤ r ≤ R) |
| `operating.U_inf` | wind speed [m/s], scalar or list |
| `operating.Omega` / `rpm` / `tsr` | exactly one; scalar or list. Scalars broadcast against lists |
| `air.rho`, `air.mu` | air density and dynamic viscosity |
| `structure.model` | `solid` (polygon §6.1), `shell` (thin wall §6.2), `user` (§6.3) |
| `structure.skin_thickness` / `skin_thickness_frac` | shell wall thickness [m] or fraction of chord |
| `structure.material.E` | Young's modulus [Pa] (used for deflection) |
| `structure.material.sigma_allow_tension/compression` | allowables [Pa] (positive numbers) |
| `structure.material.density` | [kg/m³]; enables gravity and centrifugal terms |
| `structure.include_gravity/centrifugal` | toggle the mass terms |
| `solver.re_tol`, `solver.re_max_passes` | Reynolds outer loop: stop when Re changes < tol |
| `output.dir`, `output.plots` | output directory and PNG toggle |

For `model: user`, give `structure.user_section_props` as lists with one entry per station:
`A`, `S_11`, `S_22`, `S_12`, `points_1`, `points_2` (equal-length point lists), and optionally
`centroid_1`, `centroid_2` and `frame: chord`. With `frame: rotor` (the default), 1 = t̂ and
2 = n̂. With `frame: chord`, 1 = x′ (LE→TE) and 2 = y′ (suction side), and the inputs are rotated
by θ.

Defaults for every key are in [`src/bem_blade/config.py`](src/bem_blade/config.py).

## Outputs

### Console report

The console report has four panels:

- **MAX MOMENTS:** max |M_flap|, |M_edge|, the resultant, root gravity moment and centrifugal
  force. Each has its station and, in a sweep, the operating point that governs it.
- **STRENGTH:** max tensile and compressive σ with location, safety factors (✔ ≥ 2, ▲ 1–2,
  ✖ < 1) and tip deflection.
- **OPERATING POINTS** (sweeps only): one row per point. ★ marks the best C_P, ▲ the max moment,
  ◆ the lowest SF, and ~ a point where more than half the stations are stalled.
- **Notes:** warnings grouped by type. The full per-point list is in `summary.json`.

Colors are dropped automatically when output is piped, or when `NO_COLOR` is set.

### Files

One stations CSV is written per operating point: `stations.csv`, or `stations_rpm3600.csv` etc.
in a sweep. It has r, c, θ, φ, α, a, a′, F, W, Re, Cl, Cd, cn, ct, p_N, p_T, V_N, V_T, M_flap,
M_edge, A, S_tt, S_nn, S_tn, max tensile/compressive σ with location (surface, x/c, gravity case),
stall flag, aero-model flag and BEM bracket region.

| file | contents |
|---|---|
| `envelope.csv` | (sweeps only) per-station max \|M_flap\|, \|M_edge\|, extreme σ, and the governing U/RPM |
| `summary.json` | T, Q, P, C_T, C_P, moment and stress maxima with locations, safety factors, tip deflection, warnings, limitations, and the resolved config |
| `loads.png`, `moments.png`, `stress.png`, `induction.png`, `sweep.png` | plots (§9) |

**BEM region flags** (`bem_region`):

| value | meaning |
|---|---|
| 1 | normal, φ ∈ (0, 90°] |
| 2 | propeller brake |
| 3 | φ > 90° |
| 0 | no bracket found; loads set to 0 and a warning is raised |
| −1 | station at r_hub or R, where F = 0, so no load by construction (`bem` only) |

In `blade_element` mode every station is region 1, with a = a′ = 0 and F = 1. When
`operating.rpm` is `freewheel`, `summary.json` also gets a `freewheel` list of
`{U_inf, Omega_ss, rpm_ss}`.

## Python API

```python
from bem_blade import load_config, run, write_outputs, print_summary, summary_text

cfg = load_config("examples/blade7.yaml", overrides=["operating.rpm=[3000,3600]"])
res = run(cfg)                       # res.out: dict of numpy arrays, leading axis = op point
print(res.out["M_flap"][:, 0])       # root flap moment per operating point [N m]
write_outputs(res, "results/custom")
print_summary(res)                   # the colored console report
log_text = summary_text(res)         # same report as plain text
```

`load_config` also accepts a plain dict. Lower-level pieces are pure functions and can be used
on their own:

| module | contents |
|---|---|
| `bem_blade.naca` | geometry, α_L0 |
| `bem_blade.aero` | `FallbackPolar`, `PolarTable` (JAX pytrees with `.cl_cd(alpha, Re)`) |
| `bem_blade.bem` | `solve_stations` (BEM), `blade_element_stations`, `trapezoid_weights`, `rotor_totals` |
| `bem_blade.model` | `run`, `rotor_torque(blade, U, Omega)`, `freewheel_omega(blade, U, solver)` |
| `bem_blade.section` | polygon/shell properties |
| `bem_blade.structure` | moments, stress, deflection |

## Implementation notes and deviations from PLAN.md

- **Root finder.** It uses fixed 80-step bisection on the §5 brackets instead of Brent. It is
  equally guaranteed, branch-free under `vmap`, and converges to float64 resolution. An
  independent `scipy.optimize.brentq` reference matches φ to 1e-10 (`tests/test_bem.py`).
- **Propeller-brake region (φ < 0).** This uses a = k/(k−1), i.e. residual
  sin φ (1−k) − cos φ (1−k′)/λ_r. That is the Ning (2014)/CCBlade branch; the spec's
  a = k/(1+k) is singular there. Region 2 is always flagged.
- **Hub and tip stations.** Stations at exactly r_hub or R have F = 0, so they carry zero load
  and are marked −1.
- **Mass beyond the last station.** Mass per length is held constant from r[-1] to R for the
  gravity and centrifugal terms.
- **Gravity.** It is applied along ±t̂ as two load cases. The governing case is reported, with
  the stress location tagged `g+t` or `g-t`.
- **NACA 0012 area test.** The closed-TE coefficient mandated in §3 gives 0.6809·t·c. The
  0.685·t·c in §11 belongs to the open-TE coefficient. The test checks both.
- **ccblade cross-check.** Not done: ccblade isn't on PyPI (it ships inside WISDEM). The brentq
  reference above stands in for it.

## Blade 7 data provenance and assumptions

- **Geometry:** `data/Blade7_geometry.xlsx`, as given for the lab: 20 stations from the hub
  joint (r = 25.4 mm) to the tip (r = 177.8 mm).
- **Material:** Formlabs Grey Resin V4.1 TDS (25011042-TDS-ENUS), post-cured 30 min at 60 °C.
  E = 2.201 GPa and UTS = 47 MPa (ASTM D638-14).
- **Not on the TDS (assumed):**
  - compressive allowable = 47 MPa (taken equal to UTS; conservative for SLA resins);
  - density = 1180 kg/m³ (typical cured SLA resin);
  - flexural strength (90 MPa) is *not* used, because it is a modulus of rupture.
- **Low-Re aerodynamics:** Re at 15 m/s is about 2×10⁴ to 1×10⁵. The airfoiltools XFOIL polars
  start at Re = 5×10⁴ (lower Re is clamped to that row) and only cover about −9° to +14°…19°,
  depending on Re. Outside that range each row is blended linearly over 10° into the flat-plate
  post-stall model (§4.3), and those stations are flagged as extrapolated. At low RPM much of the
  blade is stalled and lands in that blended region.

## Tests

```bash
uv run pytest
```

The suite covers every check in §11, plus a few more:

- **Geometry:** NACA thickness, area and α_L0.
- **Sections:** rectangle and thin-tube second moments; rotation invariance of max|σ| and of
  the area/trace/determinant invariants.
- **Stress:** the symmetric-section reduction; re-integrated resultants; the §7.2 sign check.
- **Moments and deflection:** uniform/linear load moments; cantilever tip deflection pL⁴/8EI.
- **BEM:** σ′ → 0 gives geometric inflow; momentum balance B·p_N = 4πrρU²a(1−a)F to 1e-6;
  C_P < 16/27 on random blades; the brentq cross-check; unit scaling (C_T, C_P invariant,
  moments ×8); sweep == individual runs.
- **Blade element** (`tests/test_blade_element.py`, the write-up's model):
  - a = a′ = 0, F = 1 and φ = atan2(U∞, Ωr), with every station loaded;
  - [p_T; p_N] = M(φ)[L′; D′], and the magnitude is preserved;
  - p_T > 0 ⇔ C_L/C_D > Ωr/U∞;
  - the trapezoid weights sum to R − r_hub;
  - Q(Ω_ss) ≈ 0, with a + to − crossing;
  - Ω_ss matches `scripts/handout_method.py` to within 2 RPM.
- **Config:** I/O tests.

## Limitations

These are reported in `summary.json` and are not silently ignored:

- Steady, uniform, axial inflow on a rigid blade.
- No extreme/parked, dynamic or fatigue load cases.
- No buckling, shear or torsion stress.
- No 3D/rotational stall augmentation or dynamic stall.
- The shell model is a uniform thin wall (real blades carry bending in spar caps).
- Sweeps use a fixed θ with no control model.
- `blade_element` has no induction, so it overpredicts α, the loads and C_P (C_P can exceed the
  Betz limit). The freewheeling speed neglects bearing and generator torque.
