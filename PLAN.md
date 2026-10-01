# Blade Root Bending & Stress from BEM — Theory and Implementation Spec

Handoff spec for implementing a steady-state blade-element-momentum (BEM) solver that takes a NACA airfoil plus spanwise chord and section-angle arrays, computes aerodynamic loads, integrates them into spanwise bending moments, and returns the maximum bending moment and maximum normal stress for given material properties.

All theory needed to implement is in here. Sign conventions are fixed in §1 and used consistently throughout; do not mix in conventions from other references without converting.

---

## 0. Scope and assumptions

In scope:
- Steady, uniform, axial inflow (no yaw, tilt, cone, shear, tower shadow).
- One operating point (or a list of wind speeds for an envelope sweep, §8).
- Rigid-blade aerodynamics (loads computed on the undeflected blade; no aeroelastic coupling).
- Spanwise normal stress from flapwise + edgewise bending, optionally plus centrifugal axial stress and gravity.
- Static strength check against allowable stress.

Out of scope (report as limitations in output, do not silently ignore):
- Extreme/parked load cases (IEC 50-year gust), dynamic and fatigue loads.
- Shell buckling, shear stress, torsion from pitching moment.
- 3D/rotational stall augmentation, dynamic stall.

---

## 1. Coordinate system and sign conventions

Rotor frame at a blade station, right-handed triad **(t̂, n̂, ẑ)**:
- **ẑ**: spanwise, hub → tip (points toward a viewer looking at the section from the tip).
- **n̂**: axial, downwind (direction of the wind).
- **t̂**: in the rotor plane, in the direction of blade motion.

Check: t̂ × n̂ = ẑ, n̂ × ẑ = t̂, ẑ × t̂ = n̂.

Section angle **θ(r)** (input, = local twist + collective pitch): angle from the rotor plane to the chord line, positive with the trailing edge rotated downwind (leading edge rotated upwind). Larger θ → lower angle of attack.

Unit vectors fixed to the section:
- Chord direction, leading edge → trailing edge: **ĉ = −cos θ · t̂ + sin θ · n̂**
- Suction-side normal (the NACA +y side for positive camber): **ŝ = sin θ · t̂ + cos θ · n̂**

(ĉ·ŝ = 0. At θ = 0 the suction side faces downwind and the leading edge points in the direction of motion. This is the normal operating orientation: the wind strikes the pressure side.)

---

## 2. Inputs

Suggested config (YAML or JSON):

```yaml
airfoil:
  naca: "2412"            # 4-digit NACA designation, used at all stations
  polar_file: null        # optional: tabulated polars, see §4.1 (strongly preferred over fallback)
  alpha_stall_deg: 12.0   # fallback model only
  cd0: null               # fallback model only; null = flat-plate estimate from Re

rotor:
  B: 3                    # number of blades
  R: 40.0                 # tip radius [m]
  r_hub: 2.0              # hub radius [m]
  pitch_deg: 0.0          # collective pitch added to theta_deg [deg]

stations:                 # same length, r strictly increasing, r_hub <= r[0], r[-1] <= R
  r: [...]                # [m]
  chord: [...]            # [m]
  theta_deg: [...]        # twist + local pitch [deg], convention §1

operating:
  U_inf: 10.0             # [m/s], or a list for an envelope sweep (§8)
  Omega: 1.75             # [rad/s]; alternatively give tsr and compute Omega = tsr*U_inf/R

air:
  rho: 1.225              # [kg/m^3]
  mu: 1.81e-5             # [Pa s]

structure:
  model: "shell"          # "solid" | "shell" | "user"
  skin_thickness: 0.02    # shell only, absolute [m]; alternatively skin_thickness_frac (fraction of local chord), see §6.2
  user_section_props: null  # "user" model: per-station A, S_tt, S_nn, S_tn, centroid, contour points
  material:
    E: 40.0e9             # [Pa]
    sigma_allow_tension: 200.0e6      # [Pa]
    sigma_allow_compression: 150.0e6  # [Pa]
    density: 1900.0       # [kg/m^3], optional; enables gravity and centrifugal terms
  include_gravity: true
  include_centrifugal: true
```

Internally convert all angles to radians.

---

## 3. NACA 4-digit geometry

Designation **MPXX** → m = M/100 (max camber), p = P/10 (location of max camber), t = XX/100 (max thickness), all as fractions of chord.

Use cosine spacing for x ∈ [0, 1]: x_i = (1 − cos β_i)/2 with β_i uniform on [0, π]. 100–200 points per surface.

Thickness (use the closed-trailing-edge coefficient −0.1036 so the polygon closes):

    y_t = 5t [0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − 0.1036x⁴]

Camber line (if m = 0 or p = 0, the airfoil is symmetric: y_c = 0, dy_c/dx = 0):

    x < p:   y_c = (m/p²)(2px − x²),                 dy_c/dx = (2m/p²)(p − x)
    x ≥ p:   y_c = (m/(1−p)²)((1 − 2p) + 2px − x²),  dy_c/dx = (2m/(1−p)²)(p − x)

Surfaces, with ψ = atan(dy_c/dx):

    upper: x_u = x − y_t sin ψ,  y_u = y_c + y_t cos ψ
    lower: x_l = x + y_t sin ψ,  y_l = y_c − y_t cos ψ

Assemble one closed, counter-clockwise contour (TE → upper surface → LE → lower surface → TE, de-duplicating the shared LE/TE points) in normalized coordinates, then scale by chord c(r) at each station.

**Zero-lift angle (thin airfoil theory).** With x = (1 − cos θ₀)/2,

    α_L0 = −(1/π) ∫₀^π (dy_c/dx)(cos θ₀ − 1) dθ₀

Integrate numerically (e.g. 2000-point trapezoid or `scipy.integrate.quad` split at the θ₀ corresponding to x = p, where dy_c/dx has a kink).

---

## 4. Section aerodynamics: C_l(α, Re), C_d(α, Re)

Definitions (per unit span, relative velocity W):

    L' = ½ρW²c C_l   (perpendicular to W)
    D' = ½ρW²c C_d   (parallel to W)

### 4.1 Preferred: tabulated polars
If `polar_file` is given: columns `Re, alpha_deg, Cl, Cd`. Interpolate linearly in α, linearly in log(Re). The table must cover α ∈ [−180°, 180°] (or at least the range BEM visits, roughly [−10°, 90°]); if it does not, extend with the post-stall model in §4.3 and log a warning.

Optional generator if available in the environment (verify the API against the installed package's docs before relying on it): NeuralFoil (`pip install neuralfoil`) produces C_l, C_d for arbitrary coordinates, α, and Re; XFOIL can be driven via subprocess. Either may be used to build the polar table up front at the Re range of the rotor. Do not make them hard dependencies.

### 4.2 Fallback: attached flow (|α − α_L0| small)

    C_l,lin = 2π (α − α_L0)

Drag (viscous only; inviscid 2D drag is zero). If `cd0` is null, estimate from a fully turbulent flat plate wetted on both sides:

    C_d0 = 2 · 0.074 / Re^(1/5),   Re = ρWc/μ

and add a mild quadratic rise: C_d,att = C_d0 + 0.01·(α − α_L0)² (α in rad). This is a rough model; label results computed with it as "fallback aero".

### 4.3 Fallback: stall and post-stall blend
BEM's root search visits large α, so the model must be defined for all α. Blend to a 2D flat plate with a smooth weight:

    f(α) = 1 / (1 + exp(−(|α| − α_stall)/Δ)),   Δ = 2° (in rad)
    C_l = (1 − f)·C_l,lin + f·sin(2α)
    C_d = (1 − f)·C_d,att + f·(C_d0 + 2 sin²α)

The flat-plate terms come from a normal-force coefficient C_N ≈ 2 resolved into lift and drag. Flag any converged station with |α| > α_stall in the output.

---

## 5. BEM solve (per station)

Notation at radius r: local solidity σ' = Bc/(2πr), local speed ratio λ_r = Ωr/U∞, inflow angle φ, angle of attack α = φ − θ (with θ including collective pitch).

Force coefficients in rotor axes:

    c_n = C_l cos φ + C_d sin φ     (along n̂, thrust)
    c_t = C_l sin φ − C_d cos φ     (along t̂, torque)

Prandtl tip and hub loss (use |sin φ|):

    F_tip = (2/π) arccos( exp( −B(R − r) / (2 r |sin φ|) ) )
    F_hub = (2/π) arccos( exp( −B(r − r_hub) / (2 r_hub |sin φ|) ) )
    F = F_tip · F_hub

Induction factors from φ (momentum balance on the annulus):

    k  = σ' c_n / (4 F sin²φ)
    k' = σ' c_t / (4 F sin φ cos φ)

    if k ≤ 2/3:  a = k / (1 + k)                          # momentum region, a ≤ 0.4
    else:                                                  # Buhl (2005) high-induction correction
        g1 = 2Fk − (10/9 − F)
        g2 = 2Fk − F(4/3 − F)
        g3 = 2Fk − (25/9 − 2F)
        a  = 1 − 1/(2√g2)              if |g3| < 1e-6
        a  = (g1 − √g2) / g3           otherwise

    a' = k' / (1 − k')

**Solve a single residual in φ** (Ning 2014, guaranteed convergence; this is the formulation used by CCBlade):

    Res(φ) = sin φ / (1 − a) − cos φ (1 − k') / λ_r

Root-find with a bracketing method (Brent, `scipy.optimize.brentq`):
1. Try φ ∈ (ε, π/2], ε = 1e-6. If the signs at the ends differ, solve there.
2. Otherwise try (−π/4, −ε), then (π/2, π − ε). Log which region was used. Anything other than region 1 means an unusual operating state; flag it.

**Do not** use plain fixed-point iteration on (a, a'); it fails to converge near a ≈ 0.4 and at high loading.

**Reynolds dependence.** C_l and C_d depend on Re = ρWc/μ, and W depends on the solution. Outer loop: start with W₀ = √(U∞² + (Ωr)²), solve, recompute W from the converged a and a', and re-solve until Re changes by < 1% (2–3 passes).

**Post-solve quantities:**

    W = √( (U∞(1 − a))² + (Ωr(1 + a'))² )
    q = ½ρW²
    p_N(r) = q c c_n      # force per unit span along n̂ [N/m], per blade
    p_T(r) = q c c_t      # force per unit span along t̂ [N/m], per blade

At the tip F → 0, so p_N and p_T → 0. If r[-1] < R, append a node at r = R with p_N = p_T = 0 for load integration only.

**Rotor totals** (trapezoid over r, including the tip node):

    T = B ∫ p_N dr,   Q = B ∫ p_T r dr,   P = QΩ
    C_T = T / (½ρU∞²πR²),   C_P = P / (½ρU∞³πR²)

Sanity: C_P < 16/27; C_T typically 0.6–0.9 near rated conditions.

---

## 6. Section structural properties

All second moments are about the section centroid, expressed in the rotor in-plane coordinates (x_t along t̂, x_n along n̂):

    A = ∫ dA,  S_tt = ∫ x_t² dA,  S_nn = ∫ x_n² dA,  S_tn = ∫ x_t x_n dA

Workflow per station:
1. Scale the normalized NACA contour by c(r): (x', y') in meters, x' along the chord LE→TE, y' toward the suction side.
2. Compute section properties in (x', y'), then shift to the centroid (parallel axis theorem).
3. Map centroid-relative contour points to rotor coordinates using §1:

       x_t = −x' cos θ + y' sin θ
       x_n =  x' sin θ + y' cos θ

   Either rotate the second moments with the same transformation or simply recompute them from the mapped points. Recomputing is simpler and less error-prone.

### 6.1 Solid section (`model: solid`)
Polygon (vertices (x_i, y_i), counter-clockwise, closed so i+1 wraps), with c_i = x_i y_{i+1} − x_{i+1} y_i:

    A     = ½ Σ c_i
    x̄     = (1/(6A)) Σ (x_i + x_{i+1}) c_i
    ȳ     = (1/(6A)) Σ (y_i + y_{i+1}) c_i
    ∫y²   = (1/12) Σ c_i (y_i² + y_i y_{i+1} + y_{i+1}²)
    ∫x²   = (1/12) Σ c_i (x_i² + x_i x_{i+1} + x_{i+1}²)
    ∫xy   = (1/24) Σ c_i (x_i y_{i+1} + 2x_i y_i + 2x_{i+1} y_{i+1} + x_{i+1} y_i)

These are about the origin; shift to the centroid with the parallel-axis theorem (subtract A·ȳ², A·x̄², A·x̄ȳ respectively).

### 6.2 Thin-walled shell (`model: shell`)
Treat the airfoil contour as the wall midline with thickness t_s (absolute `skin_thickness`, or `skin_thickness_frac`·c). Valid for t_s ≪ c; warn if t_s > 0.05c. For each straight segment k from (x₁, y₁) to (x₂, y₂), length L:

    A_k    = t_s L
    ∫y² dA = t_s L (y₁² + y₁y₂ + y₂²) / 3
    ∫x² dA = t_s L (x₁² + x₁x₂ + x₂²) / 3
    ∫xy dA = t_s L (2x₁y₁ + x₁y₂ + x₂y₁ + 2x₂y₂) / 6

Sum over segments, compute the centroid from first moments (∫x dA = t_s L (x₁ + x₂)/2, likewise y), then shift. Real blades carry most bending in spar caps; the uniform shell is a first-order model, and the output should say so.

### 6.3 User-supplied (`model: user`)
Accept per-station A, S_tt, S_nn, S_tn (or chord-frame values plus θ to rotate), centroid offset, and the contour points at which to evaluate stress. Validate that the arrays align with `stations.r`.

---

## 7. Bending moments, stress, deflection

### 7.1 Moments (per blade, cantilever clamped at the root, free at the tip)
At a station r₀, the moment of the outboard load is M(r₀) = ∫_{r₀}^{R} (r − r₀) ẑ × (p_T t̂ + p_N n̂) dr. Using ẑ × t̂ = n̂ and ẑ × n̂ = −t̂:

    M_flap(r₀) = ∫_{r₀}^{R} p_N (r − r₀) dr      # out-of-plane, ≥ 0 in normal operation
    M_edge(r₀) = ∫_{r₀}^{R} p_T (r − r₀) dr      # in-plane, aerodynamic
    M_t = −M_flap,   M_n = M_edge                # components along t̂ and n̂

Implement as two cumulative integrations from the tip (more accurate than a double loop):

    V_N(r) = ∫_r^R p_N dr',      M_flap(r) = ∫_r^R V_N dr'
    (same for T)

using cumulative trapezoid on a reversed array. Report V (shear) too.

**Gravity (optional, needs density and section area):** mass per length m'(r) = ρ_mat A(r). The worst static gravity case is the blade horizontal, with gravity along ±t̂. Add M_grav(r₀) = ∫ m' g (r − r₀) dr to M_edge with **both signs** as two load cases, and report the governing one.

**Centrifugal (optional):** N(r₀) = ∫_{r₀}^{R} m' Ω² r dr, tensile, giving a uniform σ_N = N/A.

The root moment is at r[0]. If r[0] > r_hub, the true root moment is not computed; warn.

### 7.2 Normal stress (unsymmetric bending)
For a centroid-relative point (x_t, x_n) on the section, with D = S_tt S_nn − S_tn²:

    σ_z = [ (M_t S_tt + M_n S_tn) x_n − (M_t S_tn + M_n S_nn) x_t ] / D  +  N/A

Derivation: assume σ = a·x_t + b·x_n (plane sections). Equilibrium gives M_t = ∫σ x_n dA and M_n = −∫σ x_t dA. Solve the 2×2 system. Do not assume principal axes; airfoil sections with camber and twist have S_tn ≠ 0.

Sign check: for pure flap load (p_N > 0) on a symmetric section aligned with the axes, σ = M_t x_n / S_nn < 0 on the downwind side (x_n > 0). That is compression downwind and tension upwind, which is correct for a cantilever bent downwind.

Evaluate σ_z at every contour point at every station. Report:
- max tensile σ, max compressive σ, and the station and chordwise location (x/c, upper or lower surface) of each;
- safety factors SF_t = σ_allow,t / σ_max,t and SF_c = σ_allow,c / |σ_min|.

### 7.3 Tip deflection (optional output)
Curvatures from the same linear stress field (ε = σ/E = −δ_t'' x_t − δ_n'' x_n):

    δ_n'' = −(M_t S_tt + M_n S_tn) / (E D)      # out-of-plane, + downwind
    δ_t'' =  (M_t S_tn + M_n S_nn) / (E D)      # in-plane

Integrate twice from the root with δ = δ' = 0 at r[0] (cumulative trapezoid). Report tip δ_n. A large deflection relative to R means the rigid-blade assumption is questionable; warn if δ_n,tip > 0.1R.

---

## 8. Envelope sweep (optional)
If `U_inf` is a list (and Ω is given per speed, or TSR is given), run §5–§7 for each speed and report, per station, the maximum |M_flap|, |M_edge|, and the extreme σ across speeds, plus which speed governs. Flap root moment usually peaks near rated wind speed for a pitch-regulated rotor. With fixed θ and no control model, expect a monotonic rise; state this limitation.

---

## 9. Outputs

1. `stations.csv`, one row per station: r, c, θ, φ, α, a, a', F, W, Re, C_l, C_d, c_n, c_t, p_N, p_T, V_N, V_T, M_flap, M_edge, A, S_tt, S_nn, S_tn, σ_max_t, σ_min_c, location of extremes, stall flag, aero-model flag.
2. `summary.json`: T, Q, P, C_T, C_P; max M_flap, M_edge, and resultant √(M_flap² + M_edge²) with stations; global max tensile and compressive σ with station and location; safety factors; tip deflection; list of warnings and limitations.
3. Plots (PNG): p_N and p_T vs r; M_flap and M_edge vs r; σ extremes vs r with allowable lines; a and α vs r.
4. A short console summary.

---

## 10. Suggested module layout

    naca.py          # §3 geometry + α_L0
    aero.py          # §4 polar interface: PolarTable, FallbackPolar (same .cl_cd(alpha, Re) API)
    bem.py           # §5 per-station solver + rotor totals
    section.py       # §6 solid / shell / user section properties, frame mapping
    structure.py     # §7 moment integration, stress, deflection
    run.py           # config load, orchestration, outputs
    tests/

Python + NumPy + SciPy + Matplotlib. Keep functions pure and unit-testable; no global state.

---

## 11. Required tests (write these before trusting outputs)

Geometry
- NACA 0012: max thickness 0.12c at x ≈ 0.30c; solid area ≈ 0.0822 c² (= 0.685·t·c).
- Symmetric airfoil gives α_L0 = 0; NACA 2412 thin-airfoil α_L0 ≈ −2.08°.

Section properties
- Solid rectangle b×h: S = bh³/12 about the centroid, product moment 0.
- Thin circular tube (polygon of 400 segments, radius ρ, wall t): S = πρ³t within 0.1%.
- Rotating a section and its moment vector by the same angle leaves max |σ| unchanged.

Stress
- Symmetric section, S_tn = 0, reduces to σ = M_t x_n/S_nn − M_n x_t/S_tt.
- Pure M_t on a rectangle: resultant ∫σ x_n dA reproduces M_t, ∫σ x_t dA = 0 (re-integrate numerically).

Moments
- Uniform p over [0, L]: root moment pL²/2, root shear pL.
- Linearly increasing p (0 at root to p₀ at tip): root moment p₀L²/3.

BEM
- σ' → 0 (tiny chord): a, a' → 0 and φ → atan(1/λ_r) at the geometric inflow.
- Converged state satisfies both the blade-element and the annular-momentum thrust:
  B·p_N = 4πrρU∞² a(1 − a) F  (momentum region, a ≤ 0.4) to within 1e-6 relative.
- C_P < 16/27 on any input.
- If `ccblade` is installable, cross-check one case (same polars, tip/hub loss on) to within 1%.

Units: run one case in SI and the same case with lengths scaled by 2 and Ω halved (same λ). C_T and C_P should be unchanged; moments scale by 2³ (with q, c, r: q c r² → ×1·2·4).

---

## 12. Reference equations at a glance

    Velocity triangle:  tan φ = U∞(1 − a) / (Ωr(1 + a')),  α = φ − θ
    Section loads:      L' = ½ρW²c C_l ⟂ W,   D' = ½ρW²c C_d ∥ W
    Rotor axes:         p_N = q c (C_l cos φ + C_d sin φ),   p_T = q c (C_l sin φ − C_d cos φ)
    Momentum (annulus): dT = 4πrρU∞² a(1 − a) F dr,   dQ = 4πr³ρU∞Ω a'(1 − a) F dr
    Moments:            M_flap(r₀) = ∫ p_N (r − r₀) dr,   M_edge(r₀) = ∫ p_T (r − r₀) dr
    Stress:             σ = [(M_t S_tt + M_n S_tn) x_n − (M_t S_tn + M_n S_nn) x_t]/D + N/A

References: Hansen, *Aerodynamics of Wind Turbines* (BEM, Prandtl tip loss); Ning, "A simple solution method for the blade element momentum equations with guaranteed convergence," *Wind Energy* (2014); Buhl, NREL/TP-500-36834 (2005), high-induction correction; Abbott & von Doenhoff, *Theory of Wing Sections* (NACA geometry, thin airfoil theory).
