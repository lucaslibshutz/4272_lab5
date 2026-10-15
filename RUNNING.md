# RUNNING

## Setup
```bash
uv sync                       # creates .venv with Python 3.12 (pinned in .python-version)
uv sync --extra neuralfoil    # only needed to generate NeuralFoil polar tables
```
There's no manual venv activation: prefix commands with `uv run`.

## Entry points
| what | command |
|---|---|
| Blade 7, handout model, freewheeling speed (U = 15 m/s) | `uv run bem-blade run examples/blade7.yaml` |
| single RPM | `uv run bem-blade run examples/blade7.yaml --set operating.rpm=3600` |
| RPM sweep | `uv run bem-blade run examples/blade7.yaml --set "operating.rpm=[1000,2000,3600]"` |
| momentum (BEM) model instead | `uv run bem-blade run examples/blade7.yaml --set solver.model=bem` |
| handout method, standalone (write-up §4 table, Ω_ss, blend sensitivity) | `uv run python scripts/handout_method.py` |
| any config | `uv run bem-blade run <config.yaml> [-o outdir] [--set k.path=v ...] [--no-plots] [-q]` |
| re-download Blade 7 polars (airfoiltools, XFOIL Ncrit 9) | `uv run python scripts/fetch_airfoiltools.py` |
| NeuralFoil polars (alternative) | `uv run bem-blade polar 4412 --re 5e3 1e4 ... -o data/naca4412_neuralfoil.csv` |
| tests | `uv run pytest` (~8 s) |
| standalone submission script (every number in the write-up, no package install) | `cd submission && uv run lab5_blade7.py` |

Outputs are written to `output.dir` (default `results/<name>/`), which is git-ignored.

## Gotchas
- **YAML numbers.** `40.0e9` is a float here: a YAML 1.2 float resolver is installed, whereas
  plain PyYAML would read it as a string. NACA codes must be quoted (`"0012"`), or YAML parses
  them as numbers (or octal).
- **Operating point.** Give exactly one of `operating.Omega`, `rpm` or `tsr`. A scalar
  broadcasts against a list, so `U_inf: 15` with `rpm: [...]` is an RPM sweep.
  `rpm: freewheel` solves Q(Ω) = 0 instead. It scans `solver.freewheel_rpm` = [min, max, step]
  (default 1000–60000 by 250) for the last + to − sign change, then bisects to 1 RPM. If there's
  no sign change, the run fails with an error saying so; widen the range.
- **Model.** `solver.model` defaults to `bem`, so configs without it behave as before. Blade 7
  sets `blade_element` (the lab handout's model, derived in `../lab5_writeup`).
- **Relative paths.** `polar_file` and `stations.file` resolve relative to the config file,
  not the CWD.
- **Startup time.** The first call JIT-compiles, so expect about 2–3 s. Re-runs inside one
  Python process reuse the compiled function.
- **Hub and tip stations.** In `bem` mode, stations at exactly `r_hub` or `R` show
  `bem_region = -1` and zero load (Prandtl F = 0); that is expected. In `blade_element` mode
  they carry load, as in the handout.
- **Stale outputs.** A run overwrites files in `output.dir` but doesn't delete old ones. After
  switching from a sweep to a single point, old `stations_rpm*.csv` files remain until you clear
  the directory.
- **Polar coverage.** The airfoiltools table spans only about −9° to +14°…19°. Outside it, the
  loader blends to a flat plate (`load_polar_table(blend_deg=10)`), and the run flags those points
  as `table-extrapolated`.
- **Units in a new spreadsheet.** If a new geometry sheet uses inches or a different pitch sign,
  convert it first. The reader assumes m / deg and the PLAN.md §1 convention (LE upstream = +).
- **Submission script.** `submission/lab5_blade7.py` is a single-file re-implementation of the
  package (report frame and equations, PEP 723 header so `uv run` installs jax/numpy/matplotlib
  itself). It needs `naca4412_airfoiltools.csv` next to it; Blade 7 geometry is embedded. Its
  right-rule Omega_ss is 18,324 RPM vs 18,323 from `scripts/handout_method.py` (bisection to 1 RPM
  vs brentq).
