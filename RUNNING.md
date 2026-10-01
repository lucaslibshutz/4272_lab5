# RUNNING

## Setup
```bash
uv sync                       # creates .venv with Python 3.12 (pinned in .python-version)
uv sync --extra neuralfoil    # only needed to regenerate polar tables
```
There's no manual venv activation: prefix commands with `uv run`.

## Entry points
| what | command |
|---|---|
| Blade 7 RPM sweep (U = 15 m/s) | `uv run bem-blade run examples/blade7.yaml` |
| single RPM | `uv run bem-blade run examples/blade7.yaml --set operating.rpm=3600` |
| any config | `uv run bem-blade run <config.yaml> [-o outdir] [--set k.path=v ...] [--no-plots] [-q]` |
| regenerate polars | `uv run bem-blade polar 4412 --re 5e3 1e4 ... -o data/naca4412_neuralfoil.csv` |
| tests | `uv run pytest` (~5 s) |

Outputs are written to `output.dir` (default `results/<name>/`), which is git-ignored.

## Gotchas
- **YAML numbers.** `40.0e9` is a float here: a YAML 1.2 float resolver is installed, whereas
  plain PyYAML would read it as a string. NACA codes must be quoted (`"0012"`), or YAML parses
  them as numbers (or octal).
- **Operating point.** Give exactly one of `operating.Omega`, `rpm` or `tsr`. A scalar
  broadcasts against a list, so `U_inf: 15` with `rpm: [...]` is an RPM sweep.
- **Relative paths.** `polar_file` and `stations.file` resolve relative to the config file,
  not the CWD.
- **Startup time.** The first call JIT-compiles, so expect about 2–3 s. Re-runs inside one
  Python process reuse the compiled function.
- **Hub and tip stations.** Stations at exactly `r_hub` or `R` show `bem_region = -1` and zero
  load (Prandtl F = 0). This is expected.
- **Units in a new spreadsheet.** If a new geometry sheet uses inches or a different pitch sign,
  convert it first. The reader assumes m / deg and the PLAN.md §1 convention (LE upstream = +).
