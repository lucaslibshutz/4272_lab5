"""Colored console summary (rich). Degrades to plain text when piped or NO_COLOR is set.

Layout: header -> MAX MOMENTS -> STRENGTH -> per-operating-point table (sweeps) -> notes.
"""

from __future__ import annotations

import io
from collections import defaultdict

import numpy as np
from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .model import Result, op_label, rpm
from .report import point_summary

SF_OK, SF_WARN = 2.0, 1.0   # >= 2 OK, 1..2 low margin, < 1 fails the static check

PREFIXES = ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"))


def _prefix(x: float) -> tuple[float, str]:
    x = abs(x)
    for f, p in PREFIXES:
        if x >= f:
            return f, p
    return 1.0, ""


def eng(x: float, unit: str, digits: int = 3) -> str:
    f, p = _prefix(x)
    return f"{x / f:.{digits}g} {p}{unit}"


def sf_text(sf: float) -> Text:
    if not np.isfinite(sf):
        return Text("    ∞ ✔ OK", style="green")
    if sf >= SF_OK:
        return Text(f"{sf:5.2f} ✔ OK", style="green")
    if sf >= SF_WARN:
        return Text(f"{sf:5.2f} ▲ LOW", style="bold yellow")
    return Text(f"{sf:5.2f} ✖ FAIL", style="bold red")


def _header(res: Result, title: str | None) -> Panel:
    cfg, b = res.cfg, res.blade
    mat = cfg["structure"]["material"]
    U, W = res.U, rpm(res)
    if len(U) == 1:
        op = f"U∞ = {U[0]:g} m/s · {W[0]:.0f} RPM · TSR {res.Omega[0] * b.rotor.R / U[0]:.2f}"
    else:
        span = lambda a, f: f"{f(a.min())}–{f(a.max())}" if np.ptp(a) > 0 else f(a[0])
        op = (f"U∞ = {span(U, lambda v: f'{v:g}')} m/s · {span(W, lambda v: f'{v:.0f}')} RPM "
              f"· {len(U)} operating points")
    aero = "fallback aero" if b.polar.kind == "fallback" else "polar table"
    lines = Text.assemble(
        ("Rotor     ", "dim"), f"NACA {cfg['airfoil']['naca']} · B = {int(b.rotor.B)} · "
        f"R = {eng(b.rotor.R, 'm')} · r_hub = {eng(b.rotor.r_hub, 'm')} · {aero}\n",
        ("Operating ", "dim"), op + "\n",
        ("Structure ", "dim"), f"{cfg['structure']['model']} section · E = {eng(mat['E'], 'Pa')}"
        f" · σ_allow = +{eng(mat['sigma_allow_tension'], 'Pa')} / "
        f"−{eng(mat['sigma_allow_compression'], 'Pa')}",
    )
    return Panel(lines, title=f"[bold]bem-blade[/] · {title}" if title else "[bold]bem-blade[/]",
                 title_align="left", border_style="blue", box=box.ROUNDED)


def _argmax2(a):
    """(op, station) index of the max of a 2-D array."""
    return np.unravel_index(np.argmax(a), a.shape)


def _table(**kw) -> Table:
    return Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold dim",
                 **kw)


def _max_moments(res: Result) -> Panel:
    o, r = res.out, np.asarray(res.blade.geom.r)
    n, R = len(r), res.blade.rotor.R
    sweep = len(res.U) > 1
    Mf, Me = o["M_flap"][:, :n], o["M_edge"][:, :n]
    Mr = np.hypot(Mf, Me)

    t = _table()
    t.add_column("", style="bold", no_wrap=True)
    t.add_column("max", justify="right", style="bold cyan", no_wrap=True)
    t.add_column("at r", justify="right", no_wrap=True)
    t.add_column("r/R", justify="right", style="dim", no_wrap=True)
    if sweep:
        t.add_column("worst at", style="magenta", no_wrap=True)

    def row(name, value, j, i=None, unit="N·m"):
        cells = [name, eng(value, unit), eng(r[j], "m"), f"{r[j] / R:.2f}"]
        if sweep:
            cells.append(op_label(res, i) if i is not None else "—")
        t.add_row(*cells)

    for name, mag, signed in (("M_flap  flapwise", np.abs(Mf), Mf),
                              ("M_edge  edgewise", np.abs(Me), Me),
                              ("M_res   resultant", Mr, Mr)):
        i, j = _argmax2(mag)
        row(name, float(signed[i, j]), j, i)
    extra = []
    if res.blade.include_gravity:
        row("M_grav  gravity", float(np.max(np.abs(o["M_grav"][:, 0]))), 0)
        extra.append("M_grav is edgewise, applied as ± load cases in the stress check")
    if res.blade.include_centrifugal:
        i = int(np.argmax(o["N"][:, 0]))
        row("N       centrifugal", float(o["N"][i, 0]), 0, i, unit="N")
    cap = Text("M_flap/M_edge/M_res: aerodynamic, per blade. " + "; ".join(extra), style="dim")
    return Panel(Group(t, cap), title="[bold]MAX MOMENTS[/]", title_align="left",
                 border_style="cyan", box=box.ROUNDED)


def _short_loc(loc: str) -> str:
    """'lower x/c=0.130 g+t' -> 'lower 13%c, g+t'."""
    parts = loc.split()
    out = []
    for p in parts:
        if p.startswith("x/c="):
            out.append(f"{float(p[4:]):.0%}c")
        else:
            out.append(p)
    return " ".join(out)


def _strength(res: Result) -> Panel:
    pts = [point_summary(res, i) for i in range(len(res.U))]
    sweep = len(pts) > 1
    it = int(np.argmax([p["sigma_max_tension"]["value"] for p in pts]))
    ic = int(np.argmin([p["sigma_max_compression"]["value"] for p in pts]))

    t = _table()
    t.add_column("", style="bold", no_wrap=True)
    t.add_column("max σ_z", justify="right", style="bold cyan", no_wrap=True)
    t.add_column("at r", justify="right", no_wrap=True)
    t.add_column("where", style="dim", no_wrap=True)
    if sweep:
        t.add_column("worst at", style="magenta", no_wrap=True)
    t.add_column("SF", no_wrap=True)
    for label, i, key, sf in (("tension", it, "sigma_max_tension", "SF_tension"),
                              ("compression", ic, "sigma_max_compression", "SF_compression")):
        s = pts[i][key]
        cells = [label, eng(s["value"], "Pa"), eng(s["r"], "m"), _short_loc(s["location"])]
        if sweep:
            cells.append(op_label(res, i))
        t.add_row(*cells, sf_text(pts[i][sf]))
    tip = [abs(p["tip_deflection_n"]) for p in pts]
    k = int(np.argmax(tip))
    foot = Text.assemble(("tip deflection ", "bold"), (eng(tip[k], "m"), "bold cyan"),
                         (f" ({tip[k] / res.blade.rotor.R:.1%} of R)", "dim"),
                         (f"  {op_label(res, k)}" if sweep else "", "magenta"))
    legend = Text(f"SF  ✔ ≥ {SF_OK:g}   ▲ {SF_WARN:g}–{SF_OK:g}   ✖ < {SF_WARN:g}   "
                  "· g±t = gravity load case", style="dim")
    return Panel(Group(t, foot, legend), title="[bold]STRENGTH[/]", title_align="left",
                 border_style="cyan", box=box.ROUNDED)


def _sweep_table(res: Result) -> Group:
    o, n = res.out, len(res.blade.geom.r)
    pts = [point_summary(res, i) for i in range(len(res.U))]
    vary_u = np.ptp(res.U) > 0
    Mf0, Me0 = o["M_flap"][:, 0], o["M_edge"][:, 0]
    Mres = np.hypot(o["M_flap"][:, :n], o["M_edge"][:, :n]).max(axis=1)
    smax = o["sigma_max"].max(axis=1)
    smin = o["sigma_min"].min(axis=1)

    def unit(vals, u):
        f, p = _prefix(float(np.max(np.abs(vals))) or 1.0)
        return f, f"{p}{u}"

    fP, uP = unit(o["P"], "W")
    fT, uT = unit(o["T"], "N")
    fM, uM = unit(np.r_[Mf0, Me0], "N·m")
    fS, uS = unit(np.r_[smax, smin], "Pa")

    t = Table(box=box.SIMPLE_HEAVY, header_style="bold", pad_edge=False, collapse_padding=True,
              show_edge=False)
    col = lambda h, **k: t.add_column(escape(h), justify="right", no_wrap=True, **k)
    if vary_u:
        col("U∞\n[m/s]")
    col("RPM\n")
    col("TSR\n")
    col(f"P\n[{uP}]")
    col("C_P\n")
    col(f"T\n[{uT}]")
    col(f"M_flap\n[{uM}]", style="cyan")
    col(f"M_edge\n[{uM}]", style="cyan")
    col(f"σ_t\n[{uS}]")
    col(f"σ_c\n[{uS}]")
    t.add_column("min SF\n", no_wrap=True)
    t.add_column("\n", no_wrap=True)

    best_cp = int(np.argmax(o["CP"]))
    worst_m = int(np.argmax(Mres))
    worst_sf = int(np.argmin([min(p["SF_tension"], p["SF_compression"]) for p in pts]))
    stalled = defaultdict(int)
    for kind, i, rs in res.flags:
        if kind == "stall":
            stalled[i] = len(rs)
    for i, p in enumerate(pts):
        marks = Text()
        if i == best_cp:
            marks.append("★", style="bold green")
        if i == worst_m:
            marks.append("▲", style="bold magenta")
        if i == worst_sf:
            marks.append("◆", style="bold red")
        if stalled[i] > n // 2:
            marks.append("~", style="yellow")
        hot = "bold magenta" if i == worst_m else ""
        sf = min(p["SF_tension"], p["SF_compression"])
        t.add_row(*([f"{res.U[i]:g}"] if vary_u else []),
                  Text(f"{p['rpm']:.0f}", style=hot), f"{p['tsr']:.2f}",
                  f"{o['P'][i] / fP:.3g}",
                  Text(f"{o['CP'][i]:.3f}", style="bold green" if i == best_cp else ""),
                  f"{o['T'][i] / fT:.3g}",
                  Text(f"{Mf0[i] / fM:.3g}", style=hot or "cyan"),
                  Text(f"{Me0[i] / fM:.3g}", style="cyan"),
                  f"{smax[i] / fS:.3g}", f"{smin[i] / fS:.3g}", sf_text(sf), marks)
    legend = Text.assemble(
        ("★", "bold green"), " best C_P   ", ("▲", "bold magenta"), " max moment   ",
        ("◆", "bold red"), " lowest SF   ", ("~", "yellow"), " >half the stations stalled   ",
        ("M_flap/M_edge at the root", "dim"))
    return Group(Text("OPERATING POINTS", style="bold"), t, legend)


NOTE_TEXT = {
    "stall": "|α| > α_stall",
    "extrapolated": "polar extrapolated outside table (α or Re)",
    "region": "unusual BEM state (φ outside (0°, 90°])",
    "no_bracket": "no BEM solution, loads set to 0",
    "residual": "BEM residual > 1e-6 (pole, not root)",
    "betz": "C_P above Betz: check inputs",
    "deflection": "tip deflection > 0.1 R",
}
SEVERE = {"no_bracket", "residual", "betz"}


def _notes(res: Result) -> Panel | None:
    n, n_op = len(res.blade.geom.r), len(res.U)
    lines = []
    by_kind = defaultdict(list)
    for kind, i, rs in res.flags:
        by_kind[kind].append((i, rs))
    for kind, items in by_kind.items():
        style = "bold red" if kind in SEVERE else "yellow"
        worst_i, worst_rs = max(items, key=lambda x: len(x[1]))
        where = (f", up to {len(worst_rs)}/{n} stations ({op_label(res, worst_i)})"
                 if worst_rs else "")
        pts = f"{len(items)}/{n_op} points" if n_op > 1 else "this point"
        lines.append(Text.assemble(("● ", style), (NOTE_TEXT.get(kind, kind), style),
                                   (f" — {pts}{where}", "")))
    flagged = {w for w in res.warnings if w.startswith("[")}
    for w in res.warnings:
        if w not in flagged:
            lines.append(Text.assemble(("● ", "dim"), (w, "dim")))
    if not lines:
        return None
    return Panel(Group(*lines), title="[bold]Notes[/] (full list in summary.json)",
                 title_align="left", border_style="yellow", box=box.ROUNDED)


def print_summary(res: Result, title: str | None = None, out_dir=None,
                  console: Console | None = None) -> None:
    con = console or Console(highlight=False)
    con.print(_header(res, title))
    con.print(_max_moments(res))
    con.print(_strength(res))
    if len(res.U) > 1:
        con.print(_sweep_table(res))
    else:
        p = point_summary(res, 0)
        con.print(Text.assemble(
            ("Performance  ", "bold"), f"P = {eng(p['P'], 'W')}   T = {eng(p['T'], 'N')}   "
            f"Q = {eng(p['Q'], 'N·m')}   ", ("C_P = ", ""), (f"{p['CP']:.3f}", "bold green"),
            f"   C_T = {p['CT']:.3f}"))
    notes = _notes(res)
    if notes is not None:
        con.print(notes)
    if out_dir is not None:
        con.print(Text.assemble(("Outputs → ", "dim"), (str(out_dir), "bold underline")))


def summary_text(res: Result, title: str | None = None, width: int = 140) -> str:
    """Plain-text rendering (no ANSI), e.g. for logs."""
    con = Console(record=True, width=width, color_system=None, file=io.StringIO())
    print_summary(res, title, console=con)
    return con.export_text()
