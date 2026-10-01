"""PNG plots (PLAN.md §9). Multiple operating points are overlaid with a sequential blue ramp."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .model import op_label  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#9ec5f4", "#6ea8ec", "#3f8ae0", "#2a78d6", "#1f5fae", "#174886", "#10335f"]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "font.size": 10, "legend.frameon": False,
})


def _xsweep(res):
    """Sweep x-axis: U if it varies, else RPM."""
    if np.ptp(res.U) > 0:
        return res.U, "U_inf [m/s]"
    return res.Omega * 60 / (2 * np.pi), "rotor speed [RPM]"


def _si(arr):
    """(scale, prefix) so that arr/scale has a readable magnitude."""
    m = float(np.nanmax(np.abs(arr))) if np.size(arr) else 0.0
    for f, pre in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if m >= f:
            return f, pre
    return 1.0, ""


CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("ramp", RAMP)
MAX_LEGEND = 6  # more operating points than this -> colorbar instead of a legend


def _colors(n):
    if n == 1:
        return [BLUE]
    return [CMAP(t) for t in np.linspace(0, 1, n)]


def _key(fig, axes, res):
    """Legend for a few operating points, a colorbar for many."""
    n = len(res.U)
    if n == 1:
        return
    if n <= MAX_LEGEND:
        np.ravel(axes)[0].legend(fontsize=8, ncol=min(n, 3))
        return
    x, label = _xsweep(res)
    sm = matplotlib.cm.ScalarMappable(cmap=CMAP, norm=matplotlib.colors.Normalize(x.min(), x.max()))
    fig.colorbar(sm, ax=axes, label=label, fraction=0.05, pad=0.02, aspect=40)


def _pair(ax, x, ys, labels, ylabel, colors):
    """Two quantities with the same unit on one axis (BLUE solid, ORANGE dashed)."""
    for y, lab, col, ls in zip(ys, labels, colors, ("-", "--")):
        ax.plot(x, y, color=col, ls=ls, label=lab)
    ax.set_ylabel(ylabel)


def make_plots(res, out_dir: Path) -> None:
    o, g = res.out, res.blade.geom
    r, r_ext = np.asarray(g.r), np.asarray(g.r_ext)
    n_op = len(res.U)
    cols = _colors(n_op)
    mat = res.cfg["structure"]["material"]

    def per_op(fname, title, panels):
        fig, axes = plt.subplots(len(panels), 1, figsize=(7, 3.0 * len(panels)), sharex=True,
                                 squeeze=False, layout="constrained")
        for ax, (ylabel, getter, xs) in zip(axes[:, 0], panels):
            for i in range(n_op):
                lab = op_label(res, i) if n_op > 1 else None
                ax.plot(xs, getter(i), color=cols[i], label=lab)
            ax.set_ylabel(ylabel)
            ax.axhline(0, color=INK2, lw=0.8)
        axes[-1, 0].set_xlabel("r [m]")
        axes[0, 0].set_title(title, loc="left")
        _key(fig, axes, res)
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    fp, pp = _si(np.r_[o["pN_ext"].ravel(), o["pT_ext"].ravel()])
    fm, pm = _si(np.r_[o["M_flap"].ravel(), o["M_edge"].ravel()])
    per_op("loads.png", "Aerodynamic load per unit span (per blade)", [
        (f"p_N (flapwise) [{pp}N/m]", lambda i: o["pN_ext"][i] / fp, r_ext),
        (f"p_T (edgewise) [{pp}N/m]", lambda i: o["pT_ext"][i] / fp, r_ext)])
    per_op("moments.png", "Bending moments (aerodynamic)", [
        (f"M_flap [{pm}N m]", lambda i: o["M_flap"][i] / fm, r_ext),
        (f"M_edge [{pm}N m]", lambda i: o["M_edge"][i] / fm, r_ext)])
    per_op("induction.png", "Axial induction and angle of attack", [
        ("a [-]", lambda i: o["a"][i], r),
        ("alpha [deg]", lambda i: np.rad2deg(o["alpha"][i]), r)])

    # Stress extremes with allowables
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for i in range(n_op):
        lab = op_label(res, i) if n_op > 1 else None
        ax.plot(r, o["sigma_max"][i] / 1e6, color=cols[i], label=lab)
        ax.plot(r, o["sigma_min"][i] / 1e6, color=cols[i], ls="--")
    for val, txt in ((mat["sigma_allow_tension"], "allowable tension"),
                     (-mat["sigma_allow_compression"], "allowable compression")):
        ax.axhline(val / 1e6, color=INK2, lw=1.2, ls=":")
        ax.annotate(txt, (r[-1], val / 1e6), ha="right", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("r [m]")
    ax.set_ylabel("sigma_z [MPa]")
    ax.set_title("Max tensile (solid) and compressive (dashed) normal stress", loc="left")
    _key(fig, ax, res)
    fig.savefig(out_dir / "stress.png", dpi=150)
    plt.close(fig)

    if n_op > 1:  # performance vs wind speed: two panels, never a dual axis
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout="constrained")
        x, xlabel = _xsweep(res)
        _pair(axes[0], x, [o["CP"], o["CT"]], ["C_P", "C_T"], "coefficient [-]",
              (BLUE, ORANGE))
        axes[0].axhline(16 / 27, color=INK2, lw=1.0, ls=":")
        axes[0].annotate("Betz 16/27", (x[0], 16 / 27), va="bottom", fontsize=8, color=INK2)
        axes[0].legend(fontsize=8)
        axes[1].plot(x, o["M_flap"][:, 0], color=BLUE, marker="o", ms=4)
        axes[1].set_ylabel("root M_flap [N m]")
        axes[1].set_xlabel(xlabel)
        axes[0].set_title("Rotor performance and root flap moment across the sweep", loc="left")
        fig.savefig(out_dir / "sweep.png", dpi=150)
        plt.close(fig)
