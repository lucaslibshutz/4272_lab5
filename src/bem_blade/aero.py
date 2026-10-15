"""Section aerodynamics C_l(alpha, Re), C_d(alpha, Re) (PLAN.md §4).

Both polar classes are JAX pytrees exposing the same ``cl_cd(alpha, Re)`` API, so they can be
passed straight into jitted/vmapped BEM code. ``alpha`` in radians.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


def wrap_angle(a):
    """Wrap to [-pi, pi)."""
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def flat_plate_cd0(Re):
    """Fully turbulent flat plate, both sides wetted: C_d0 = 2 * 0.074 / Re^(1/5)."""
    return 2 * 0.074 / Re**0.2


@dataclass(frozen=True)
class FallbackPolar:
    """Thin-airfoil lift + flat-plate drag, blended to a 2D flat plate past stall (§4.2-4.3)."""

    alpha_L0: float
    alpha_stall: float
    cd0: float                    # NaN -> Re-dependent flat-plate estimate
    delta: float = float(np.deg2rad(2.0))

    kind = "fallback"

    def cl_cd(self, alpha, Re):
        a = wrap_angle(alpha)
        cd0 = jnp.where(jnp.isnan(self.cd0), flat_plate_cd0(Re), self.cd0)
        da = a - self.alpha_L0
        cl_lin = 2 * jnp.pi * da
        cd_att = cd0 + 0.01 * da**2
        f = jax.nn.sigmoid((jnp.abs(a) - self.alpha_stall) / self.delta)
        cl = (1 - f) * cl_lin + f * jnp.sin(2 * a)
        cd = (1 - f) * cd_att + f * (cd0 + 2 * jnp.sin(a) ** 2)
        return cl, cd

    def in_table(self, alpha, Re):
        """Fallback has no table; every evaluation is 'model'."""
        return jnp.zeros(jnp.broadcast_shapes(jnp.shape(alpha), jnp.shape(Re)), bool)


jax.tree_util.register_dataclass(
    FallbackPolar, data_fields=["alpha_L0", "alpha_stall", "cd0", "delta"], meta_fields=[])


@dataclass(frozen=True)
class PolarTable:
    """Tabulated polars resampled onto a common alpha grid spanning [-180, 180] deg.

    Linear in alpha, linear in log(Re); Re outside the table is clamped to the nearest row.
    """

    log_re: jnp.ndarray      # (nRe,)
    alpha: jnp.ndarray       # (nA,) rad, common grid
    cl: jnp.ndarray          # (nRe, nA)
    cd: jnp.ndarray          # (nRe, nA)
    alpha_lo: float          # rad, range covered by the source table at every Re
    alpha_hi: float

    kind = "table"

    def _interp(self, tab, a, log_re):
        rows = jax.vmap(lambda row: jnp.interp(a, self.alpha, row))(tab)
        return jnp.interp(log_re, self.log_re, rows)

    def cl_cd(self, alpha, Re):
        @jnp.vectorize
        def one(a, re):
            a = wrap_angle(a)
            lr = jnp.log(re)
            return self._interp(self.cl, a, lr), self._interp(self.cd, a, lr)

        return one(alpha, Re)

    def in_table(self, alpha, Re):
        a = wrap_angle(alpha)
        lr = jnp.log(Re)
        return ((a >= self.alpha_lo) & (a <= self.alpha_hi)
                & (lr >= self.log_re[0] - 1e-9) & (lr <= self.log_re[-1] + 1e-9))


jax.tree_util.register_dataclass(
    PolarTable, data_fields=["log_re", "alpha", "cl", "cd", "alpha_lo", "alpha_hi"],
    meta_fields=[])


def load_polar_table(path: str | Path, d_alpha_deg: float = 0.1,
                     blend_deg: float = 10.0) -> tuple[PolarTable, list[str]]:
    """Read a CSV/whitespace table with columns ``Re, alpha_deg, Cl, Cd``.

    Returns the table and a list of warnings. Outside the tabulated alpha range each Re row is
    extended with the flat-plate post-stall model of §4.3 (Cl = sin 2a, Cd = Cd_min + 2 sin^2 a),
    reached by a linear blend from the row's last tabulated point over ``blend_deg``. The blend
    keeps Cl, Cd continuous in alpha: tables that stop near stall (e.g. XFOIL from airfoiltools,
    ~14-19 deg) would otherwise jump by ~60% in Cl at the edge.
    """
    path = Path(path)
    text = path.read_text()
    delim = "," if "," in text.splitlines()[0] else None
    data = np.genfromtxt(path, names=True, delimiter=delim, dtype=float)
    names = {n.lower(): n for n in data.dtype.names}
    try:
        re_col, a_col, cl_col, cd_col = (data[names[k]] for k in ("re", "alpha_deg", "cl", "cd"))
    except KeyError as e:
        raise ValueError(f"{path}: need columns Re, alpha_deg, Cl, Cd; got {data.dtype.names}") from e

    warnings: list[str] = []
    grid_deg = np.arange(-180.0, 180.0 + 1e-9, d_alpha_deg)
    grid = np.deg2rad(grid_deg)
    res = np.unique(re_col)
    cl_rows, cd_rows, lo, hi = [], [], [], []
    for re in res:
        sel = re_col == re
        order = np.argsort(a_col[sel])
        a, cl, cd = a_col[sel][order], cl_col[sel][order], cd_col[sel][order]
        cdmin = cd.min()
        # weight of the flat-plate model: 0 inside the table, ramping to 1 over blend_deg outside
        f = np.clip(np.maximum(a[0] - grid_deg, grid_deg - a[-1]) / blend_deg, 0.0, 1.0)
        cl_rows.append((1 - f) * np.interp(grid_deg, a, cl) + f * np.sin(2 * grid))
        cd_rows.append((1 - f) * np.interp(grid_deg, a, cd) + f * (cdmin + 2 * np.sin(grid) ** 2))
        lo.append(a[0])
        hi.append(a[-1])
    if min(lo) > -180.0 or max(hi) < 180.0:
        warnings.append(
            f"polar table covers alpha in [{max(lo):.1f}, {min(hi):.1f}] deg; extended with the "
            "flat-plate post-stall model (§4.3) outside that range")
    if len(res) == 1:  # duplicate the row so log-Re interpolation is well defined
        res = np.r_[res, res * 10.0]
        cl_rows, cd_rows = cl_rows * 2, cd_rows * 2
    table = PolarTable(
        log_re=jnp.asarray(np.log(res)), alpha=jnp.asarray(grid),
        cl=jnp.asarray(np.array(cl_rows)), cd=jnp.asarray(np.array(cd_rows)),
        alpha_lo=float(np.deg2rad(max(lo))), alpha_hi=float(np.deg2rad(min(hi))))
    return table, warnings
