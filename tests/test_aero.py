"""Polar table extension past the tabulated alpha range (PLAN.md §4.1, §4.3)."""

import jax.numpy as jnp
import numpy as np
import pytest

from bem_blade.aero import load_polar_table


@pytest.fixture
def ragged_table(tmp_path):
    """Two Re rows with different alpha ranges, like XFOIL output that stops near stall."""
    rows = ["Re,alpha_deg,Cl,Cd"]
    for re, hi in ((5e4, 14.0), (1e5, 17.0)):
        for a in np.arange(-8.0, hi + 1e-9, 0.5):
            rows.append(f"{re},{a},{0.4 + 0.1 * a},{0.01 + 0.0005 * a**2}")
    p = tmp_path / "polar.csv"
    p.write_text("\n".join(rows))
    return p


def test_extension_is_continuous_at_table_edge(ragged_table):
    table, _ = load_polar_table(ragged_table)
    for re, hi in ((5e4, 14.0), (1e5, 17.0)):
        a = jnp.deg2rad(jnp.array([hi - 0.05, hi + 0.05]))
        cl, cd = (np.asarray(v) for v in table.cl_cd(a, jnp.full(2, re)))
        assert abs(cl[1] - cl[0]) < 0.01
        assert abs(cd[1] - cd[0]) < 0.001


def test_extension_reaches_flat_plate(ragged_table):
    blend = 10.0
    table, warnings = load_polar_table(ragged_table, blend_deg=blend)
    a_deg = 17.0 + blend + 5.0
    cl, cd = table.cl_cd(jnp.deg2rad(a_deg), 1e5)
    a = np.deg2rad(a_deg)
    assert float(cl) == pytest.approx(np.sin(2 * a), abs=1e-6)
    assert float(cd) == pytest.approx(0.01 + 2 * np.sin(a) ** 2, abs=1e-6)
    assert warnings  # the extension is reported


def test_in_table_uses_common_range(ragged_table):
    table, _ = load_polar_table(ragged_table)
    # 15 deg is tabulated at Re 1e5 only, so a lookup between the rows is flagged as extrapolated
    assert bool(table.in_table(jnp.deg2rad(13.0), 7e4))
    assert not bool(table.in_table(jnp.deg2rad(15.0), 7e4))
