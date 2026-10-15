"""Blade-element model (solver.model = blade_element) against the lab write-up, §2-§5."""

import importlib.util

import jax.numpy as jnp
import numpy as np
import pytest

from bem_blade import load_config, run
from bem_blade.bem import trapezoid_weights
from bem_blade.model import build_blade, rotor_torque

from conftest import ROOT

BLADE7 = ROOT / "examples" / "blade7.yaml"


def _handout_method():
    spec = importlib.util.spec_from_file_location("handout_method",
                                                  ROOT / "scripts" / "handout_method.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def at_3600():
    cfg = load_config(BLADE7, ["operating.rpm=3600"])
    res = run(cfg)
    return cfg, res, {k: (v[0] if hasattr(v, "shape") and v.shape[:1] == (1,) else v)
                      for k, v in res.out.items()}


@pytest.fixture(scope="module")
def freewheel():
    cfg = load_config(BLADE7)
    return cfg, run(cfg)


def test_no_induction_and_geometric_inflow(at_3600):
    """a = a' = 0, F = 1, phi = atan2(U, Omega r), W = |(Omega r, U)| at every station."""
    cfg, res, o = at_3600
    r, U, Om = np.asarray(cfg["stations"]["r"]), 15.0, 3600 * 2 * np.pi / 60
    assert np.all(o["a"] == 0) and np.all(o["ap"] == 0) and np.all(o["F"] == 1)
    assert np.allclose(o["phi"], np.arctan2(U, Om * r), rtol=0, atol=1e-14)
    assert np.allclose(o["W"], np.hypot(U, Om * r), rtol=1e-14)
    assert np.all(o["region"] == 1)          # hub and tip stations are solved and loaded
    assert o["pT"][0] != 0 and o["pT"][-1] != 0
    assert res.re_passes == 1                # W does not depend on the solution


def test_force_projection_is_rotation(at_3600):
    """[p_T; p_N] = M(phi) [L'; D'] with M = [[sin, -cos], [cos, sin]], and |.| is preserved."""
    cfg, _, o = at_3600
    c = np.asarray(cfg["stations"]["chord"])
    q = 0.5 * 1.225 * o["W"] ** 2
    L, D = q * c * o["cl"], q * c * o["cd"]
    s, co = np.sin(o["phi"]), np.cos(o["phi"])
    M = np.stack([np.stack([s, -co], -1), np.stack([co, s], -1)], -2)       # (n, 2, 2)
    pTN = np.einsum("nij,nj->ni", M, np.stack([L, D], -1))
    assert np.allclose(pTN[:, 0], o["pT"], rtol=1e-12, atol=1e-12)
    assert np.allclose(pTN[:, 1], o["pN"], rtol=1e-12, atol=1e-12)
    assert np.allclose(o["pT"] ** 2 + o["pN"] ** 2, (q * c) ** 2 * (o["cl"] ** 2 + o["cd"] ** 2),
                       rtol=1e-12)


@pytest.mark.parametrize("rpm", [1000, 3600, 17574, 30000])
def test_drive_condition(rpm):
    """Per station, p_T > 0  <=>  C_L / C_D > Omega r / U  (write-up eq. drive_condition)."""
    cfg = load_config(BLADE7, [f"operating.rpm={rpm}"])
    o = run(cfg).out
    r, U, Om = np.asarray(cfg["stations"]["r"]), 15.0, rpm * 2 * np.pi / 60
    assert np.all(o["cd"][0] > 0)
    assert np.array_equal(o["pT"][0] > 0, o["cl"][0] / o["cd"][0] > Om * r / U)


def test_trapezoid_weights_sum_to_blade_length():
    cfg = load_config(BLADE7)
    r = np.asarray(cfg["stations"]["r"])
    w = np.asarray(trapezoid_weights(jnp.asarray(r)))
    assert w.shape == r.shape
    assert w.sum() == pytest.approx(cfg["rotor"]["R"] - cfg["rotor"]["r_hub"], rel=1e-14)
    assert w[0] == pytest.approx(0.5 * (r[1] - r[0])) and w[-1] == pytest.approx(0.5 * (r[-1] - r[-2]))


def test_freewheel_torque_vanishes(freewheel):
    cfg, res = freewheel
    assert res.freewheel
    blade = build_blade(cfg)
    Om = res.Omega[0]
    Q_ss = rotor_torque(blade, 15.0, Om, cfg["solver"])[0]
    Q_ref = rotor_torque(blade, 15.0, 3600 * 2 * np.pi / 60, cfg["solver"])[0]
    assert abs(Q_ss) < 1e-3 * Q_ref
    # Q crosses from + to - at Omega_ss (the rotor speeds up below it and slows above it)
    dOm = 5 * 2 * np.pi / 60
    lo, hi = rotor_torque(blade, 15.0, [Om - dOm, Om + dOm], cfg["solver"])
    assert lo > 0 > hi


def test_freewheel_matches_handout_script(freewheel):
    """Package Omega_ss vs. the independent scripts/handout_method.py (trapezoid rule)."""
    _, res = freewheel
    hm = _handout_method()
    ref = hm.omega_ss_rpm(load_polar_for(hm), "trap", hm.blade7_geometry(), 15.0)
    rpm_ss = res.Omega[0] * 60 / (2 * np.pi)
    assert rpm_ss == pytest.approx(ref, abs=2.0)


def load_polar_for(hm):
    from bem_blade.aero import load_polar_table
    return load_polar_table(hm.POLAR)[0]


def test_bem_default_unchanged(base_cfg):
    """Configs without solver.model keep the momentum model."""
    cfg = load_config(base_cfg)
    assert cfg["solver"]["model"] == "bem"
    assert build_blade(cfg).model == "bem"
