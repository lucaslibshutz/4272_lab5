import copy

import numpy as np
import pytest
from scipy.optimize import brentq

from bem_blade import load_config, run
from bem_blade.model import build_blade

from conftest import ROOT


def _single(cfg):
    res = run(load_config(cfg))
    return res, {k: (v[0] if hasattr(v, "shape") and v.shape[:1] == (1,) else v)
                 for k, v in res.out.items()}


def test_tiny_chord_gives_geometric_inflow(base_cfg):
    base_cfg["stations"]["chord"] = [1e-7] * len(base_cfg["stations"]["r"])
    base_cfg["structure"]["model"] = "solid"
    res, o = _single(base_cfg)
    r = np.asarray(base_cfg["stations"]["r"])
    lam_r = 1.75 * r / 10.0
    inner = slice(1, None)  # r[0] = r_hub carries no load by construction
    assert np.allclose(o["a"][inner], 0, atol=1e-6)
    assert np.allclose(o["ap"][inner], 0, atol=1e-6)
    assert np.allclose(o["phi"][inner], np.arctan(1 / lam_r[inner]), atol=1e-6)


def test_momentum_balance(base_cfg):
    """B p_N = 4 pi r rho U^2 a (1-a) F in the momentum region (a <= 0.4)."""
    res, o = _single(base_cfg)
    r, rho, U = np.asarray(base_cfg["stations"]["r"]), 1.225, 10.0
    m = (o["region"] == 1) & (o["a"] <= 0.4) & (o["a"] > 0)
    assert m.sum() >= 10
    lhs = 3 * o["pN"][m]
    rhs = 4 * np.pi * r[m] * rho * U**2 * o["a"][m] * (1 - o["a"][m]) * o["F"][m]
    assert np.allclose(lhs, rhs, rtol=1e-6)


def test_matches_brentq_reference(base_cfg):
    """Independent NumPy + scipy.brentq implementation of §5 (region 1) at fixed Re."""
    cfg = load_config(base_cfg)
    blade = build_blade(cfg)
    res = run(cfg)
    polar = blade.polar
    B, R, rh = 3.0, 40.0, 2.0
    U, Om = 10.0, 1.75
    for j in range(1, len(cfg["stations"]["r"]) - 1):
        r, c = cfg["stations"]["r"][j], cfg["stations"]["chord"][j]
        th = np.deg2rad(cfg["stations"]["theta_deg"][j])
        Re = res.out["Re"][0, j]

        def resid(phi):
            cl, cd = (float(v) for v in polar.cl_cd(phi - th, Re))
            s, co = np.sin(phi), np.cos(phi)
            cn, ct = cl * co + cd * s, cl * s - cd * co
            F = (2 / np.pi * np.arccos(np.exp(-B * (R - r) / (2 * r * abs(s))))
                 * 2 / np.pi * np.arccos(np.exp(-B * (r - rh) / (2 * rh * abs(s)))))
            sig = B * c / (2 * np.pi * r)
            k, kp = sig * cn / (4 * F * s**2), sig * ct / (4 * F * s * co)
            if k <= 2 / 3:
                a = k / (1 + k)
            else:
                g1, g2, g3 = 2 * F * k - (10 / 9 - F), 2 * F * k - F * (4 / 3 - F), \
                    2 * F * k - (25 / 9 - 2 * F)
                a = 1 - 1 / (2 * np.sqrt(g2)) if abs(g3) < 1e-6 else (g1 - np.sqrt(g2)) / g3
            return s / (1 - a) - co * (1 - kp) / (Om * r / U)

        phi_ref = brentq(resid, 1e-6, np.pi / 2, xtol=1e-14)
        assert res.out["phi"][0, j] == pytest.approx(phi_ref, abs=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_cp_below_betz_random_blades(base_cfg, seed):
    rng = np.random.default_rng(seed)
    n = len(base_cfg["stations"]["r"])
    base_cfg["stations"]["chord"] = list(rng.uniform(0.5, 4.0, n))
    base_cfg["stations"]["theta_deg"] = list(rng.uniform(-5, 25, n))
    base_cfg["operating"] = {"U_inf": float(rng.uniform(4, 20)), "tsr": float(rng.uniform(2, 12))}
    res = run(load_config(base_cfg))
    assert res.out["CP"][0] < 16 / 27


def test_units_scaling(base_cfg):
    """Lengths x2, Omega /2 (same lambda): C_T, C_P unchanged; moments x8.

    cd0 fixed so the polar is Re-independent (Re doubles with chord).
    """
    base_cfg["airfoil"]["cd0"] = 0.01
    base_cfg["structure"]["include_gravity"] = False
    _, o1 = _single(copy.deepcopy(base_cfg))
    s = base_cfg
    s["rotor"]["R"] *= 2
    s["rotor"]["r_hub"] *= 2
    s["stations"]["r"] = [2 * v for v in s["stations"]["r"]]
    s["stations"]["chord"] = [2 * v for v in s["stations"]["chord"]]
    s["operating"]["Omega"] /= 2
    _, o2 = _single(s)
    assert o2["CT"] == pytest.approx(o1["CT"], rel=1e-10)
    assert o2["CP"] == pytest.approx(o1["CP"], rel=1e-10)
    assert np.allclose(o2["M_flap"], 8 * o1["M_flap"], rtol=1e-9)
    assert np.allclose(o2["M_edge"], 8 * o1["M_edge"], rtol=1e-9)


def test_sweep_equals_individual_runs(base_cfg):
    """vmap over operating points gives the same answer as separate single-point runs."""
    base_cfg["operating"] = {"U_inf": [6.0, 11.0], "tsr": 7.0}
    sweep = run(load_config(base_cfg)).out
    for i, u in enumerate([6.0, 11.0]):
        cfg = copy.deepcopy(base_cfg)
        cfg["operating"] = {"U_inf": u, "tsr": 7.0}
        one = run(load_config(cfg)).out
        assert np.allclose(sweep["M_flap"][i], one["M_flap"][0], rtol=1e-8)
        assert sweep["CP"][i] == pytest.approx(one["CP"][0], rel=1e-8)


def test_blade7_example_runs():
    cfg = load_config(ROOT / "examples" / "blade7.yaml",
                      overrides=["operating.rpm=[2000, 3600, 5000]"])
    res = run(cfg)
    o = res.out
    assert np.all(o["CP"] < 16 / 27)
    assert np.all(o["region"][:, 1:-1] == 1)  # r[0] = r_hub and r[-1] = R carry F = 0
    # design point (TSR 4.47 ~ 3600 RPM) should be the best of the three
    assert np.argmax(o["CP"]) == 1
