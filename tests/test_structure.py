import jax.numpy as jnp
import numpy as np
import pytest

from bem_blade.structure import bending_stress, curvatures, deflection, shear_and_moment


def test_symmetric_reduction():
    rng = np.random.default_rng(0)
    xt, xn = rng.normal(size=50), rng.normal(size=50)
    Mt, Mn, Stt, Snn = 3.0, -2.0, 0.7, 0.2
    s = bending_stress(xt, xn, Mt, Mn, 0.0, 1.0, Stt, Snn, 0.0)
    assert np.allclose(s, Mt * xn / Snn - Mn * xt / Stt)


def test_pure_Mt_resultants_on_rectangle():
    """Re-integrate sigma over a fine rectangle grid: int sigma x_n = M_t, int sigma x_t = 0."""
    b, h, n = 0.4, 0.1, 800
    xt = (np.arange(n) + 0.5) / n * b - b / 2
    xn = (np.arange(n) + 0.5) / n * h - h / 2
    XT, XN = np.meshgrid(xt, xn)
    dA = (b / n) * (h / n)
    Stt, Snn = h * b**3 / 12, b * h**3 / 12
    Mt = 5.0
    s = np.asarray(bending_stress(XT.ravel(), XN.ravel(), Mt, 0.0, 0.0, b * h, Stt, Snn, 0.0))
    assert np.sum(s * XN.ravel()) * dA == pytest.approx(Mt, rel=1e-5)
    assert abs(np.sum(s * XT.ravel()) * dA) < 1e-10
    # sign check (§7.2): pure flap (M_t < 0) -> compression downwind (x_n > 0)
    s_flap = bending_stress(np.array([0.0]), np.array([h / 2]), -Mt, 0.0, 0.0, b * h, Stt, Snn, 0)
    assert float(s_flap[0]) < 0


def test_uniform_load():
    L, p = 7.0, 3.0
    r = jnp.linspace(0, L, 11)
    V, M = shear_and_moment(jnp.full_like(r, p), r)
    assert float(V[0]) == pytest.approx(p * L)
    assert float(M[0]) == pytest.approx(p * L**2 / 2, rel=1e-12)
    assert float(M[-1]) == 0.0 and float(V[-1]) == 0.0


def test_linear_load():
    L, p0 = 5.0, 2.0
    r = jnp.linspace(0, L, 2001)
    _, M = shear_and_moment(p0 * r / L, r)
    assert float(M[0]) == pytest.approx(p0 * L**2 / 3, rel=1e-6)


def test_cantilever_tip_deflection():
    """Uniform load on a uniform beam: delta_tip = p L^4 / (8 E I)."""
    L, p, E, I = 2.0, 10.0, 1e9, 1e-6
    r = jnp.linspace(0, L, 4001)
    _, M = shear_and_moment(jnp.full_like(r, p), r)
    kn, _ = curvatures(-M, 0.0 * M, E, I, I, 0.0)   # M_t = -M_flap
    d = deflection(kn, r)
    assert float(d[-1]) == pytest.approx(p * L**4 / (8 * E * I), rel=1e-5)
