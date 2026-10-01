import jax.numpy as jnp
import numpy as np
import pytest

from bem_blade.naca import naca4_contour
from bem_blade.section import airfoil_sections, central, polygon_moments, shell_moments
from bem_blade.structure import bending_stress


def test_solid_rectangle():
    b, h = 0.3, 0.1
    x = np.array([0, b, b, 0.0]) + 1.0   # offset origin to exercise the parallel-axis shift
    y = np.array([0, 0, h, h]) - 2.0
    A, xc, yc, Ixx, Iyy, Ixy = central(polygon_moments(jnp.asarray(x), jnp.asarray(y)))
    assert float(A) == pytest.approx(b * h)
    assert float(Iyy) == pytest.approx(b * h**3 / 12, rel=1e-12)
    assert float(Ixx) == pytest.approx(h * b**3 / 12, rel=1e-12)
    assert abs(float(Ixy)) < 1e-15


def test_thin_tube():
    rho, t = 0.5, 0.002
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    m = central(shell_moments(jnp.asarray(rho * np.cos(th)), jnp.asarray(rho * np.sin(th)), t))
    expected = np.pi * rho**3 * t
    assert float(m[3]) == pytest.approx(expected, rel=1e-3)
    assert float(m[4]) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("model", ["solid", "shell"])
def test_rotation_invariance_of_max_stress(model):
    """Rotate section and moment vector by the same angle: max |sigma| unchanged."""
    contour = jnp.asarray(naca4_contour("4412", 120).xy)
    th0, dth = 0.1, 0.7
    c = jnp.array([1.3])
    sigmas = []
    for th in (th0, th0 + dth):
        sec = airfoil_sections(contour, c, jnp.array([th]), model, jnp.array([0.01]))
        # moment fixed in the chord frame, expressed in rotor axes (same map as points)
        Mx, My = 120.0, -45.0
        M_t = -Mx * np.cos(th) + My * np.sin(th)
        M_n = Mx * np.sin(th) + My * np.cos(th)
        s = bending_stress(sec.pts_t, sec.pts_n, jnp.array([M_t]), jnp.array([M_n]),
                           jnp.zeros(1), sec.A, sec.S_tt, sec.S_nn, sec.S_tn)
        sigmas.append(float(jnp.max(jnp.abs(s))))
    assert sigmas[0] == pytest.approx(sigmas[1], rel=1e-10)


def test_rotation_preserves_invariants():
    """Area, trace and determinant of the second-moment tensor are rotation invariants."""
    contour = jnp.asarray(naca4_contour("2412", 120).xy)
    sec = airfoil_sections(contour, jnp.array([2.0, 2.0]), jnp.array([0.0, 0.9]), "solid")
    tr = np.asarray(sec.S_tt + sec.S_nn)
    det = np.asarray(sec.S_tt * sec.S_nn - sec.S_tn**2)
    assert np.asarray(sec.A)[0] == pytest.approx(np.asarray(sec.A)[1], rel=1e-12)
    assert tr[0] == pytest.approx(tr[1], rel=1e-12)
    assert det[0] == pytest.approx(det[1], rel=1e-10)
    assert np.all(np.asarray(sec.A) > 0)
