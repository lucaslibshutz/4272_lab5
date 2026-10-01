import numpy as np
import pytest

from bem_blade.naca import naca4_contour, thickness, zero_lift_angle
from bem_blade.section import polygon_moments


def test_naca0012_max_thickness():
    x = np.linspace(0, 1, 100001)
    yt = thickness(x, 0.12)
    assert 2 * yt.max() == pytest.approx(0.12, rel=1e-3)
    assert x[np.argmax(yt)] == pytest.approx(0.30, abs=0.01)


def test_naca0012_area():
    """Closed-TE thickness (-0.1036, §3) integrates to 0.68088 t c, not the 0.685 t c quoted in
    PLAN.md §11 -- that figure belongs to the open-TE coefficient (-0.1015), checked below."""
    xy = naca4_contour("0012", 200).xy
    A = float(polygon_moments(xy[:, 0], xy[:, 1])[0])  # CCW in chord frame -> positive
    closed = 2 * 5 * (0.2969 * 2 / 3 - 0.1260 / 2 - 0.3516 / 3 + 0.2843 / 4 - 0.1036 / 5)
    open_te = 2 * 5 * (0.2969 * 2 / 3 - 0.1260 / 2 - 0.3516 / 3 + 0.2843 / 4 - 0.1015 / 5)
    assert closed == pytest.approx(0.6809, abs=1e-4)
    assert open_te == pytest.approx(0.685, abs=1e-3)
    assert A == pytest.approx(closed * 0.12, rel=1e-4)


def test_contour_closed_and_ccw():
    xy = naca4_contour("4412", 150).xy
    assert np.allclose(xy[0], [1.0, 0.0], atol=1e-12)        # starts at TE
    assert float(polygon_moments(xy[:, 0], xy[:, 1])[0]) > 0  # CCW
    assert len(np.unique(xy.round(12), axis=0)) == len(xy)   # no duplicated LE/TE


def test_zero_lift_angle():
    assert zero_lift_angle("0012") == 0.0
    assert np.rad2deg(zero_lift_angle("2412")) == pytest.approx(-2.08, abs=0.01)
