"""NACA 4-digit geometry and thin-airfoil zero-lift angle (PLAN.md §3).

Pure NumPy: geometry is built once per run, so there is nothing for JAX to speed up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class Contour:
    """Closed CCW contour in normalized chord coordinates (x' LE->TE, y' toward suction side).

    Order: TE -> upper surface -> LE -> lower surface (the closing TE point is implicit).
    """

    xy: np.ndarray        # (N, 2)
    is_upper: np.ndarray  # (N,) bool, True for upper (suction) surface points incl. TE and LE


def parse_naca4(code: str) -> tuple[float, float, float]:
    """'MPXX' -> (m, p, t) as chord fractions."""
    code = str(code).strip().upper().removeprefix("NACA").strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"expected a 4-digit NACA designation, got {code!r}")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    if t <= 0:
        raise ValueError(f"NACA {code}: thickness must be > 0")
    return m, p, t


def camber_line(x: np.ndarray, m: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean camber line y_c and slope dy_c/dx."""
    x = np.asarray(x, dtype=float)
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    fore = x < p
    yc = np.where(fore, m / p**2 * (2 * p * x - x**2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(fore, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return yc, dyc


def thickness(x: np.ndarray, t: float) -> np.ndarray:
    """Half-thickness with the closed-trailing-edge coefficient (-0.1036)."""
    x = np.asarray(x, dtype=float)
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1036 * x**4)


def naca4_contour(code: str, n_per_surface: int = 150) -> Contour:
    """Cosine-spaced closed CCW contour, normalized by chord."""
    m, p, t = parse_naca4(code)
    beta = np.linspace(0.0, np.pi, n_per_surface)
    x = 0.5 * (1.0 - np.cos(beta))          # 0 (LE) -> 1 (TE)
    yc, dyc = camber_line(x, m, p)
    yt = thickness(x, t)
    psi = np.arctan(dyc)
    xu, yu = x - yt * np.sin(psi), yc + yt * np.cos(psi)
    xl, yl = x + yt * np.sin(psi), yc - yt * np.cos(psi)

    upper = np.column_stack([xu, yu])[::-1]  # TE -> LE (includes both)
    lower = np.column_stack([xl, yl])[1:-1]  # LE -> TE, shared LE/TE removed
    xy = np.vstack([upper, lower])
    is_upper = np.r_[np.ones(len(upper), bool), np.zeros(len(lower), bool)]
    return Contour(xy=xy, is_upper=is_upper)


def zero_lift_angle(code: str) -> float:
    """Thin-airfoil alpha_L0 [rad] = -(1/pi) * int_0^pi dyc/dx (cos th - 1) dth."""
    m, p, _ = parse_naca4(code)
    if m == 0.0 or p == 0.0:
        return 0.0

    def integrand(th: float) -> float:
        x = 0.5 * (1.0 - np.cos(th))
        return float(camber_line(np.array(x), m, p)[1]) * (np.cos(th) - 1.0)

    th_p = np.arccos(1.0 - 2.0 * p)  # slope kink at x = p
    val = quad(integrand, 0.0, th_p)[0] + quad(integrand, th_p, np.pi)[0]
    return -val / np.pi
