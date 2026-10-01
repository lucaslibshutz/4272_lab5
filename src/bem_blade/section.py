"""Section structural properties (PLAN.md §6), vectorized over stations with JAX.

Frames: chord frame (x' LE->TE, y' toward suction side); rotor frame (x_t along t-hat,
x_n along n-hat). The map (§1)

    x_t = -x' cos(th) + y' sin(th),   x_n = x' sin(th) + y' cos(th)

has determinant -1 (the chord frame is left-handed about z-hat), so a CCW contour in the chord
frame is CW in the rotor frame. We map the points first and reverse their order before applying
the signed polygon formulas; shell (segment) formulas are orientation-independent.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SectionProps(NamedTuple):
    """Per-station properties about the centroid, rotor frame. Arrays lead with the station axis."""

    A: jnp.ndarray
    S_tt: jnp.ndarray     # int x_t^2 dA
    S_nn: jnp.ndarray     # int x_n^2 dA
    S_tn: jnp.ndarray     # int x_t x_n dA
    pts_t: jnp.ndarray    # (n_st, n_pts) centroid-relative stress-evaluation points
    pts_n: jnp.ndarray


def chord_to_rotor(xp, yp, theta):
    ct, st = jnp.cos(theta), jnp.sin(theta)
    return -xp * ct + yp * st, xp * st + yp * ct


def polygon_moments(x, y):
    """Solid polygon (CCW, implicitly closed): A, int x dA, int y dA, int x^2, int y^2, int xy."""
    x1, y1 = jnp.roll(x, -1), jnp.roll(y, -1)
    c = x * y1 - x1 * y
    A = 0.5 * jnp.sum(c)
    Sx = jnp.sum((x + x1) * c) / 6
    Sy = jnp.sum((y + y1) * c) / 6
    Ixx = jnp.sum(c * (x**2 + x * x1 + x1**2)) / 12
    Iyy = jnp.sum(c * (y**2 + y * y1 + y1**2)) / 12
    Ixy = jnp.sum(c * (x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y)) / 24
    return A, Sx, Sy, Ixx, Iyy, Ixy


def shell_moments(x, y, t):
    """Thin-walled closed midline of wall thickness t: same outputs as polygon_moments."""
    x1, y1 = jnp.roll(x, -1), jnp.roll(y, -1)
    tL = t * jnp.hypot(x1 - x, y1 - y)
    A = jnp.sum(tL)
    Sx = jnp.sum(tL * (x + x1)) / 2
    Sy = jnp.sum(tL * (y + y1)) / 2
    Ixx = jnp.sum(tL * (x**2 + x * x1 + x1**2)) / 3
    Iyy = jnp.sum(tL * (y**2 + y * y1 + y1**2)) / 3
    Ixy = jnp.sum(tL * (2 * x * y + x * y1 + x1 * y + 2 * x1 * y1)) / 6
    return A, Sx, Sy, Ixx, Iyy, Ixy


def central(moments):
    """Shift origin moments to the centroid (parallel-axis theorem)."""
    A, Sx, Sy, Ixx, Iyy, Ixy = moments
    xc, yc = Sx / A, Sy / A
    return A, xc, yc, Ixx - A * xc**2, Iyy - A * yc**2, Ixy - A * xc * yc


def _airfoil_station(contour, chord, theta, t_skin, shell: bool):
    xt, xn = chord_to_rotor(chord * contour[:, 0], chord * contour[:, 1], theta)
    if shell:
        m = shell_moments(xt, xn, t_skin)
    else:
        m = polygon_moments(xt[::-1], xn[::-1])  # restore CCW in the rotor frame
    A, ct, cn, S_tt, S_nn, S_tn = central(m)
    return A, S_tt, S_nn, S_tn, xt - ct, xn - cn


def airfoil_sections(contour, chord, theta, model: str, t_skin=None) -> SectionProps:
    """Scaled + rotated NACA sections. ``contour`` (n_pts, 2) normalized; ``theta`` in rad."""
    shell = model == "shell"
    t_skin = jnp.zeros_like(chord) if t_skin is None else jnp.broadcast_to(t_skin, chord.shape)
    fn = jax.vmap(lambda c, th, ts: _airfoil_station(contour, c, th, ts, shell))
    return SectionProps(*jax.jit(fn)(chord, theta, t_skin))


def user_sections(A, S11, S22, S12, pts_1, pts_2, centroid_1, centroid_2, theta=None
                  ) -> SectionProps:
    """User-supplied properties. If ``theta`` is given, inputs are in the chord frame
    (1 = x', 2 = y') and are rotated to the rotor frame; otherwise 1 = t, 2 = n."""
    p1 = jnp.asarray(pts_1) - jnp.asarray(centroid_1)[:, None]
    p2 = jnp.asarray(pts_2) - jnp.asarray(centroid_2)[:, None]
    A, S11, S22, S12 = map(jnp.asarray, (A, S11, S22, S12))
    if theta is None:
        return SectionProps(A, S11, S22, S12, p1, p2)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    # x_t = a x' + b y', x_n = d x' + e y'  =>  S_tt = a^2 S11 + 2ab S12 + b^2 S22, etc.
    a, b, d, e = -ct, st, st, ct
    S_tt = a * a * S11 + 2 * a * b * S12 + b * b * S22
    S_nn = d * d * S11 + 2 * d * e * S12 + e * e * S22
    S_tn = a * d * S11 + (a * e + b * d) * S12 + b * e * S22
    q_t, q_n = chord_to_rotor(p1, p2, theta[:, None])
    return SectionProps(A, S_tt, S_nn, S_tn, q_t, q_n)
