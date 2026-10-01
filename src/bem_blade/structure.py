"""Shear, bending moments, normal stress, deflection (PLAN.md §7). Pure JAX.

Per blade, cantilever clamped at r[0], free at R. Rotor-frame moment components:
    M_t = -M_flap,  M_n = M_edge.
"""

from __future__ import annotations

import jax.numpy as jnp

G = 9.81


def integrate_from_tip(y, r):
    """I(r_i) = int_{r_i}^{R} y dr' by cumulative trapezoid on the reversed array."""
    seg = 0.5 * (y[1:] + y[:-1]) * jnp.diff(r)
    return jnp.concatenate([jnp.cumsum(seg[::-1])[::-1], jnp.zeros(1)])


def integrate_from_root(y, r):
    """I(r_i) = int_{r_0}^{r_i} y dr' (cumulative trapezoid)."""
    seg = 0.5 * (y[1:] + y[:-1]) * jnp.diff(r)
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])


def shear_and_moment(p, r):
    """V(r) = int_r^R p,  M(r) = int_r^R V  (= int_r^R p (r' - r) dr')."""
    V = integrate_from_tip(p, r)
    return V, integrate_from_tip(V, r)


def bending_stress(x_t, x_n, M_t, M_n, N, A, S_tt, S_nn, S_tn):
    """Unsymmetric bending + axial stress at centroid-relative points (§7.2).

    x_t, x_n: (..., n_pts); section quantities broadcast with a trailing point axis added.
    """
    e = lambda v: jnp.asarray(v)[..., None]
    D = e(S_tt * S_nn - S_tn**2)
    return ((e(M_t * S_tt + M_n * S_tn) * x_n - e(M_t * S_tn + M_n * S_nn) * x_t) / D
            + e(N / A))


def curvatures(M_t, M_n, E, S_tt, S_nn, S_tn):
    """(delta_n'', delta_t'') from the linear strain field; delta_n positive downwind."""
    D = S_tt * S_nn - S_tn**2
    kn = -(M_t * S_tt + M_n * S_tn) / (E * D)
    kt = (M_t * S_tn + M_n * S_nn) / (E * D)
    return kn, kt


def deflection(kappa, r):
    """Double integration from the root with delta = delta' = 0 at r[0]."""
    return integrate_from_root(integrate_from_root(kappa, r), r)
