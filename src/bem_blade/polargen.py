"""Optional polar-table generation with NeuralFoil (PLAN.md §4.1).

NeuralFoil is an optional extra (``uv sync --extra neuralfoil``); nothing else imports this module.
Verified against neuralfoil 0.3.3: ``get_aero_from_coordinates(coordinates, alpha[deg], Re,
model_size)`` with Selig-ordered coordinates (TE -> upper -> LE -> lower -> TE).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .naca import naca4_contour


def generate_polar_csv(naca: str, re_list, out_path: str | Path,
                       alpha_deg=np.arange(-20.0, 30.01, 0.5), model_size: str = "large") -> Path:
    try:
        import neuralfoil as nf
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("NeuralFoil not installed; run `uv sync --extra neuralfoil`") from e

    xy = naca4_contour(naca, 150).xy
    coords = np.vstack([xy, xy[:1]])  # close at the TE for Selig ordering
    alpha_deg = np.asarray(alpha_deg, float)
    rows = []
    for re in re_list:
        out = nf.get_aero_from_coordinates(coordinates=coords, alpha=alpha_deg, Re=float(re),
                                           model_size=model_size)
        for a, cl, cd in zip(alpha_deg, np.ravel(out["CL"]), np.ravel(out["CD"])):
            rows.append((float(re), a, cl, cd))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, np.array(rows), delimiter=",", header="Re,alpha_deg,Cl,Cd",
               comments="", fmt=["%.6g", "%.3f", "%.6f", "%.6f"])
    return out_path
