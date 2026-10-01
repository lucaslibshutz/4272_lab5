import copy
from pathlib import Path

import pytest

import bem_blade  # noqa: F401  (enables float64)

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "airfoil": {"naca": "2412", "alpha_stall_deg": 12.0, "cd0": None},
    "rotor": {"B": 3, "R": 40.0, "r_hub": 2.0, "pitch_deg": 0.0},
    "stations": {
        "r": [2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0,
              39.0],
        "chord": [3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 2.95, 2.61, 2.33, 2.11, 1.93, 1.78,
                  1.70],
        "theta_deg": [20.0, 20.0, 20.0, 18.1, 14.2, 10.3, 7.4, 5.3, 3.9, 2.5, 1.6, 0.9, 0.3,
                      -0.15, -0.44],
    },
    "operating": {"U_inf": 10.0, "Omega": 1.75},
    "structure": {
        "model": "shell", "skin_thickness": 0.02,
        "material": {"E": 40e9, "sigma_allow_tension": 200e6,
                     "sigma_allow_compression": 150e6, "density": 1900.0},
    },
}


@pytest.fixture
def base_cfg():
    return copy.deepcopy(BASE)
