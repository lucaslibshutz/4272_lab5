"""Steady BEM -> blade root bending moment and normal stress (see PLAN.md)."""

import jax

jax.config.update("jax_enable_x64", True)  # float64 everywhere: BEM residuals and stresses

from .config import load_config  # noqa: E402
from .model import Result, build_blade, run  # noqa: E402
from .report import run_from_config, write_outputs  # noqa: E402
from .console import print_summary, summary_text  # noqa: E402

__all__ = ["load_config", "run", "build_blade", "Result", "write_outputs", "print_summary",
           "summary_text", "run_from_config"]
