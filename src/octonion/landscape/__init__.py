"""Optimization landscape analysis toolkit.

Provides Hessian eigenspectrum analysis, loss surface curvature measurement,
gradient statistics collection, and go/no-go gate evaluation for
characterizing the optimization landscape of models using octonionic,
quaternionic, complex, and real-valued representations.
"""

from octonion.landscape._curvature import measure_curvature  # noqa: F401
from octonion.landscape._experiment import (  # noqa: F401
    LandscapeConfig,
    run_landscape_experiment,
)
from octonion.landscape._gate import GateVerdict, evaluate_gate  # noqa: F401
from octonion.landscape._gradient_stats import (  # noqa: F401
    collect_gradient_stats,
    collect_gradient_variance_across_seeds,
)
from octonion.landscape._hessian import (  # noqa: F401
    compute_full_hessian,
    compute_hessian_spectrum,
    hessian_vector_product,
    stochastic_lanczos,
)

__all__ = [
    "collect_gradient_stats",
    "collect_gradient_variance_across_seeds",
    "compute_full_hessian",
    "compute_hessian_spectrum",
    "evaluate_gate",
    "GateVerdict",
    "hessian_vector_product",
    "LandscapeConfig",
    "measure_curvature",
    "run_landscape_experiment",
    "stochastic_lanczos",
]
