"""Optimization landscape analysis toolkit.

Provides Hessian eigenspectrum analysis and loss surface curvature
measurement for characterizing the optimization landscape of models
using octonionic, quaternionic, complex, and real-valued representations.
"""

from octonion.landscape._curvature import measure_curvature
from octonion.landscape._hessian import (
    compute_full_hessian,
    compute_hessian_spectrum,
    hessian_vector_product,
    stochastic_lanczos,
)

__all__ = [
    "compute_full_hessian",
    "compute_hessian_spectrum",
    "hessian_vector_product",
    "measure_curvature",
    "stochastic_lanczos",
]
