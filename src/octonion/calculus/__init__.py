"""GHR calculus extension for octonionic differentiation.

This submodule implements the Generalized Hamilton-Real (GHR) calculus
formalism extended from quaternions to octonions, providing:

- **Wirtinger derivative pair** (df/do, df/do*): The fundamental derivative
  decomposition for octonionic functions, analogous to complex Wirtinger
  derivatives but extended to the 8-dimensional octonionic domain.

- **Analytic Jacobians**: Closed-form 8x8 real Jacobian matrices for all
  7 octonionic primitives (mul, exp, log, conjugate, inverse, inner_product,
  cross_product).

- **Numeric Jacobians**: Finite-difference central-difference Jacobian
  approximation for verification and testing.

The GHR formalism adds value over plain R^8 gradient computation by:
1. Providing derivatives in octonionic form (structured, not 8 separate reals)
2. Tracking both Wirtinger derivatives for non-real-valued intermediates
3. Enabling product and chain rules that respect octonionic structure

Usage::

    from octonion.calculus import ghr_derivative, conjugate_derivative
    from octonion.calculus import numeric_jacobian
    from octonion.calculus import jacobian_mul, jacobian_exp

References:
    - Xu & Mandic (2015), "The theory of quaternion matrix derivatives"
      (GHR calculus foundation for quaternions)
    - Baez (2002), "The Octonions" (algebraic structure)
    - This implementation: native octonionic derivation, NOT
      Parcollet et al. real-component reduction

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from octonion.calculus._ghr import (
    conjugate_derivative,
    ghr_derivative,
    wirtinger_from_jacobian,
)
from octonion.calculus._jacobians import (
    jacobian_conjugate,
    jacobian_cross_product,
    jacobian_exp,
    jacobian_inner_product,
    jacobian_inverse,
    jacobian_log,
    jacobian_mul,
)
from octonion.calculus._autograd_functions import (
    OctonionConjugateFunction,
    OctonionCrossProductFunction,
    OctonionExpFunction,
    OctonionInnerProductFunction,
    OctonionInverseFunction,
    OctonionLogFunction,
    OctonionMulFunction,
)
from octonion.calculus._gradcheck import octonion_gradcheck, octonion_gradgradcheck
from octonion.calculus._numeric import numeric_jacobian, numeric_jacobian_2arg

__all__ = [
    # GHR Wirtinger formalism
    "ghr_derivative",
    "conjugate_derivative",
    "wirtinger_from_jacobian",
    # Analytic Jacobians
    "jacobian_mul",
    "jacobian_exp",
    "jacobian_log",
    "jacobian_conjugate",
    "jacobian_inverse",
    "jacobian_inner_product",
    "jacobian_cross_product",
    # Autograd Functions
    "OctonionMulFunction",
    "OctonionExpFunction",
    "OctonionLogFunction",
    "OctonionConjugateFunction",
    "OctonionInverseFunction",
    "OctonionInnerProductFunction",
    "OctonionCrossProductFunction",
    # Gradient checking
    "octonion_gradcheck",
    "octonion_gradgradcheck",
    # Numeric Jacobians
    "numeric_jacobian",
    "numeric_jacobian_2arg",
]
