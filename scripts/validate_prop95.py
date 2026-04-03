"""Numerical validation of Proposition 9.5 (subalgebra proximity bound).

Tests whether ‖[a,b,c]‖ = O(ε²) or O(ε) when a,b,c are within ε of a
quaternionic subalgebra. Measures the empirical scaling exponent.
"""

import torch
from octonion import Octonion
from octonion._octonion import associator


def make_near_subalgebra(
    parallel_coeffs: torch.Tensor,
    perp_coeffs: torch.Tensor,
    epsilon: float,
) -> Octonion:
    """Create a unit octonion within ε of subalgebra span{1, e1, e2, e4}.

    parallel_coeffs: 4 coefficients for {1, e1, e2, e4} components
    perp_coeffs: 4 coefficients for {e3, e5, e6, e7} components (will be scaled by ε)
    """
    # Subalgebra span{1, e1, e2, e4} -> indices 0, 1, 2, 4
    # Perpendicular: indices 3, 5, 6, 7
    data = torch.zeros(8, dtype=torch.float64)
    par_indices = [0, 1, 2, 4]
    perp_indices = [3, 5, 6, 7]

    for i, idx in enumerate(par_indices):
        data[idx] = parallel_coeffs[i]
    for i, idx in enumerate(perp_indices):
        data[idx] = epsilon * perp_coeffs[i]

    # Normalize to unit octonion
    data = data / data.norm()
    return Octonion(data)


def measure_scaling():
    """Measure how ‖[a,b,c]‖ scales with ε."""
    torch.manual_seed(42)

    # Random parallel and perpendicular components
    a_par = torch.randn(4, dtype=torch.float64)
    a_par = a_par / a_par.norm()
    a_perp = torch.randn(4, dtype=torch.float64)
    a_perp = a_perp / a_perp.norm()

    b_par = torch.randn(4, dtype=torch.float64)
    b_par = b_par / b_par.norm()
    b_perp = torch.randn(4, dtype=torch.float64)
    b_perp = b_perp / b_perp.norm()

    c_par = torch.randn(4, dtype=torch.float64)
    c_par = c_par / c_par.norm()
    c_perp = torch.randn(4, dtype=torch.float64)
    c_perp = c_perp / c_perp.norm()

    epsilons = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    norms = []

    print(f"{'ε':>10}  {'‖[a,b,c]‖':>14}  {'ratio to prev':>14}  {'log ratio':>10}")
    print("-" * 55)

    prev_norm = None
    prev_eps = None
    for eps in epsilons:
        a = make_near_subalgebra(a_par, a_perp, eps)
        b = make_near_subalgebra(b_par, b_perp, eps)
        c = make_near_subalgebra(c_par, c_perp, eps)

        assoc = associator(a, b, c)
        norm = assoc._data.norm().item()
        norms.append(norm)

        if prev_norm is not None and norm > 0 and prev_norm > 0:
            eps_ratio = prev_eps / eps
            norm_ratio = prev_norm / norm
            # If O(ε^α), then norm_ratio ≈ eps_ratio^α
            import math

            alpha = math.log(norm_ratio) / math.log(eps_ratio)
            print(f"{eps:>10.4f}  {norm:>14.8e}  {norm_ratio:>14.4f}  α≈{alpha:>7.3f}")
        else:
            print(f"{eps:>10.4f}  {norm:>14.8e}")

        prev_norm = norm
        prev_eps = eps

    print()
    print("If O(ε²), α should ≈ 2.0")
    print("If O(ε),  α should ≈ 1.0")

    # Also: explicit counterexample with basis elements
    print("\n--- Explicit counterexample ---")
    print("Subalgebra S = span{1, e1, e2, e4}")
    print("a_∥ = e1, b_∥ = e2, c_⊥ = e3")

    e1 = Octonion(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
    e2 = Octonion(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float64))
    e3 = Octonion(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float64))

    result = associator(e1, e2, e3)
    print(f"[e1, e2, e3] = {result._data}")
    print(f"‖[e1, e2, e3]‖ = {result._data.norm().item()}")
    print("This is nonzero => terms with exactly 1 perp component do NOT vanish.")


def measure_scaling_same_element():
    """Test O(ε) vs O(ε²) when a,b,c are near the SAME element q (not just same subalgebra)."""
    torch.manual_seed(42)

    # Fixed base element in the subalgebra
    q_coeffs = torch.randn(4, dtype=torch.float64)
    q_coeffs = q_coeffs / q_coeffs.norm()
    q = torch.zeros(8, dtype=torch.float64)
    q[0], q[1], q[2], q[4] = q_coeffs[0], q_coeffs[1], q_coeffs[2], q_coeffs[3]

    # Three different perturbation directions (generic, not confined to subalgebra)
    da = torch.randn(8, dtype=torch.float64)
    da = da / da.norm()
    db = torch.randn(8, dtype=torch.float64)
    db = db / db.norm()
    dc = torch.randn(8, dtype=torch.float64)
    dc = dc / dc.norm()

    epsilons = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    print("\n=== Near same ELEMENT (a ≈ b ≈ c ≈ q) ===")
    print(f"{'ε':>10}  {'‖[a,b,c]‖':>14}  {'ratio to prev':>14}  {'log ratio':>10}")
    print("-" * 55)

    prev_norm = None
    prev_eps = None
    for eps in epsilons:
        a = Octonion((q + eps * da) / (q + eps * da).norm())
        b = Octonion((q + eps * db) / (q + eps * db).norm())
        c = Octonion((q + eps * dc) / (q + eps * dc).norm())

        assoc = associator(a, b, c)
        norm = assoc._data.norm().item()

        if prev_norm is not None and norm > 0 and prev_norm > 0:
            import math
            eps_ratio = prev_eps / eps
            norm_ratio = prev_norm / norm
            alpha = math.log(norm_ratio) / math.log(eps_ratio)
            print(f"{eps:>10.4f}  {norm:>14.8e}  {norm_ratio:>14.4f}  α≈{alpha:>7.3f}")
        else:
            print(f"{eps:>10.4f}  {norm:>14.8e}")

        prev_norm = norm
        prev_eps = eps

    print()
    print("If O(ε²), α should ≈ 2.0")
    print("If O(ε),  α should ≈ 1.0")
    print()
    print("WHY: when a ≈ b ≈ c ≈ q, the single-⊥ terms become [δa, q, q],")
    print("[q, δb, q], [q, q, δc] — all zero by alternativity/flexibility!")


if __name__ == "__main__":
    measure_scaling()
    measure_scaling_same_element()
