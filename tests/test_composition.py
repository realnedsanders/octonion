"""Tests for parenthesization-aware composition of octonionic operations.

Tests cover:
- Binary tree types and Catalan number generation (all_parenthesizations)
- Tree evaluation with autograd tracking (evaluate_tree)
- Non-associativity verification (different parenthesizations -> different results)
- Chain rule composition (compose_jacobians vs numeric Jacobian)
- Naive vs correct chain rule difference
- Parenthesization inspector (tree_to_string, inspect_tree)
- SC-2: Exhaustive parenthesization gradient check for 5-operand chains
- SC-3: Naive vs correct gradient demonstration with quantitative report

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

import json
import os

import torch

from octonion.calculus._composition import (
    CompositionBuilder,
    Leaf,
    Node,
    all_parenthesizations,
    evaluate_tree,
)
from octonion.calculus._chain_rule import (
    compose_jacobians,
    naive_chain_rule_jacobian,
)
from octonion.calculus._inspector import inspect_tree, tree_to_string
from octonion.calculus._numeric import numeric_jacobian


# ============================================================================
# TestBinaryTrees: Verify Catalan numbers and tree structure
# ============================================================================


class TestBinaryTrees:
    """Verify all_parenthesizations generates correct Catalan number of trees."""

    def test_one_operand(self) -> None:
        """all_parenthesizations(1) returns [Leaf(0)] -- 1 tree."""
        trees = all_parenthesizations(1)
        assert len(trees) == 1
        assert isinstance(trees[0], Leaf)
        assert trees[0].index == 0

    def test_two_operands(self) -> None:
        """all_parenthesizations(2) returns 1 tree: Node(mul, Leaf(0), Leaf(1))."""
        trees = all_parenthesizations(2)
        assert len(trees) == 1
        tree = trees[0]
        assert isinstance(tree, Node)
        assert tree.op == "mul"
        assert isinstance(tree.left, Leaf) and tree.left.index == 0
        assert isinstance(tree.right, Leaf) and tree.right.index == 1

    def test_three_operands_catalan_2(self) -> None:
        """all_parenthesizations(3) returns 2 trees -- C_2 = 2."""
        trees = all_parenthesizations(3)
        assert len(trees) == 2

    def test_four_operands_catalan_5(self) -> None:
        """all_parenthesizations(4) returns 5 trees -- C_3 = 5."""
        trees = all_parenthesizations(4)
        assert len(trees) == 5

    def test_five_operands_catalan_14(self) -> None:
        """all_parenthesizations(5) returns 14 trees -- C_4 = 14."""
        trees = all_parenthesizations(5)
        assert len(trees) == 14

    def test_tree_structures_unique(self) -> None:
        """All trees for n=4 should have unique string representations."""
        trees = all_parenthesizations(4)
        strings = [tree_to_string(t) for t in trees]
        assert len(set(strings)) == 5

    def test_leaf_indices_complete(self) -> None:
        """Each tree for n=4 references all operand indices 0..3."""
        trees = all_parenthesizations(4)
        for tree in trees:
            indices = _collect_leaf_indices(tree)
            assert sorted(indices) == [0, 1, 2, 3]


class TestEvaluateTree:
    """Verify evaluate_tree produces correct results with autograd tracking."""

    def test_single_leaf(self) -> None:
        """evaluate_tree on a Leaf returns the corresponding operand."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        result = evaluate_tree(Leaf(0), [x])
        assert torch.allclose(result, x)

    def test_two_operand_mul(self) -> None:
        """evaluate_tree(Node(mul, Leaf(0), Leaf(1))) matches octonion_mul."""
        from octonion import octonion_mul

        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)

        tree = Node("mul", Leaf(0), Leaf(1))
        result = evaluate_tree(tree, [a, b])
        expected = octonion_mul(a, b)
        assert torch.allclose(result, expected)

    def test_gradient_tracking(self) -> None:
        """evaluate_tree preserves autograd graph for backward pass."""
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)

        tree = Node("mul", Leaf(0), Leaf(1))
        result = evaluate_tree(tree, [a, b])
        loss = result.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_three_operand_left_assoc(self) -> None:
        """(a*b)*c via evaluate_tree matches manual computation."""
        from octonion import octonion_mul

        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(8, dtype=torch.float64)
        c = torch.randn(8, dtype=torch.float64)

        tree = Node("mul", Node("mul", Leaf(0), Leaf(1)), Leaf(2))
        result = evaluate_tree(tree, [a, b, c])
        expected = octonion_mul(octonion_mul(a, b), c)
        assert torch.allclose(result, expected)

    def test_three_operand_right_assoc(self) -> None:
        """a*(b*c) via evaluate_tree matches manual computation."""
        from octonion import octonion_mul

        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(8, dtype=torch.float64)
        c = torch.randn(8, dtype=torch.float64)

        tree = Node("mul", Leaf(0), Node("mul", Leaf(1), Leaf(2)))
        result = evaluate_tree(tree, [a, b, c])
        expected = octonion_mul(a, octonion_mul(b, c))
        assert torch.allclose(result, expected)


class TestNonAssociativity:
    """Verify that different parenthesizations produce different results."""

    def test_two_parenthesizations_differ(self) -> None:
        """(a*b)*c != a*(b*c) for generic octonions (non-associativity)."""
        torch.manual_seed(42)
        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(8, dtype=torch.float64)
        c = torch.randn(8, dtype=torch.float64)

        trees = all_parenthesizations(3)
        assert len(trees) == 2
        r1 = evaluate_tree(trees[0], [a, b, c])
        r2 = evaluate_tree(trees[1], [a, b, c])

        # Results should differ (non-associativity)
        assert not torch.allclose(r1, r2, atol=1e-10)

    def test_gradients_differ_by_parenthesization(self) -> None:
        """Gradients w.r.t. inputs differ between parenthesizations."""
        torch.manual_seed(42)
        a1 = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b1 = torch.randn(8, dtype=torch.float64, requires_grad=True)
        c1 = torch.randn(8, dtype=torch.float64, requires_grad=True)
        # Clone for second parenthesization
        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        c2 = c1.detach().clone().requires_grad_(True)

        trees = all_parenthesizations(3)
        r1 = evaluate_tree(trees[0], [a1, b1, c1])
        r2 = evaluate_tree(trees[1], [a2, b2, c2])

        r1.sum().backward()
        r2.sum().backward()

        # At least one gradient should differ
        grad_diff = torch.norm(a1.grad - a2.grad).item()
        assert grad_diff > 1e-10, f"Gradients too similar: diff={grad_diff}"


class TestChainRule:
    """Verify compose_jacobians matches numeric Jacobian."""

    def test_two_operand_chain(self) -> None:
        """compose_jacobians for a*b matches numeric Jacobian."""
        from octonion.calculus._numeric import numeric_jacobian

        torch.manual_seed(42)
        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(8, dtype=torch.float64)

        tree = Node("mul", Leaf(0), Leaf(1))
        jacobians = compose_jacobians(tree, [a, b])

        # Should have 2 Jacobians (one per operand)
        assert len(jacobians) == 2

        # Check J w.r.t. a
        def f_a(x: torch.Tensor) -> torch.Tensor:
            return evaluate_tree(tree, [x, b])

        J_numeric_a = numeric_jacobian(f_a, a)
        assert torch.allclose(jacobians[0], J_numeric_a, atol=1e-7)

        # Check J w.r.t. b
        def f_b(x: torch.Tensor) -> torch.Tensor:
            return evaluate_tree(tree, [a, x])

        J_numeric_b = numeric_jacobian(f_b, b)
        assert torch.allclose(jacobians[1], J_numeric_b, atol=1e-7)

    def test_three_operand_chain(self) -> None:
        """compose_jacobians for (a*b)*c matches numeric Jacobian."""
        from octonion.calculus._numeric import numeric_jacobian

        torch.manual_seed(42)
        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(8, dtype=torch.float64)
        c = torch.randn(8, dtype=torch.float64)

        tree = Node("mul", Node("mul", Leaf(0), Leaf(1)), Leaf(2))
        jacobians = compose_jacobians(tree, [a, b, c])
        assert len(jacobians) == 3

        # Check each J against numeric
        for idx, name in enumerate(["a", "b", "c"]):
            ops = [a, b, c]

            def f(x: torch.Tensor, i=idx) -> torch.Tensor:
                ops_copy = list(ops)
                ops_copy[i] = x
                return evaluate_tree(tree, ops_copy)

            J_numeric = numeric_jacobian(f, ops[idx])
            assert torch.allclose(
                jacobians[idx], J_numeric, atol=1e-6
            ), f"Jacobian w.r.t. {name} differs: max err={torch.abs(jacobians[idx] - J_numeric).max().item()}"


class TestNaiveVsCorrect:
    """Verify naive_chain_rule_jacobian differs from correct compose_jacobians."""

    def test_naive_differs_for_right_assoc(self) -> None:
        """Naive (left-to-right) chain rule != correct for right-associated tree."""
        torch.manual_seed(42)
        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(8, dtype=torch.float64)
        c = torch.randn(8, dtype=torch.float64)

        # Right-associated: a * (b * c)
        right_tree = Node("mul", Leaf(0), Node("mul", Leaf(1), Leaf(2)))
        correct_jacs = compose_jacobians(right_tree, [a, b, c])

        # Naive always assumes left-to-right: (a*b)*c
        naive_jacs = naive_chain_rule_jacobian([a, b, c])

        # At least one Jacobian should differ
        any_diff = False
        for i in range(3):
            diff = torch.norm(correct_jacs[i] - naive_jacs[i]).item()
            if diff > 1e-8:
                any_diff = True
                break

        assert any_diff, "Naive and correct Jacobians should differ for right-associated tree"


class TestInspector:
    """Verify tree_to_string and inspect_tree produce correct output."""

    def test_tree_to_string_leaf(self) -> None:
        """Leaf(0) -> 'x0'."""
        assert tree_to_string(Leaf(0)) == "x0"

    def test_tree_to_string_simple(self) -> None:
        """Node(mul, Leaf(0), Leaf(1)) -> '(x0 * x1)'."""
        tree = Node("mul", Leaf(0), Leaf(1))
        assert tree_to_string(tree) == "(x0 * x1)"

    def test_tree_to_string_nested(self) -> None:
        """Nested tree produces correct notation."""
        tree = Node("mul", Node("mul", Leaf(0), Leaf(1)), Leaf(2))
        result = tree_to_string(tree)
        assert result == "((x0 * x1) * x2)"

    def test_tree_to_string_right_assoc(self) -> None:
        """Right-associated tree distinguishable from left."""
        left = Node("mul", Node("mul", Leaf(0), Leaf(1)), Leaf(2))
        right = Node("mul", Leaf(0), Node("mul", Leaf(1), Leaf(2)))
        assert tree_to_string(left) != tree_to_string(right)
        assert tree_to_string(right) == "(x0 * (x1 * x2))"

    def test_inspect_tree_nonempty(self) -> None:
        """inspect_tree produces non-empty string."""
        tree = Node("mul", Node("mul", Leaf(0), Leaf(1)), Leaf(2))
        result = inspect_tree(tree)
        assert len(result) > 0
        assert "x0" in result
        assert "x1" in result
        assert "x2" in result

    def test_inspect_tree_leaf(self) -> None:
        """inspect_tree on a leaf returns the operand name."""
        result = inspect_tree(Leaf(0))
        assert "x0" in result


class TestCompositionBuilder:
    """Verify CompositionBuilder API."""

    def test_from_parenthesization(self) -> None:
        """from_parenthesization creates a builder with the correct tree."""
        builder = CompositionBuilder.from_parenthesization(3, 0)
        assert builder.tree is not None

    def test_evaluate(self) -> None:
        """Builder evaluate matches evaluate_tree."""
        torch.manual_seed(42)
        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(8, dtype=torch.float64)
        c = torch.randn(8, dtype=torch.float64)

        builder = CompositionBuilder.from_parenthesization(3, 0)
        result = builder.evaluate([a, b, c])
        expected = evaluate_tree(builder.tree, [a, b, c])
        assert torch.allclose(result, expected)

    def test_callable(self) -> None:
        """Builder is callable (via __call__)."""
        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(8, dtype=torch.float64)

        builder = CompositionBuilder.from_parenthesization(2, 0)
        result = builder([a, b])
        assert result.shape == (8,)

    def test_inspect(self) -> None:
        """Builder inspect returns non-empty string."""
        builder = CompositionBuilder.from_parenthesization(3, 0)
        result = builder.inspect()
        assert len(result) > 0


# ============================================================================
# SC-2: Exhaustive parenthesization gradient check for 5-operand chains
# ============================================================================


class TestParenthesizationExhaustive:
    """SC-2: Gradient check passes for all 14 Catalan(4) parenthesizations."""

    def test_parenthesization_exhaustive(self) -> None:
        """All 14 parenthesizations of 5 operands pass gradient check at 1e-5."""
        torch.manual_seed(42)
        trees = all_parenthesizations(5)
        assert len(trees) == 14

        report: list[dict] = []

        for tree_idx, tree in enumerate(trees):
            tree_str = tree_to_string(tree)

            # Generate 5 random float64 octonion tensors with small norm
            # Multiply BEFORE setting requires_grad to keep them as leaf tensors
            operands = [
                (torch.randn(8, dtype=torch.float64) * 0.5).requires_grad_(True)
                for _ in range(5)
            ]

            # Evaluate the tree
            result = evaluate_tree(tree, operands)

            # Compute scalar loss and backward
            loss = result.sum()
            loss.backward()

            # Collect autograd gradients
            autograd_grads = [op.grad.clone() for op in operands]

            # Compare against numeric Jacobian for each operand
            per_operand_errors: list[float] = []
            for op_idx in range(5):
                # Create function for numeric differentiation
                def f(x: torch.Tensor, idx=op_idx) -> torch.Tensor:
                    ops_copy = [
                        o.detach().clone() for o in operands
                    ]
                    ops_copy[idx] = x
                    return evaluate_tree(tree, ops_copy)

                J_numeric = numeric_jacobian(f, operands[op_idx].detach())
                # Autograd gradient = sum of rows of the Jacobian (since loss = sum)
                numeric_grad = J_numeric.sum(dim=0)

                rel_err = torch.abs(autograd_grads[op_idx] - numeric_grad) / (
                    torch.abs(numeric_grad) + 1e-15
                )
                max_rel = rel_err.max().item()
                per_operand_errors.append(max_rel)

            max_error = max(per_operand_errors)
            mean_error = sum(per_operand_errors) / len(per_operand_errors)
            report.append(
                {
                    "tree": tree_str,
                    "tree_idx": tree_idx,
                    "max_rel_error": max_error,
                    "mean_rel_error": mean_error,
                    "per_operand_errors": per_operand_errors,
                }
            )

            assert max_error < 1e-5, (
                f"Pattern {tree_idx} ({tree_str}) failed: "
                f"max_rel_error={max_error:.2e}"
            )

        # Compute max gradient difference between patterns
        all_grads = []
        for tree in trees:
            torch.manual_seed(42)  # Same operands for all patterns
            operands = [
                (torch.randn(8, dtype=torch.float64) * 0.5).requires_grad_(True)
                for _ in range(5)
            ]
            result = evaluate_tree(tree, operands)
            result.sum().backward()
            grad_vec = torch.cat([op.grad for op in operands])
            all_grads.append(grad_vec)

        max_diff = 0.0
        for i in range(len(all_grads)):
            for j in range(i + 1, len(all_grads)):
                diff = torch.norm(all_grads[i] - all_grads[j]).item()
                max_diff = max(max_diff, diff)

        # Save quantitative report
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
        )
        os.makedirs(results_dir, exist_ok=True)
        full_report = {
            "n_operands": 5,
            "n_parenthesizations": 14,
            "per_pattern": report,
            "max_gradient_difference_between_patterns": max_diff,
            "all_passed": all(r["max_rel_error"] < 1e-5 for r in report),
        }
        report_path = os.path.join(results_dir, "parenthesization_report.json")
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)

        # Print summary
        print(f"\n=== SC-2 Parenthesization Report ===")
        print(f"Patterns tested: {len(report)}")
        print(
            f"Max rel error across all: "
            f"{max(r['max_rel_error'] for r in report):.2e}"
        )
        print(f"Max gradient diff between patterns: {max_diff:.4f}")
        print(f"All passed: {full_report['all_passed']}")

    def test_parenthesization_autograd_gradcheck(self) -> None:
        """SC-2 supplement: torch.autograd.gradcheck passes for all 14 patterns."""
        torch.manual_seed(42)
        trees = all_parenthesizations(5)

        for tree_idx, tree in enumerate(trees):
            operands = tuple(
                (torch.randn(8, dtype=torch.float64) * 0.5).requires_grad_(True)
                for _ in range(5)
            )

            def f(*ops: torch.Tensor, t=tree) -> torch.Tensor:
                return evaluate_tree(t, list(ops))

            passed = torch.autograd.gradcheck(
                f, operands, eps=1e-6, atol=1e-5, rtol=1e-3
            )
            assert passed, (
                f"gradcheck failed for pattern {tree_idx}: "
                f"{tree_to_string(tree)}"
            )


# ============================================================================
# SC-3: Naive vs correct gradient demonstration
# ============================================================================


class TestNaiveVsCorrectDemonstration:
    """SC-3: Demonstrate naive vs correct gradient difference quantitatively."""

    def test_naive_vs_correct_differs(self) -> None:
        """Naive chain rule produces different gradients than correct implementation."""
        torch.manual_seed(42)

        differences: list[float] = []
        n_trials = 20

        for trial in range(n_trials):
            torch.manual_seed(trial)
            operands = [torch.randn(8, dtype=torch.float64) * 0.5 for _ in range(3)]

            # Right-associated tree: a * (b * c)
            right_tree = Node("mul", Leaf(0), Node("mul", Leaf(1), Leaf(2)))
            correct_jacs = compose_jacobians(right_tree, operands)

            # Naive: assumes left-to-right (a*b)*c
            naive_jacs = naive_chain_rule_jacobian(operands)

            # Compute gradient difference
            total_diff = sum(
                torch.norm(correct_jacs[i] - naive_jacs[i]).item()
                for i in range(3)
            )
            differences.append(total_diff)

        # The differences should be consistently non-zero
        mean_diff = sum(differences) / len(differences)
        assert mean_diff > 1e-3, (
            f"Mean gradient difference too small: {mean_diff:.2e}. "
            f"Expected substantial difference due to non-associativity."
        )

        # Report magnitude
        print(f"\n=== SC-3 Naive vs Correct ===")
        print(f"Trials: {n_trials}")
        print(f"Mean gradient difference: {mean_diff:.6f}")
        print(f"Min: {min(differences):.6f}, Max: {max(differences):.6f}")

    def test_naive_vs_correct_five_operands(self) -> None:
        """Demonstrate difference grows with depth for 5-operand chains."""
        torch.manual_seed(42)

        # Test multiple non-left-associated trees
        trees_5 = all_parenthesizations(5)
        # The last tree is the fully left-associated one (naive == correct), skip it
        left_tree_str = tree_to_string(trees_5[-1])

        differences_by_tree: list[tuple[str, float]] = []

        for tree in trees_5:
            if tree_to_string(tree) == left_tree_str:
                continue  # Skip the left-associated tree
            operands = [torch.randn(8, dtype=torch.float64) * 0.5 for _ in range(5)]
            correct_jacs = compose_jacobians(tree, operands)
            naive_jacs = naive_chain_rule_jacobian(operands)

            total_diff = sum(
                torch.norm(correct_jacs[i] - naive_jacs[i]).item()
                for i in range(5)
            )
            differences_by_tree.append((tree_to_string(tree), total_diff))

        # At least some should show substantial differences
        max_diff = max(d for _, d in differences_by_tree)
        assert max_diff > 1e-2, (
            f"Max gradient difference across trees too small: {max_diff:.2e}"
        )


# ============================================================================
# Mixed-operation compositions
# ============================================================================


class TestMixedOperations:
    """Test compositions with mixed operation types."""

    def test_mixed_op_exp_of_product(self) -> None:
        """exp(a * b) gradient check passes."""
        torch.manual_seed(42)
        a = (torch.randn(8, dtype=torch.float64) * 0.3).requires_grad_(True)
        b = (torch.randn(8, dtype=torch.float64) * 0.3).requires_grad_(True)

        # exp(a * b): unary op on binary result
        tree = Node("exp", Node("mul", Leaf(0), Leaf(1)), None)

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return evaluate_tree(tree, [x, y])

        passed = torch.autograd.gradcheck(
            f, (a, b), eps=1e-6, atol=1e-5, rtol=1e-3
        )
        assert passed

    def test_mixed_op_log_exp_product(self) -> None:
        """log(exp(a * b)) gradient check passes (should be near identity)."""
        torch.manual_seed(42)
        # Keep small to stay in principal branch
        a = (torch.randn(8, dtype=torch.float64) * 0.2).requires_grad_(True)
        b = (torch.randn(8, dtype=torch.float64) * 0.2).requires_grad_(True)

        # log(exp(a * b))
        tree = Node(
            "log",
            Node("exp", Node("mul", Leaf(0), Leaf(1)), None),
            None,
        )

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return evaluate_tree(tree, [x, y])

        passed = torch.autograd.gradcheck(
            f, (a, b), eps=1e-6, atol=1e-5, rtol=1e-3
        )
        assert passed

    def test_mixed_op_conjugate_product(self) -> None:
        """conjugate(a * b) gradient check passes."""
        torch.manual_seed(42)
        a = (torch.randn(8, dtype=torch.float64) * 0.5).requires_grad_(True)
        b = (torch.randn(8, dtype=torch.float64) * 0.5).requires_grad_(True)

        # conjugate(a * b)
        tree = Node("conjugate", Node("mul", Leaf(0), Leaf(1)), None)

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return evaluate_tree(tree, [x, y])

        passed = torch.autograd.gradcheck(
            f, (a, b), eps=1e-6, atol=1e-5, rtol=1e-3
        )
        assert passed

    def test_mixed_op_inverse_product(self) -> None:
        """inverse(a * b) gradient check passes."""
        torch.manual_seed(42)
        # Ensure non-zero product
        a = (torch.randn(8, dtype=torch.float64) * 0.5 + 0.5).requires_grad_(True)
        b = (torch.randn(8, dtype=torch.float64) * 0.5 + 0.5).requires_grad_(True)

        # inverse(a * b)
        tree = Node("inverse", Node("mul", Leaf(0), Leaf(1)), None)

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return evaluate_tree(tree, [x, y])

        passed = torch.autograd.gradcheck(
            f, (a, b), eps=1e-6, atol=1e-5, rtol=1e-3
        )
        assert passed

    def test_mixed_op_nested_chain(self) -> None:
        """Nested mixed: conjugate(a * b) * c gradient check passes."""
        torch.manual_seed(42)
        a = (torch.randn(8, dtype=torch.float64) * 0.3).requires_grad_(True)
        b = (torch.randn(8, dtype=torch.float64) * 0.3).requires_grad_(True)
        c = (torch.randn(8, dtype=torch.float64) * 0.3).requires_grad_(True)

        # conjugate(a * b) * c
        tree = Node(
            "mul",
            Node("conjugate", Node("mul", Leaf(0), Leaf(1)), None),
            Leaf(2),
        )

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return evaluate_tree(tree, [x, y, z])

        passed = torch.autograd.gradcheck(
            f, (a, b, c), eps=1e-6, atol=1e-5, rtol=1e-3
        )
        assert passed


# ============================================================================
# Helpers
# ============================================================================


def _collect_leaf_indices(tree: Leaf | Node) -> list[int]:
    """Recursively collect all leaf indices from a tree."""
    if isinstance(tree, Leaf):
        return [tree.index]
    result = _collect_leaf_indices(tree.left)
    if tree.right is not None:
        result.extend(_collect_leaf_indices(tree.right))
    return result
