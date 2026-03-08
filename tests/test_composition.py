"""Tests for parenthesization-aware composition of octonionic operations.

Tests cover:
- Binary tree types and Catalan number generation (all_parenthesizations)
- Tree evaluation with autograd tracking (evaluate_tree)
- Non-associativity verification (different parenthesizations -> different results)
- Chain rule composition (compose_jacobians vs numeric Jacobian)
- Naive vs correct chain rule difference
- Parenthesization inspector (tree_to_string, inspect_tree)

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

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
