"""Parenthesization-aware composition of octonionic operations.

Provides binary tree types for representing different parenthesizations of
n-ary multiplications, Catalan number enumeration of all possible
parenthesizations, and autograd-tracked evaluation of those trees.

Key insight: octonion multiplication is non-associative, so (a*b)*c != a*(b*c).
Different parenthesizations define different mathematical functions with
different gradients. This module provides the infrastructure to enumerate
and evaluate all possible parenthesizations.

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from octonion.calculus._autograd_functions import (
    OctonionConjugateFunction,
    OctonionExpFunction,
    OctonionInverseFunction,
    OctonionLogFunction,
    OctonionMulFunction,
)

# Type alias for the tree structure
TreeNode = Union["Leaf", "Node"]


@dataclass(frozen=True)
class Leaf:
    """A leaf node representing an operand by its index (0-based)."""

    index: int


@dataclass(frozen=True)
class Node:
    """An internal node representing an operation applied to subtrees.

    For binary ops (mul): left and right are both subtrees.
    For unary ops (exp, log, conjugate, inverse): left is the subtree, right is None.
    """

    op: str  # "mul", "exp", "log", "conjugate", "inverse"
    left: TreeNode
    right: TreeNode | None  # None for unary ops


def _reindex(tree: TreeNode, offset: int) -> TreeNode:
    """Shift all leaf indices in a tree by the given offset."""
    if isinstance(tree, Leaf):
        return Leaf(tree.index + offset)
    new_left = _reindex(tree.left, offset)
    new_right = _reindex(tree.right, offset) if tree.right is not None else None
    return Node(tree.op, new_left, new_right)


def all_parenthesizations(n: int, op: str = "mul") -> list[TreeNode]:
    """Generate all C_{n-1} binary tree structures for n operands.

    For n operands in a chain of binary operations, there are C_{n-1}
    distinct parenthesizations where C_k is the k-th Catalan number.

    Catalan numbers: C_0=1, C_1=1, C_2=2, C_3=5, C_4=14, C_5=42, ...

    Args:
        n: Number of operands (>= 1).
        op: Operation label for internal nodes (default "mul").

    Returns:
        List of tree structures, each a Leaf (for n=1) or Node.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if n == 1:
        return [Leaf(0)]

    results: list[TreeNode] = []
    for split in range(1, n):
        left_trees = all_parenthesizations(split, op)
        right_trees = all_parenthesizations(n - split, op)
        for lt in left_trees:
            for rt in right_trees:
                # Reindex the right subtree so indices start at `split`
                rt_reindexed = _reindex(rt, split)
                results.append(Node(op, lt, rt_reindexed))

    return results


# Map from operation names to their autograd Function apply methods
_OP_DISPATCH = {
    "mul": lambda a, b: OctonionMulFunction.apply(a, b),
    "exp": lambda a, _: OctonionExpFunction.apply(a),
    "log": lambda a, _: OctonionLogFunction.apply(a),
    "conjugate": lambda a, _: OctonionConjugateFunction.apply(a),
    "inverse": lambda a, _: OctonionInverseFunction.apply(a),
}


def evaluate_tree(tree: TreeNode, operands: list[torch.Tensor]) -> torch.Tensor:
    """Recursively evaluate a binary tree on the given operands.

    Each leaf reads from the operands list by index. Each internal node
    applies its operation using the corresponding autograd Function, ensuring
    correct parenthesization of the computation graph for backward passes.

    Args:
        tree: Binary tree specifying the computation structure.
        operands: List of [..., 8] octonion tensors.

    Returns:
        The evaluation result as a [..., 8] tensor.
    """
    if isinstance(tree, Leaf):
        return operands[tree.index]

    left_val = evaluate_tree(tree.left, operands)

    right_val = evaluate_tree(tree.right, operands) if tree.right is not None else None

    dispatch = _OP_DISPATCH.get(tree.op)
    if dispatch is None:
        raise ValueError(f"Unknown operation: {tree.op}")

    return dispatch(left_val, right_val)


def build_mixed_tree(ops: list[str], structure: TreeNode) -> TreeNode:
    """Replace operation labels in a tree structure with the given sequence.

    Traverses the tree in left-to-right order and assigns operations from
    the ops list to each internal node.

    Args:
        ops: List of operation names, one per internal node.
        structure: Tree structure to relabel.

    Returns:
        New tree with operations from ops assigned to internal nodes.

    Raises:
        ValueError: If len(ops) != number of internal nodes.
    """
    op_iter = iter(ops)

    def _relabel(tree: TreeNode) -> TreeNode:
        if isinstance(tree, Leaf):
            return tree
        new_left = _relabel(tree.left)
        new_right = _relabel(tree.right) if tree.right is not None else None
        try:
            new_op = next(op_iter)
        except StopIteration:
            raise ValueError("Not enough operations for the tree structure")
        return Node(new_op, new_left, new_right)

    result = _relabel(structure)

    # Check no ops left over
    remaining = list(op_iter)
    if remaining:
        raise ValueError(
            f"Too many operations: {len(ops)} ops for tree with "
            f"{len(ops) - len(remaining)} internal nodes"
        )

    return result


class CompositionBuilder:
    """Build and evaluate parenthesized octonionic compositions.

    Wraps a binary tree structure and provides convenience methods for
    evaluation, inspection, and gradient computation.
    """

    def __init__(self, tree: TreeNode) -> None:
        self.tree = tree

    @classmethod
    def from_parenthesization(cls, n: int, index: int) -> CompositionBuilder:
        """Create from the index-th parenthesization of n operands.

        Args:
            n: Number of operands.
            index: Index into the list of all parenthesizations (0-based).

        Returns:
            CompositionBuilder with the selected tree structure.
        """
        trees = all_parenthesizations(n)
        if index < 0 or index >= len(trees):
            raise IndexError(
                f"Parenthesization index {index} out of range [0, {len(trees)})"
            )
        return cls(trees[index])

    def evaluate(self, operands: list[torch.Tensor]) -> torch.Tensor:
        """Evaluate the composition tree on the given operands."""
        return evaluate_tree(self.tree, operands)

    def __call__(self, operands: list[torch.Tensor]) -> torch.Tensor:
        """Evaluate the composition (callable interface)."""
        return self.evaluate(operands)

    def inspect(self) -> str:
        """Return ASCII tree representation."""
        from octonion.calculus._inspector import inspect_tree

        return inspect_tree(self.tree)
