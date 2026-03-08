"""Parenthesization-aware chain rule for octonionic compositions.

Computes the Jacobian of composed octonionic operations by traversing
the computation tree bottom-up. At each internal node, the chain rule
composes the node's operation Jacobian with its subtree Jacobians.

Key insight: While octonion multiplication is non-associative, matrix
multiplication of 8x8 Jacobians IS associative. The chain rule application
is standard -- only the tree structure (which determines which operations
contribute to the Jacobian) matters.

The "naive" chain rule always assumes left-to-right association regardless
of the actual tree structure, demonstrating that parenthesization matters
for octonionic gradients.

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

import torch

from octonion.calculus._composition import Leaf, Node, TreeNode, evaluate_tree
from octonion.calculus._jacobians import jacobian_mul


def compose_jacobians(
    tree: TreeNode, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Compute the Jacobian of the full composition w.r.t. each operand.

    Traverses the tree bottom-up. At each internal node with operation "mul":
      J_total_wrt_x = J_mul_wrt_a @ J_left_wrt_x + J_mul_wrt_b @ J_right_wrt_x

    For a leaf node, the Jacobian w.r.t. its own operand is I (identity)
    and w.r.t. all other operands is 0.

    Args:
        tree: Binary tree specifying the computation structure.
        operands: List of [..., 8] octonion tensors.

    Returns:
        List of n tensors, each [8, 8], representing d(output)/d(operand_i).
    """
    n = len(operands)

    def _compose(subtree: TreeNode) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns (value, [J_wrt_op0, J_wrt_op1, ..., J_wrt_op_{n-1}]).

        value: The evaluation of this subtree.
        J_wrt_op_i: 8x8 Jacobian of this subtree's output w.r.t. operand i.
        """
        if isinstance(subtree, Leaf):
            val = operands[subtree.index]
            jacs = []
            eye = torch.eye(8, dtype=val.dtype, device=val.device)
            zero = torch.zeros(8, 8, dtype=val.dtype, device=val.device)
            for i in range(n):
                jacs.append(eye if i == subtree.index else zero)
            return val, jacs

        # Recursive case: Node
        left_val, left_jacs = _compose(subtree.left)

        if subtree.right is not None:
            right_val, right_jacs = _compose(subtree.right)
        else:
            # Unary operations not yet implemented in chain rule
            raise NotImplementedError(
                f"Chain rule for unary operation '{subtree.op}' not yet implemented"
            )

        if subtree.op == "mul":
            # Get Jacobians of mul w.r.t. its two arguments
            J_a, J_b = jacobian_mul(left_val, right_val)

            # Compute composed Jacobians:
            # J_total_wrt_x = J_mul_wrt_a @ J_left_wrt_x + J_mul_wrt_b @ J_right_wrt_x
            val = evaluate_tree(subtree, operands)
            composed_jacs = []
            for i in range(n):
                j_i = J_a @ left_jacs[i] + J_b @ right_jacs[i]
                composed_jacs.append(j_i)

            return val, composed_jacs
        else:
            raise NotImplementedError(
                f"Chain rule for operation '{subtree.op}' not yet implemented"
            )

    _, jacobians = _compose(tree)
    return jacobians


def naive_chain_rule_jacobian(
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute Jacobians assuming left-to-right association.

    Always parenthesizes as ((o1 * o2) * o3) * o4 * ... regardless of
    the actual tree structure. This is the "naive" baseline that ignores
    parenthesization.

    Args:
        operands: List of [..., 8] octonion tensors.

    Returns:
        List of n Jacobians [8, 8], each d(output)/d(operand_i).
    """
    n = len(operands)
    if n < 2:
        raise ValueError("Need at least 2 operands for chain rule")

    # Build left-to-right tree: ((o0 * o1) * o2) * ...
    tree: Leaf | Node = Leaf(0)
    for i in range(1, n):
        tree = Node("mul", tree, Leaf(i))

    # Reindex -- the above tree already has correct indices since we
    # build it sequentially
    return compose_jacobians(tree, operands)
