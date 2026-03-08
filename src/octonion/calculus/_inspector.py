"""ASCII tree renderer for parenthesization debugging.

Provides two representations:
1. Mathematical notation: "(x0 * (x1 * x2))"
2. ASCII art tree display with Unicode box-drawing characters

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from octonion.calculus._composition import Leaf, Node, TreeNode


# Operation symbols for mathematical notation
_OP_SYMBOLS = {
    "mul": "*",
    "exp": "exp",
    "log": "log",
    "conjugate": "conj",
    "inverse": "inv",
}


def tree_to_string(tree: TreeNode) -> str:
    """Convert tree to mathematical notation.

    Examples:
        Leaf(0) -> "x0"
        Node("mul", Leaf(0), Leaf(1)) -> "(x0 * x1)"
        Node("mul", Node("mul", Leaf(0), Leaf(1)), Leaf(2)) -> "((x0 * x1) * x2)"
        Node("exp", Leaf(0), None) -> "exp(x0)"

    Args:
        tree: Binary tree to convert.

    Returns:
        String representation in mathematical notation.
    """
    if isinstance(tree, Leaf):
        return f"x{tree.index}"

    sym = _OP_SYMBOLS.get(tree.op, tree.op)

    left_str = tree_to_string(tree.left)

    if tree.right is None:
        # Unary operation
        return f"{sym}({left_str})"
    else:
        # Binary operation
        right_str = tree_to_string(tree.right)
        return f"({left_str} {sym} {right_str})"


def inspect_tree(tree: TreeNode) -> str:
    """Render an ASCII art tree display.

    Uses Unicode box-drawing characters for clean output.

    Example for ((x0 * x1) * x2):
        *
        +-- *
        |   +-- x0
        |   +-- x1
        +-- x2

    Args:
        tree: Binary tree to render.

    Returns:
        Multi-line string with ASCII art tree.
    """
    lines: list[str] = []
    _render(tree, lines, prefix="", is_last=True, is_root=True)
    return "\n".join(lines)


def _render(
    tree: TreeNode,
    lines: list[str],
    prefix: str,
    is_last: bool,
    is_root: bool,
) -> None:
    """Recursively render tree nodes with box-drawing connectors."""
    if is_root:
        connector = ""
        child_prefix = ""
    else:
        connector = "+-- " if is_last else "+-- "
        child_prefix = prefix + ("    " if is_last else "|   ")

    if isinstance(tree, Leaf):
        lines.append(f"{prefix}{connector}x{tree.index}")
        return

    sym = _OP_SYMBOLS.get(tree.op, tree.op)
    lines.append(f"{prefix}{connector}{sym}")

    children: list[TreeNode] = [tree.left]
    if tree.right is not None:
        children.append(tree.right)

    for i, child in enumerate(children):
        is_child_last = i == len(children) - 1
        _render(child, lines, child_prefix, is_child_last, is_root=False)
