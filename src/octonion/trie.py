"""Self-organizing octonionic trie.

A hierarchical memory structure where routing uses Fano plane subalgebra
decomposition, growth is triggered by associator incompatibility, and
updates use octonionic composition. No gradient computation required.

Each node stores:
  - routing_key: fixed at creation, determines how inputs route through this node
  - content: accumulated via composition, represents the node's knowledge
  - category_counts: tracks which categories have been routed here (for evaluation)

Example:
    >>> from octonion.trie import OctonionTrie
    >>> trie = OctonionTrie(associator_threshold=0.3)
    >>> trie.insert(some_octonion, category=0)
    >>> leaf = trie.query(some_octonion)
    >>> leaf.dominant_category
    0
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from octonion._fano import FANO_PLANE
from octonion._multiplication import octonion_mul
from octonion._octonion import Octonion, associator


def subalgebra_activation(x: torch.Tensor) -> torch.Tensor:
    """Compute activation strength for each of the 7 Fano plane subalgebras.

    Each subalgebra is defined by a triple (e_i, e_j, e_k) of imaginary
    basis units. The activation is the norm of the projection onto those
    three components.

    Args:
        x: Octonion tensor of shape [..., 8].

    Returns:
        Tensor of shape [..., 7] with activation norms.
    """
    activations = []
    for triple in FANO_PLANE.triples:
        i, j, k = triple
        components = torch.stack([x[..., i], x[..., j], x[..., k]], dim=-1)
        activations.append(torch.linalg.norm(components, dim=-1))
    return torch.stack(activations, dim=-1)


@dataclass
class TrieNode:
    """A node in the octonionic trie."""

    routing_key: torch.Tensor
    content: torch.Tensor
    children: dict[int, TrieNode] = field(default_factory=dict)
    subalgebra_idx: int | None = None
    insert_count: int = 0
    category_counts: dict[int, int] = field(default_factory=dict)
    depth: int = 0
    buffer: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def dominant_category(self) -> int | None:
        """Category with the most inserts at this node."""
        if not self.category_counts:
            return None
        return max(self.category_counts, key=lambda k: self.category_counts[k])

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class OctonionTrie:
    """Self-organizing octonionic trie.

    Args:
        associator_threshold: Maximum associator norm for routing compatibility.
            Lower values produce more branching (finer discrimination).
        similarity_threshold: Minimum inner product for rumination acceptance.
        max_depth: Maximum trie depth.
        seed: Random seed for root routing key initialization.
        dtype: Tensor dtype (float64 recommended for algebraic precision).
    """

    def __init__(
        self,
        associator_threshold: float = 0.3,
        similarity_threshold: float = 0.1,
        max_depth: int = 15,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ):
        gen = torch.Generator().manual_seed(seed)
        root_key = torch.randn(8, dtype=dtype, generator=gen)
        root_key = root_key / root_key.norm()

        self.root = TrieNode(routing_key=root_key, content=root_key.clone())
        self.assoc_threshold = associator_threshold
        self.sim_threshold = similarity_threshold
        self.max_depth = max_depth
        self.dtype = dtype
        self.n_nodes = 1
        self.total_inserts = 0
        self.rumination_rejections = 0
        self.consolidation_merges = 0

    def _find_best_child(
        self, node: TrieNode, x: torch.Tensor
    ) -> tuple[int, TrieNode | None, float]:
        """Find the best child for input x at this node.

        Among existing children, selects the one most similar to x
        (highest inner product with routing key), filtered by associator
        compatibility. If no compatible child exists, returns the best
        unoccupied subalgebra slot for new child creation.

        Returns:
            (subalgebra_idx, child_or_None, associator_norm)
        """
        x_oct = Octonion(x)
        node_oct = Octonion(node.routing_key)

        best_compatible: tuple[int, TrieNode, float, float] | None = None

        for sub_idx, child in node.children.items():
            sim = torch.dot(x, child.routing_key).item()
            child_oct = Octonion(child.routing_key)
            assoc = associator(x_oct, child_oct, node_oct)
            assoc_norm = assoc.components.norm().item()

            if assoc_norm < self.assoc_threshold:
                if best_compatible is None or sim > best_compatible[3]:
                    best_compatible = (sub_idx, child, assoc_norm, sim)

        if best_compatible is not None:
            return best_compatible[0], best_compatible[1], best_compatible[2]

        # No compatible child: find best unoccupied subalgebra
        product = octonion_mul(
            node.routing_key.unsqueeze(0), x.unsqueeze(0)
        ).squeeze(0)
        activations = subalgebra_activation(product)
        ranked = activations.argsort(descending=True)

        for sub_idx in ranked:
            idx = sub_idx.item()
            if idx not in node.children:
                return idx, None, float("inf")

        # All 7 occupied, all incompatible: return most similar
        best_sim_idx = max(
            node.children.keys(),
            key=lambda k: torch.dot(x, node.children[k].routing_key).item(),
        )
        return best_sim_idx, node.children[best_sim_idx], self.assoc_threshold + 1

    def _ruminate(self, node: TrieNode, x: torch.Tensor) -> bool:
        """Geometric consistency check: is x similar to this node's history?"""
        if len(node.buffer) < 3:
            return True
        key_sim = torch.dot(x, node.routing_key).item()
        if key_sim < self.sim_threshold * 0.5:
            return False
        sims = [torch.dot(x, buf_x).item() for buf_x, _ in node.buffer]
        return sum(sims) / len(sims) > self.sim_threshold * 0.3

    def insert(self, x: torch.Tensor, category: int | None = None) -> TrieNode:
        """Insert an octonion into the trie, returning the destination node."""
        x = x.to(self.dtype)
        norm = x.norm()
        if norm > 0:
            x = x / norm

        self.total_inserts += 1
        node = self.root
        self._count(node, category)

        for _ in range(self.max_depth):
            if not node.children:
                sub_idx, _, _ = self._find_best_child(node, x)
                return self._create_child(node, x, sub_idx, category)

            sub_idx, child, assoc_norm = self._find_best_child(node, x)

            if child is None:
                return self._create_child(node, x, sub_idx, category)

            if assoc_norm < self.assoc_threshold and self._ruminate(child, x):
                node = child
                self._count(node, category)
                self._compose(node, x)
                node.buffer.append((x.clone(), category))
                continue

            if assoc_norm >= self.assoc_threshold:
                self.rumination_rejections += int(assoc_norm < self.assoc_threshold)
                # Find unoccupied slot
                product = octonion_mul(
                    node.routing_key.unsqueeze(0), x.unsqueeze(0)
                ).squeeze(0)
                activations = subalgebra_activation(product)
                for alt in activations.argsort(descending=True):
                    alt_idx = alt.item()
                    if alt_idx not in node.children:
                        return self._create_child(node, x, alt_idx, category)
                # All occupied: descend into best
                node = child
                self._count(node, category)
                continue
            else:
                # Rumination rejected
                self.rumination_rejections += 1
                product = octonion_mul(
                    node.routing_key.unsqueeze(0), x.unsqueeze(0)
                ).squeeze(0)
                activations = subalgebra_activation(product)
                for alt in activations.argsort(descending=True):
                    alt_idx = alt.item()
                    if alt_idx != sub_idx and alt_idx not in node.children:
                        return self._create_child(node, x, alt_idx, category)
                node = child
                self._count(node, category)
                continue

        self._compose(node, x)
        node.buffer.append((x.clone(), category))
        return node

    def query(self, x: torch.Tensor) -> TrieNode:
        """Route x through the trie without modification."""
        x = x.to(self.dtype)
        norm = x.norm()
        if norm > 0:
            x = x / norm

        node = self.root
        for _ in range(self.max_depth):
            if not node.children:
                return node
            _, child, _ = self._find_best_child(node, x)
            if child is None:
                return node
            node = child
        return node

    def consolidate(self) -> None:
        """Merge underused nodes into siblings."""
        self._consolidate_node(self.root)

    def stats(self) -> dict:
        """Compute trie statistics."""
        nodes: list[TrieNode] = []
        leaves: list[TrieNode] = []
        max_depth = 0

        def _walk(n: TrieNode) -> None:
            nonlocal max_depth
            nodes.append(n)
            max_depth = max(max_depth, n.depth)
            if n.is_leaf:
                leaves.append(n)
            for c in n.children.values():
                _walk(c)

        _walk(self.root)
        return {
            "n_nodes": len(nodes),
            "n_leaves": len(leaves),
            "max_depth": max_depth,
            "rumination_rejections": self.rumination_rejections,
            "consolidation_merges": self.consolidation_merges,
        }

    # ── Private helpers ──────────────────────────────────────────────

    def _count(self, node: TrieNode, category: int | None) -> None:
        node.insert_count += 1
        if category is not None:
            node.category_counts[category] = node.category_counts.get(category, 0) + 1

    def _create_child(
        self, parent: TrieNode, x: torch.Tensor, sub_idx: int, category: int | None
    ) -> TrieNode:
        child = TrieNode(
            routing_key=x.clone(),
            content=x.clone(),
            subalgebra_idx=sub_idx,
            depth=parent.depth + 1,
            buffer=deque(maxlen=30),
        )
        parent.children[sub_idx] = child
        self.n_nodes += 1
        self._count(child, category)
        child.buffer.append((x.clone(), category))
        return child

    def _compose(self, node: TrieNode, x: torch.Tensor) -> None:
        node.content = octonion_mul(
            node.content.unsqueeze(0), x.unsqueeze(0)
        ).squeeze(0)
        norm = node.content.norm()
        if norm > 0:
            node.content = node.content / norm

    def _consolidate_node(self, node: TrieNode) -> None:
        if not node.children:
            return
        for child in list(node.children.values()):
            self._consolidate_node(child)

        total = sum(c.insert_count for c in node.children.values())
        if total == 0 or len(node.children) < 2:
            return

        to_remove = [
            idx
            for idx, child in node.children.items()
            if child.insert_count / max(total, 1) < 0.05 and child.insert_count < 3
        ]
        if not to_remove or len(node.children) - len(to_remove) < 1:
            return

        surviving = {k: v for k, v in node.children.items() if k not in to_remove}
        absorber = surviving[max(surviving, key=lambda k: surviving[k].insert_count)]

        for idx in to_remove:
            removed = node.children.pop(idx)
            for cat, count in removed.category_counts.items():
                absorber.category_counts[cat] = absorber.category_counts.get(cat, 0) + count
            absorber.insert_count += removed.insert_count
            self.n_nodes -= 1
            self.consolidation_merges += 1
