"""Octonionic Trie Prototype: Stability-Plasticity Test.

Usage:
    docker compose run --rm dev uv run python scripts/run_trie_prototype.py
    docker compose run --rm dev uv run python scripts/run_trie_prototype.py --sweep
    docker compose run --rm dev uv run python scripts/run_trie_prototype.py --epochs 10 --train-per-cat 200
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch

from octonion import FANO_PLANE, associator, octonion_mul
from octonion._octonion import Octonion

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def subalgebra_activation(x: torch.Tensor) -> torch.Tensor:
    """Activation strength for each of the 7 Fano plane subalgebras."""
    activations = []
    for triple in FANO_PLANE.triples:
        i, j, k = triple
        components = torch.stack([x[..., i], x[..., j], x[..., k]], dim=-1)
        activations.append(torch.linalg.norm(components, dim=-1))
    return torch.stack(activations, dim=-1)


def oct_inner(a: torch.Tensor, b: torch.Tensor) -> float:
    """Octonionic inner product Re(conj(a) * b) = dot product as R^8 vectors."""
    return torch.dot(a, b).item()


# ── Trie ─────────────────────────────────────────────────────────────


@dataclass
class TrieNode:
    routing_key: torch.Tensor   # Fixed at creation, determines routing
    content: torch.Tensor       # Accumulated via composition
    children: dict[int, TrieNode] = field(default_factory=dict)
    subalgebra_idx: int | None = None
    insert_count: int = 0
    category_counts: dict[int, int] = field(default_factory=dict)
    depth: int = 0
    buffer: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def dominant_category(self) -> int | None:
        if not self.category_counts:
            return None
        return max(self.category_counts, key=lambda k: self.category_counts[k])

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class OctonionTrie:

    def __init__(
        self,
        associator_threshold: float = 0.3,
        similarity_threshold: float = 0.5,
        max_depth: int = 10,
        consolidation_interval: int = 100,
        dtype: torch.dtype = torch.float64,
        seed: int = 0,
    ):
        gen = torch.Generator().manual_seed(seed)
        root_key = torch.randn(8, dtype=dtype, generator=gen)
        root_key = root_key / root_key.norm()

        self.root = TrieNode(routing_key=root_key, content=root_key.clone())
        self.assoc_threshold = associator_threshold
        self.sim_threshold = similarity_threshold
        self.max_depth = max_depth
        self.consolidation_interval = consolidation_interval
        self.dtype = dtype
        self.n_nodes = 1
        self.total_inserts = 0
        self.rumination_rejections = 0
        self.consolidation_merges = 0

    # ── Routing ──────────────────────────────────────────────────────

    def _find_best_child(self, node: TrieNode, x: torch.Tensor) -> tuple[int, TrieNode | None, float]:
        """Find the best child for input x at this node.

        Returns (subalgebra_idx, child_or_None, associator_norm).

        Routing strategy:
        1. Among existing children, find the one most SIMILAR to x
           (highest inner product with routing key), filtered by
           associator compatibility.
        2. If no compatible child exists, return the best unoccupied
           subalgebra slot (for new child creation).
        3. Subalgebra activation determines WHERE to place new children;
           similarity determines WHICH existing child to route to.
        """
        x_oct = Octonion(x)
        node_oct = Octonion(node.routing_key)

        # Score all existing children by similarity + compatibility
        # Use CONTENT for similarity (adapts as node absorbs data)
        best_compatible = None  # (sub_idx, child, assoc_norm, similarity)

        for sub_idx, child in node.children.items():
            # Similarity: inner product between input and child's routing key (fixed)
            sim = oct_inner(x, child.routing_key)

            # Compatibility: associator check
            child_oct = Octonion(child.routing_key)
            assoc = associator(x_oct, child_oct, node_oct)
            assoc_norm = assoc.components.norm().item()

            if assoc_norm < self.assoc_threshold:
                if best_compatible is None or sim > best_compatible[3]:
                    best_compatible = (sub_idx, child, assoc_norm, sim)

        if best_compatible is not None:
            return best_compatible[0], best_compatible[1], best_compatible[2]

        # No compatible child: find best unoccupied subalgebra for a new one
        product = octonion_mul(
            node.routing_key.unsqueeze(0), x.unsqueeze(0)
        ).squeeze(0)
        activations = subalgebra_activation(product)
        ranked = activations.argsort(descending=True)

        for sub_idx in ranked:
            sub_idx = sub_idx.item()
            if sub_idx not in node.children:
                return sub_idx, None, float("inf")

        # All 7 occupied, all incompatible: return most similar regardless
        best_sim = None
        for sub_idx, child in node.children.items():
            sim = oct_inner(x, child.routing_key)
            if best_sim is None or sim > best_sim[1]:
                best_sim = (sub_idx, sim)
        return best_sim[0], node.children[best_sim[0]], self.assoc_threshold + 1

    # ── Rumination ───────────────────────────────────────────────────

    def _ruminate(self, node: TrieNode, x: torch.Tensor) -> bool:
        """Geometric consistency check: is x similar to what this node has seen?

        Computes mean inner product between x and the node's buffered inputs.
        If x is geometrically dissimilar to the buffer, reject it.
        Also checks similarity to the routing key (the node's prototype).
        """
        # Always accept if the buffer is sparse
        if len(node.buffer) < 3:
            return True

        # Check similarity to routing key (prototype)
        key_sim = oct_inner(x, node.routing_key)
        if key_sim < self.sim_threshold * 0.5:
            return False

        # Check similarity to recent buffer entries
        sims = [oct_inner(x, buf_x) for buf_x, _ in node.buffer]
        mean_sim = sum(sims) / len(sims)

        return mean_sim > self.sim_threshold * 0.3

    # ── Insert ───────────────────────────────────────────────────────

    def insert(self, x: torch.Tensor, category: int | None = None) -> TrieNode:
        x = x.to(self.dtype)
        norm = x.norm()
        if norm > 0:
            x = x / norm

        self.total_inserts += 1
        node = self.root
        node.insert_count += 1
        if category is not None:
            node.category_counts[category] = node.category_counts.get(category, 0) + 1

        for depth in range(self.max_depth):
            if not node.children:
                # No children yet: create first child
                sub_idx, _, _ = self._find_best_child(node, x)
                child = self._make_child(x, sub_idx, depth + 1, category)
                node.children[sub_idx] = child
                return child

            sub_idx, child, assoc_norm = self._find_best_child(node, x)

            if child is None:
                # Unoccupied subalgebra: create new child
                child = self._make_child(x, sub_idx, depth + 1, category)
                node.children[sub_idx] = child
                return child

            if assoc_norm < self.assoc_threshold:
                # Associator says compatible: check rumination
                if self._ruminate(child, x):
                    # Consistent: descend and compose
                    node = child
                    node.insert_count += 1
                    if category is not None:
                        node.category_counts[category] = (
                            node.category_counts.get(category, 0) + 1
                        )
                    self._compose(node, x)
                    node.buffer.append((x.clone(), category))
                    continue
                else:
                    # Rumination rejected: this input doesn't belong here
                    self.rumination_rejections += 1
                    # Try to find a different home
                    product = octonion_mul(
                        node.routing_key.unsqueeze(0), x.unsqueeze(0)
                    ).squeeze(0)
                    activations = subalgebra_activation(product)
                    ranked = activations.argsort(descending=True)

                    # Skip the sub_idx we just rejected
                    for alt_sub in ranked:
                        alt_sub = alt_sub.item()
                        if alt_sub == sub_idx:
                            continue
                        if alt_sub not in node.children:
                            # Create new child in unoccupied slot
                            new_child = self._make_child(x, alt_sub, depth + 1, category)
                            node.children[alt_sub] = new_child
                            return new_child

                    # All occupied: descend into least-incompatible
                    node = child  # Fall through to next depth
                    node.insert_count += 1
                    if category is not None:
                        node.category_counts[category] = (
                            node.category_counts.get(category, 0) + 1
                        )
                    continue
            else:
                # Associator says incompatible: find unoccupied subalgebra
                product = octonion_mul(
                    node.routing_key.unsqueeze(0), x.unsqueeze(0)
                ).squeeze(0)
                activations = subalgebra_activation(product)
                ranked = activations.argsort(descending=True)

                for alt_sub in ranked:
                    alt_sub = alt_sub.item()
                    if alt_sub not in node.children:
                        new_child = self._make_child(x, alt_sub, depth + 1, category)
                        node.children[alt_sub] = new_child
                        return new_child

                # All 7 occupied: descend into the child we found
                node = child
                node.insert_count += 1
                if category is not None:
                    node.category_counts[category] = (
                        node.category_counts.get(category, 0) + 1
                    )
                continue

        # Max depth: compose into current node
        self._compose(node, x)
        node.buffer.append((x.clone(), category))

        # Periodic consolidation
        if self.total_inserts % self.consolidation_interval == 0:
            self._consolidate()

        return node

    # ── Query ────────────────────────────────────────────────────────

    def query(self, x: torch.Tensor) -> TrieNode:
        """Route x through the trie using the same logic as insert."""
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
                return node  # No matching child
            node = child

        return node

    # ── Helpers ───────────────────────────────────────────────────────

    def _make_child(self, x: torch.Tensor, sub_idx: int, depth: int,
                    category: int | None) -> TrieNode:
        child = TrieNode(
            routing_key=x.clone(),
            content=x.clone(),
            subalgebra_idx=sub_idx,
            depth=depth,
            buffer=deque(maxlen=30),
        )
        child.insert_count = 1
        if category is not None:
            child.category_counts[category] = 1
        child.buffer.append((x.clone(), category))
        self.n_nodes += 1
        return child

    def _compose(self, node: TrieNode, x: torch.Tensor) -> None:
        """Compose input into node's content. Routing key is never modified."""
        node.content = octonion_mul(
            node.content.unsqueeze(0), x.unsqueeze(0)
        ).squeeze(0)
        norm = node.content.norm()
        if norm > 0:
            node.content = node.content / norm

    def _consolidate(self) -> None:
        """Merge underused nodes into siblings."""
        def _walk(node: TrieNode) -> None:
            if not node.children:
                return
            for child in list(node.children.values()):
                _walk(child)

            total = sum(c.insert_count for c in node.children.values())
            if total == 0 or len(node.children) < 2:
                return

            to_remove = []
            for sub_idx, child in node.children.items():
                if child.insert_count / max(total, 1) < 0.05 and child.insert_count < 3:
                    to_remove.append(sub_idx)

            if not to_remove or len(node.children) - len(to_remove) < 1:
                return

            surviving = {k: v for k, v in node.children.items() if k not in to_remove}
            absorber_idx = max(surviving, key=lambda k: surviving[k].insert_count)
            absorber = surviving[absorber_idx]

            for sub_idx in to_remove:
                removed = node.children.pop(sub_idx)
                for cat, count in removed.category_counts.items():
                    absorber.category_counts[cat] = absorber.category_counts.get(cat, 0) + count
                absorber.insert_count += removed.insert_count
                self.n_nodes -= 1
                self.consolidation_merges += 1

        _walk(self.root)

    def stats(self) -> dict:
        nodes, leaves, max_depth = [], [], 0
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
            "n_nodes": len(nodes), "n_leaves": len(leaves),
            "max_depth": max_depth,
            "rumination_rejections": self.rumination_rejections,
            "consolidation_merges": self.consolidation_merges,
        }

    def print_tree(self, max_depth: int = 3) -> None:
        def _p(node: TrieNode, prefix: str = "", last: bool = True) -> None:
            conn = "'" if last else "|"
            sub = f"S{node.subalgebra_idx}" if node.subalgebra_idx is not None else "root"
            cat = node.dominant_category
            cl = f"[{cat}]" if cat is not None else ""
            logger.info(f"{prefix}{conn}-- {sub}{cl} n={node.insert_count}")
            if node.depth >= max_depth and node.children:
                np = prefix + ("   " if last else "|  ")
                logger.info(f"{np}'-- ...({len(node.children)}ch)")
                return
            kids = list(node.children.values())
            for i, c in enumerate(kids):
                np = prefix + ("   " if last else "|  ")
                _p(c, np, i == len(kids) - 1)
        _p(self.root)


# ── Data + Experiment ────────────────────────────────────────────────


def gen_data(n_cats, n_samples, noise=0.05, seed=42, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    samples, labels = [], []
    for cat in range(n_cats):
        center = torch.randn(8, dtype=dtype, generator=gen)
        center = center / center.norm()
        for _ in range(n_samples):
            s = center + noise * torch.randn(8, dtype=dtype, generator=gen)
            samples.append(s / s.norm())
            labels.append(cat)
    return samples, labels


def accuracy(trie, samples, labels, cats):
    correct, total = 0, 0
    per_cat = {}
    for s, l in zip(samples, labels, strict=False):
        if l not in cats:
            continue
        total += 1
        leaf = trie.query(s)
        pred = leaf.dominant_category
        if l not in per_cat:
            per_cat[l] = {"c": 0, "t": 0}
        per_cat[l]["t"] += 1
        if pred == l:
            correct += 1
            per_cat[l]["c"] += 1
    return correct / max(total, 1), per_cat


def run_test(
    n_p1=5, n_p2=5, n_train=100, n_test=50, noise=0.05, epochs=5,
    assoc_thresh=0.3, sim_thresh=0.5, seed=42,
):
    total = n_p1 + n_p2
    logger.info(f"  cats={total} ({n_p1}+{n_p2}), train={n_train}/cat, epochs={epochs}")
    logger.info(f"  noise={noise}, assoc_thresh={assoc_thresh}, sim_thresh={sim_thresh}")

    train_s, train_l = gen_data(total, n_train, noise=noise, seed=seed)
    test_s, test_l = gen_data(total, n_test, noise=noise, seed=seed + 1000)

    trie = OctonionTrie(
        associator_threshold=assoc_thresh,
        similarity_threshold=sim_thresh,
        seed=seed,
    )

    p1 = set(range(n_p1))
    p2 = set(range(n_p1, total))
    all_c = p1 | p2

    # Phase 1
    p1_idx = [i for i, l in enumerate(train_l) if l < n_p1]
    for ep in range(epochs):
        for idx in p1_idx:
            trie.insert(train_s[idx], category=train_l[idx])
        if (ep + 1) % max(1, epochs // 2) == 0:
            trie._consolidate()

    st1 = trie.stats()
    acc1_before, pc1_before = accuracy(trie, test_s, test_l, p1)
    logger.info(f"  P1 baseline: {acc1_before:.3f} ({st1['n_nodes']}n, {st1['n_leaves']}l, d={st1['max_depth']}, rum={st1['rumination_rejections']})")
    for c in sorted(pc1_before):
        r = pc1_before[c]
        logger.info(f"    cat{c}: {r['c']}/{r['t']}")

    # Phase 2
    p2_idx = [i for i, l in enumerate(train_l) if l >= n_p1]
    for ep in range(epochs):
        for idx in p2_idx:
            trie.insert(train_s[idx], category=train_l[idx])
        if (ep + 1) % max(1, epochs // 2) == 0:
            trie._consolidate()

    st2 = trie.stats()
    acc1_after, pc1_after = accuracy(trie, test_s, test_l, p1)
    acc2, pc2 = accuracy(trie, test_s, test_l, p2)
    acc_all, _ = accuracy(trie, test_s, test_l, all_c)
    forget = acc1_before - acc1_after

    logger.info(f"  P1 after P2: {acc1_after:.3f}")
    for c in sorted(pc1_after):
        r = pc1_after[c]
        logger.info(f"    cat{c}: {r['c']}/{r['t']}")
    logger.info(f"  P2: {acc2:.3f}")
    for c in sorted(pc2):
        r = pc2[c]
        logger.info(f"    cat{c}: {r['c']}/{r['t']}")

    fl = "NONE" if forget <= 0.01 else "MILD" if forget < 0.05 else "SOME" if forget < 0.1 else "SEVERE"
    logger.info(f"  RESULT: stab={acc1_after:.3f} plast={acc2:.3f} overall={acc_all:.3f} forget={forget:+.3f}({fl}) nodes={st2['n_nodes']} rum={st2['rumination_rejections']} merges={st2['consolidation_merges']}")

    trie.print_tree(max_depth=2)

    return {
        "stability": acc1_after, "stability_before": acc1_before,
        "plasticity": acc2, "overall": acc_all,
        "forgetting": forget, "stats": st2,
    }


def run_sweep(**kw):
    configs = [
        {"assoc_thresh": 0.2, "sim_thresh": 0.3},
        {"assoc_thresh": 0.3, "sim_thresh": 0.5},
        {"assoc_thresh": 0.5, "sim_thresh": 0.5},
        {"assoc_thresh": 0.3, "sim_thresh": 0.7},
        {"assoc_thresh": 0.5, "sim_thresh": 0.7},
        {"assoc_thresh": 0.7, "sim_thresh": 0.3},
        {"assoc_thresh": 0.7, "sim_thresh": 0.7},
    ]
    results = {}
    for cfg in configs:
        label = f"a{cfg['assoc_thresh']}_s{cfg['sim_thresh']}"
        logger.info(f"\n{'~'*50}\n{label}\n{'~'*50}")
        r = run_test(**{**kw, **cfg})
        results[label] = {
            "stability": r["stability"], "plasticity": r["plasticity"],
            "forgetting": r["forgetting"], "overall": r["overall"],
            "n_nodes": r["stats"]["n_nodes"],
            "rumination": r["stats"]["rumination_rejections"],
        }

    logger.info("\n" + "=" * 70)
    logger.info("Sweep Summary")
    logger.info("=" * 70)
    logger.info(f"  {'Config':>14} | {'Stab':>6} | {'Plast':>6} | {'Forget':>7} | {'All':>6} | {'Nodes':>5} | {'Rum':>5}")
    logger.info(f"  {'':->14}-+-{'':->6}-+-{'':->6}-+-{'':->7}-+-{'':->6}-+-{'':->5}-+-{'':->5}")
    for label, r in results.items():
        logger.info(
            f"  {label:>14} | {r['stability']:>6.3f} | {r['plasticity']:>6.3f} | "
            f"{r['forgetting']:>+7.3f} | {r['overall']:>6.3f} | {r['n_nodes']:>5} | {r['rumination']:>5}"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--assoc-thresh", type=float, default=0.3)
    parser.add_argument("--sim-thresh", type=float, default=0.5)
    parser.add_argument("--categories", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--train-per-cat", type=int, default=100)
    parser.add_argument("--test-per-cat", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/trie_validation")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    n_p1 = args.categories // 2
    n_p2 = args.categories - n_p1

    kw = dict(n_p1=n_p1, n_p2=n_p2, n_train=args.train_per_cat,
              n_test=args.test_per_cat, noise=args.noise, epochs=args.epochs)

    if args.sweep:
        r = run_sweep(**kw)
        with open(Path(args.output_dir) / "sweep.json", "w") as f:
            json.dump(r, f, indent=2)
    else:
        r = run_test(assoc_thresh=args.assoc_thresh, sim_thresh=args.sim_thresh, **kw)
        with open(Path(args.output_dir) / "result.json", "w") as f:
            json.dump(r, f, indent=2, default=str)


if __name__ == "__main__":
    main()
