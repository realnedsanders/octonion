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

import math
from abc import ABC, abstractmethod
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
    _policy_state: dict = field(default_factory=dict)

    @property
    def dominant_category(self) -> int | None:
        """Category with the most inserts at this node."""
        if not self.category_counts:
            return None
        return max(self.category_counts, key=lambda k: self.category_counts[k])

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ── ThresholdPolicy abstraction ──────────────────────────────────────


class ThresholdPolicy(ABC):
    """Abstract base class for trie threshold strategies.

    The OctonionTrie delegates all threshold decisions to a ThresholdPolicy.
    This decouples the trie's self-organization logic from how thresholds
    are determined, enabling pluggable adaptation strategies.
    """

    @abstractmethod
    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        """Return the associator norm threshold for routing at this node."""
        ...

    @abstractmethod
    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        """Return the similarity threshold for rumination at this node."""
        ...

    @abstractmethod
    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        """Return (min_share, min_count) for consolidation at this node."""
        ...

    def on_insert(self, node: TrieNode, x: torch.Tensor, assoc_norm: float) -> None:
        """Optional hook called after each insertion for policy updates."""
        pass


class GlobalPolicy(ThresholdPolicy):
    """Global (fixed) threshold policy -- reproduces original hardcoded behavior.

    All nodes at all depths use the same thresholds. This is the baseline
    policy and the default when no explicit policy is provided.
    """

    def __init__(
        self,
        assoc_threshold: float = 0.3,
        sim_threshold: float = 0.1,
        min_share: float = 0.05,
        min_count: int = 3,
    ):
        self.assoc_threshold = assoc_threshold
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        return self.assoc_threshold

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self.sim_threshold

    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        return self.min_share, self.min_count


class PerNodeEMAPolicy(ThresholdPolicy):
    """Per-node EMA of observed associator norms.

    Each node maintains an exponential moving average of associator norms
    seen during insertion. The threshold adapts to mean + k * std of the
    local distribution. Falls back to base threshold until the node has
    accumulated enough observations (min_obs).

    Per-node state keys: ema_mean, ema_var, ema_count
    """

    def __init__(
        self,
        alpha: float = 0.1,
        k: float = 1.5,
        base_assoc: float = 0.3,
        sim_threshold: float = 0.1,
        min_share: float = 0.05,
        min_count: int = 3,
        min_obs: int = 3,
    ):
        self.alpha = alpha
        self.k = k
        self.base_assoc = base_assoc
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count
        self.min_obs = min_obs

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        count = node._policy_state.get("ema_count", 0)
        if count < self.min_obs:
            return self.base_assoc
        mean = node._policy_state["ema_mean"]
        var = node._policy_state["ema_var"]
        std = math.sqrt(max(var, 0.0))
        return mean + self.k * std

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self.sim_threshold

    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        return self.min_share, self.min_count

    def on_insert(self, node: TrieNode, x: torch.Tensor, assoc_norm: float) -> None:
        count = node._policy_state.get("ema_count", 0)
        if count == 0:
            node._policy_state["ema_mean"] = assoc_norm
            node._policy_state["ema_var"] = 0.0
            node._policy_state["ema_count"] = 1
        else:
            old_mean = node._policy_state["ema_mean"]
            new_mean = (1 - self.alpha) * old_mean + self.alpha * assoc_norm
            diff = assoc_norm - old_mean
            new_var = (1 - self.alpha) * node._policy_state["ema_var"] + self.alpha * diff * diff
            node._policy_state["ema_mean"] = new_mean
            node._policy_state["ema_var"] = new_var
            node._policy_state["ema_count"] = count + 1


class PerNodeMeanStdPolicy(ThresholdPolicy):
    """Per-node running mean + std using Welford's online algorithm.

    Like PerNodeEMAPolicy but uses unweighted running statistics --
    all observations contribute equally regardless of order. The threshold
    adapts to mean + k * std after sufficient observations.

    Per-node state keys: welford_mean, welford_M2, welford_count
    """

    def __init__(
        self,
        k: float = 1.5,
        base_assoc: float = 0.3,
        sim_threshold: float = 0.1,
        min_share: float = 0.05,
        min_count: int = 3,
        min_obs: int = 3,
    ):
        self.k = k
        self.base_assoc = base_assoc
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count
        self.min_obs = min_obs

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        count = node._policy_state.get("welford_count", 0)
        if count < self.min_obs:
            return self.base_assoc
        mean = node._policy_state["welford_mean"]
        M2 = node._policy_state["welford_M2"]
        var = M2 / count
        std = math.sqrt(max(var, 0.0))
        return mean + self.k * std

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self.sim_threshold

    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        return self.min_share, self.min_count

    def on_insert(self, node: TrieNode, x: torch.Tensor, assoc_norm: float) -> None:
        count = node._policy_state.get("welford_count", 0)
        if count == 0:
            node._policy_state["welford_mean"] = assoc_norm
            node._policy_state["welford_M2"] = 0.0
            node._policy_state["welford_count"] = 1
        else:
            count += 1
            old_mean = node._policy_state["welford_mean"]
            delta = assoc_norm - old_mean
            new_mean = old_mean + delta / count
            delta2 = assoc_norm - new_mean
            new_M2 = node._policy_state["welford_M2"] + delta * delta2
            node._policy_state["welford_mean"] = new_mean
            node._policy_state["welford_M2"] = new_M2
            node._policy_state["welford_count"] = count


class DepthPolicy(ThresholdPolicy):
    """Depth-dependent threshold: threshold = base * decay_factor ^ depth.

    decay_factor < 1: thresholds tighten with depth (deeper = stricter).
    decay_factor > 1: thresholds loosen with depth (deeper = more tolerant).
    decay_factor = 1: equivalent to GlobalPolicy.
    """

    def __init__(
        self,
        base_assoc: float = 0.3,
        decay_factor: float = 1.0,
        sim_threshold: float = 0.1,
        min_share: float = 0.05,
        min_count: int = 3,
    ):
        self.base_assoc = base_assoc
        self.decay_factor = decay_factor
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        return self.base_assoc * (self.decay_factor ** depth)

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self.sim_threshold

    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        return self.min_share, self.min_count


class AlgebraicPurityPolicy(ThresholdPolicy):
    """Threshold based on algebraic purity of the node's buffer.

    Uses two independent signals from the node's buffer:
    (a) Variance of associator norms between buffer entries and the routing key.
    (b) Variance of inner products between buffer entries and the routing key.

    Low variance = high algebraic purity = can tighten threshold.
    High variance = heterogeneous content = should loosen threshold.

    threshold = base * (1 + sensitivity * combined_signal)
    combined_signal = assoc_weight * norm_variance + sim_weight * sim_variance
    """

    def __init__(
        self,
        base_assoc: float = 0.3,
        assoc_weight: float = 0.5,
        sim_weight: float = 0.5,
        sensitivity: float = 2.0,
        sim_threshold: float = 0.1,
        min_share: float = 0.05,
        min_count: int = 3,
    ):
        self.base_assoc = base_assoc
        self.assoc_weight = assoc_weight
        self.sim_weight = sim_weight
        self.sensitivity = sensitivity
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        if len(node.buffer) < 3:
            return self.base_assoc

        # Compute associator norm variance across buffer entries
        node_oct = Octonion(node.routing_key)
        assoc_norms = []
        sim_values = []
        for buf_x, _ in node.buffer:
            buf_oct = Octonion(buf_x)
            a = associator(buf_oct, node_oct, node_oct)
            assoc_norms.append(a.components.norm().item())
            sim_values.append(torch.dot(buf_x, node.routing_key).item())

        # Variance of associator norms
        if len(assoc_norms) > 1:
            mean_a = sum(assoc_norms) / len(assoc_norms)
            var_a = sum((v - mean_a) ** 2 for v in assoc_norms) / len(assoc_norms)
        else:
            var_a = 0.0

        # Variance of similarity values
        if len(sim_values) > 1:
            mean_s = sum(sim_values) / len(sim_values)
            var_s = sum((v - mean_s) ** 2 for v in sim_values) / len(sim_values)
        else:
            var_s = 0.0

        combined = self.assoc_weight * var_a + self.sim_weight * var_s
        return self.base_assoc * (1.0 + self.sensitivity * combined)

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self.sim_threshold

    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        return self.min_share, self.min_count


class MetaTriePolicy(ThresholdPolicy):
    """Meta-trie optimizer: a second OctonionTrie adapts classifier thresholds.

    Per D-12: Uses the same OctonionTrie class (not a subclass).
    Per D-13: Categories are discretized threshold actions.
    Per D-14: Two input encoding modes.
    Per D-15: Two feedback signal modes.
    """

    # Threshold actions per D-13
    ACTIONS = {
        0: -0.20,  # "tighten 20%"
        1: -0.10,  # "tighten 10%"
        2:  0.00,  # "keep"
        3:  0.10,  # "loosen 10%"
        4:  0.20,  # "loosen 20%"
    }

    def __init__(
        self,
        base_assoc: float = 0.3,
        sim_threshold: float = 0.1,
        min_share: float = 0.05,
        min_count: int = 3,
        signal_encoding: str = "signal_vector",  # or "algebraic" per D-14
        feedback_signal: str = "stability",       # or "accuracy" per D-15
        update_frequency: int = 100,              # per D-16: per-N-inserts
        self_referential: bool = False,            # per D-17
        meta_seed: int = 7919,
    ):
        self.base_assoc = base_assoc
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count
        self.signal_encoding = signal_encoding
        self.feedback_signal = feedback_signal
        self.update_frequency = update_frequency
        self.self_referential = self_referential

        # Create the meta-trie per D-12
        self.meta_trie = OctonionTrie(
            associator_threshold=base_assoc,
            similarity_threshold=sim_threshold,
            seed=meta_seed,
        )

        # Per-node threshold adjustments (accumulated from meta-trie decisions)
        self._node_adjustments: dict[int, float] = {}  # id(node) -> adjustment factor
        self._insert_counter = 0
        self._convergence_history: list[float] = []  # per D-18: track threshold change rate
        self._prev_adjustments: dict[int, float] = {}

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        adjustment = self._node_adjustments.get(id(node), 0.0)
        return max(0.001, self.base_assoc * (1.0 + adjustment))

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self.sim_threshold

    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        return self.min_share, self.min_count

    def on_insert(self, node: TrieNode, x: torch.Tensor, assoc_norm: float) -> None:
        # Track stats in node._policy_state
        state = node._policy_state
        state.setdefault("meta_assoc_norms", []).append(assoc_norm)
        state["meta_insert_count"] = state.get("meta_insert_count", 0) + 1
        # Keep only last 100 norms to avoid unbounded memory
        if len(state["meta_assoc_norms"]) > 100:
            state["meta_assoc_norms"] = state["meta_assoc_norms"][-100:]

        self._insert_counter += 1
        if self._insert_counter % self.update_frequency == 0:
            self._update_thresholds(node)

    def _encode_signal_vector(self, node: TrieNode) -> torch.Tensor:
        """Encode node state as 8D signal vector per D-14 option 1."""
        state = node._policy_state
        norms = state.get("meta_assoc_norms", [0.0])
        norms_t = torch.tensor(norms, dtype=torch.float64)
        return torch.tensor([
            norms_t.mean().item(),           # assoc_norm_mean
            norms_t.std().item() if len(norms) > 1 else 0.0,  # assoc_norm_std
            len(node.children) / 7.0,        # branching_factor / 7
            node.insert_count / max(self._insert_counter, 1),  # insert_rate
            0.0,  # rumination_rate (computed from parent trie stats if available)
            node.depth / 15.0,               # depth / max_depth
            0.0,  # buffer_consistency (computed from buffer similarity)
            0.0,  # consolidation_rate
        ], dtype=torch.float64)

    def _encode_algebraic(self, node: TrieNode) -> torch.Tensor:
        """Use node's routing key as meta-trie input per D-14 option 2."""
        return node.routing_key.clone()

    def _compute_stability_signal(self, node: TrieNode) -> int:
        """Compute unsupervised stability signal per D-15 option 1.

        Returns action category (0-4) based on node stability indicators.
        Low rumination + balanced branching + consistent norms -> "keep" (2)
        """
        state = node._policy_state
        norms = state.get("meta_assoc_norms", [])
        if len(norms) < 3:
            return 2  # "keep" -- not enough data

        norms_t = torch.tensor(norms[-30:], dtype=torch.float64)
        cv = (norms_t.std() / norms_t.mean()).item() if norms_t.mean() > 1e-10 else 0.0

        # High CV = inconsistent = should tighten; Low CV = stable = can loosen
        if cv > 1.0:
            return 0  # tighten 20%
        elif cv > 0.5:
            return 1  # tighten 10%
        elif cv < 0.1:
            return 4  # loosen 20%
        elif cv < 0.2:
            return 3  # loosen 10%
        else:
            return 2  # keep

    def _update_thresholds(self, trigger_node: TrieNode) -> None:
        """Update threshold adjustments via meta-trie.

        Encodes trigger_node state, inserts into meta-trie,
        queries meta-trie for recommended action, applies adjustment.
        """
        # Encode input based on D-14
        if self.signal_encoding == "signal_vector":
            meta_input = self._encode_signal_vector(trigger_node)
        else:
            meta_input = self._encode_algebraic(trigger_node)

        # Determine category based on D-15
        if self.feedback_signal == "stability":
            action_cat = self._compute_stability_signal(trigger_node)
        else:
            action_cat = 2  # "keep" for accuracy mode (set externally)

        # Insert into meta-trie
        self.meta_trie.insert(meta_input, category=action_cat)

        # Query meta-trie for recommendation
        leaf = self.meta_trie.query(meta_input)
        recommended = leaf.dominant_category
        if recommended is not None and recommended in self.ACTIONS:
            adjustment = self.ACTIONS[recommended]
            self._node_adjustments[id(trigger_node)] = adjustment

        # Per D-17: self-referential -- meta-trie adapts its own thresholds
        if self.self_referential:
            meta_signal = self._encode_signal_vector(trigger_node)
            meta_leaf = self.meta_trie.query(meta_signal)
            if meta_leaf.dominant_category is not None:
                meta_adj = self.ACTIONS.get(meta_leaf.dominant_category, 0.0)
                self.meta_trie.assoc_threshold = max(
                    0.001, self.base_assoc * (1.0 + meta_adj)
                )

        # Per D-18: convergence tracking
        curr_adj = dict(self._node_adjustments)
        if self._prev_adjustments:
            changes = []
            for k in set(curr_adj) | set(self._prev_adjustments):
                old = self._prev_adjustments.get(k, 0.0)
                new = curr_adj.get(k, 0.0)
                changes.append(abs(new - old))
            change_rate = sum(changes) / max(len(changes), 1)
            self._convergence_history.append(change_rate)
        self._prev_adjustments = curr_adj

    @property
    def converged(self) -> bool:
        """Per D-18: converged if threshold change rate < 1%."""
        if len(self._convergence_history) < 3:
            return False
        return self._convergence_history[-1] < 0.01


class HybridPolicy(ThresholdPolicy):
    """Combines two ThresholdPolicy instances per D-09.

    Combination modes:
    - "mean": average of both policies' thresholds
    - "min": minimum (more conservative / tighter)
    - "max": maximum (more permissive / looser)
    - "adaptive": use policy_a in early epochs, transition to policy_b
    """

    def __init__(
        self,
        policy_a: ThresholdPolicy | None = None,
        policy_b: ThresholdPolicy | None = None,
        combination: str = "mean",
        transition_inserts: int = 0,  # for "adaptive" mode: switch after N inserts
    ):
        self.policy_a = policy_a if policy_a is not None else GlobalPolicy()
        self.policy_b = policy_b if policy_b is not None else GlobalPolicy()
        self.combination = combination
        self.transition_inserts = transition_inserts
        self._total_inserts = 0

    def _combine(self, val_a: float, val_b: float) -> float:
        if self.combination == "mean":
            return (val_a + val_b) / 2.0
        elif self.combination == "min":
            return min(val_a, val_b)
        elif self.combination == "max":
            return max(val_a, val_b)
        elif self.combination == "adaptive":
            # Smooth transition from policy_a to policy_b
            if self.transition_inserts <= 0:
                return val_b
            alpha = min(1.0, self._total_inserts / self.transition_inserts)
            return (1 - alpha) * val_a + alpha * val_b
        return (val_a + val_b) / 2.0

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        return self._combine(
            self.policy_a.get_assoc_threshold(node, depth),
            self.policy_b.get_assoc_threshold(node, depth),
        )

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self._combine(
            self.policy_a.get_sim_threshold(node, depth),
            self.policy_b.get_sim_threshold(node, depth),
        )

    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        ms_a, mc_a = self.policy_a.get_consolidation_params(node, depth)
        ms_b, mc_b = self.policy_b.get_consolidation_params(node, depth)
        return self._combine(ms_a, ms_b), int(self._combine(mc_a, mc_b))

    def on_insert(self, node: TrieNode, x: torch.Tensor, assoc_norm: float) -> None:
        self._total_inserts += 1
        self.policy_a.on_insert(node, x, assoc_norm)
        self.policy_b.on_insert(node, x, assoc_norm)


# ── OctonionTrie ─────────────────────────────────────────────────────


class OctonionTrie:
    """Self-organizing octonionic trie.

    Args:
        associator_threshold: Maximum associator norm for routing compatibility.
            Lower values produce more branching (finer discrimination).
        similarity_threshold: Minimum inner product for rumination acceptance.
        max_depth: Maximum trie depth.
        seed: Random seed for root routing key initialization.
        dtype: Tensor dtype (float64 recommended for algebraic precision).
        policy: Pluggable ThresholdPolicy. If None, a GlobalPolicy is created
            from associator_threshold and similarity_threshold values.
    """

    def __init__(
        self,
        associator_threshold: float = 0.3,
        similarity_threshold: float = 0.1,
        max_depth: int = 15,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
        policy: ThresholdPolicy | None = None,
    ):
        # Default threshold/policy reviewed in Phase T2 (adaptive thresholds) based
        # on cross-benchmark analysis. GlobalPolicy(assoc_threshold=0.3) remains the
        # default: the Phase T2 ThresholdPolicy abstraction added 8 strategy
        # implementations (EMA, MeanStd, Depth, AlgebraicPurity, MetaTrie, Hybrid),
        # but the global baseline with threshold 0.3 provides robust performance
        # across all 5 benchmarks (MNIST, Fashion-MNIST, CIFAR-10, Text 4/20-class).
        # Adaptive strategies are available via the `policy` parameter for tasks
        # where per-node or depth-dependent thresholds are beneficial.
        # See results/T2/analysis/statistical_report.json for full analysis.
        gen = torch.Generator().manual_seed(seed)
        root_key = torch.randn(8, dtype=dtype, generator=gen)
        root_key = root_key / root_key.norm()

        self.root = TrieNode(routing_key=root_key, content=root_key.clone())
        self.max_depth = max_depth
        self.dtype = dtype
        self.n_nodes = 1
        self.total_inserts = 0
        self.rumination_rejections = 0
        self.consolidation_merges = 0

        # Set up threshold policy
        if policy is not None:
            self.policy = policy
        else:
            self.policy = GlobalPolicy(
                assoc_threshold=associator_threshold,
                sim_threshold=similarity_threshold,
            )

    @property
    def assoc_threshold(self) -> float:
        """Backward-compatible property delegating to policy."""
        return self.policy.get_assoc_threshold(self.root, 0)

    @assoc_threshold.setter
    def assoc_threshold(self, value: float) -> None:
        """Backward-compatible setter -- only works with GlobalPolicy."""
        if isinstance(self.policy, GlobalPolicy):
            self.policy.assoc_threshold = value

    @property
    def sim_threshold(self) -> float:
        """Backward-compatible property delegating to policy."""
        return self.policy.get_sim_threshold(self.root, 0)

    @sim_threshold.setter
    def sim_threshold(self, value: float) -> None:
        """Backward-compatible setter -- only works with GlobalPolicy."""
        if isinstance(self.policy, GlobalPolicy):
            self.policy.sim_threshold = value

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

            threshold = self.policy.get_assoc_threshold(child, node.depth)
            if assoc_norm < threshold:
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
        threshold = self.policy.get_assoc_threshold(
            node.children[best_sim_idx], node.depth
        )
        return best_sim_idx, node.children[best_sim_idx], threshold + 1

    def _ruminate(self, node: TrieNode, x: torch.Tensor) -> bool:
        """Geometric consistency check: is x similar to this node's history?"""
        if len(node.buffer) < 3:
            return True
        sim_thresh = self.policy.get_sim_threshold(node, node.depth)
        key_sim = torch.dot(x, node.routing_key).item()
        if key_sim < sim_thresh * 0.5:
            return False
        sims = [torch.dot(x, buf_x).item() for buf_x, _ in node.buffer]
        return sum(sims) / len(sims) > sim_thresh * 0.3

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
                child = self._create_child(node, x, sub_idx, category)
                self.policy.on_insert(child, x, float("inf"))
                return child

            sub_idx, child, assoc_norm = self._find_best_child(node, x)

            if child is None:
                new_child = self._create_child(node, x, sub_idx, category)
                self.policy.on_insert(new_child, x, float("inf"))
                return new_child

            threshold = self.policy.get_assoc_threshold(child, node.depth)
            if assoc_norm < threshold and self._ruminate(child, x):
                node = child
                self._count(node, category)
                self._compose(node, x)
                node.buffer.append((x.clone(), category))
                self.policy.on_insert(node, x, assoc_norm)
                continue

            if assoc_norm >= threshold:
                self.rumination_rejections += int(assoc_norm < threshold)
                # Find unoccupied slot
                product = octonion_mul(
                    node.routing_key.unsqueeze(0), x.unsqueeze(0)
                ).squeeze(0)
                activations = subalgebra_activation(product)
                for alt in activations.argsort(descending=True):
                    alt_idx = alt.item()
                    if alt_idx not in node.children:
                        new_child = self._create_child(node, x, alt_idx, category)
                        self.policy.on_insert(new_child, x, assoc_norm)
                        return new_child
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
                        new_child = self._create_child(node, x, alt_idx, category)
                        self.policy.on_insert(new_child, x, assoc_norm)
                        return new_child
                node = child
                self._count(node, category)
                continue

        self._compose(node, x)
        node.buffer.append((x.clone(), category))
        self.policy.on_insert(node, x, float("inf"))
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

    # -- Private helpers --------------------------------------------------

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

        min_share, min_count = self.policy.get_consolidation_params(node, node.depth)
        to_remove = [
            idx
            for idx, child in node.children.items()
            if child.insert_count / max(total, 1) < min_share
            and child.insert_count < min_count
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
