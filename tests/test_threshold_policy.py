"""Tests for ThresholdPolicy abstraction and all strategy classes."""

import inspect
import math

import pytest
import torch

from octonion._fano import FANO_PLANE
from octonion.trie import (
    AlgebraicPurityPolicy,
    DepthPolicy,
    GlobalPolicy,
    HybridPolicy,
    MetaTriePolicy,
    OctonionTrie,
    PerNodeEMAPolicy,
    PerNodeMeanStdPolicy,
    ThresholdPolicy,
    TrieNode,
)


# -- Helpers (copied from test_trie.py for independence) ---------------------


def _make_aligned_centers(noise: float = 0.0) -> list[torch.Tensor]:
    """Create 7 category centers, each aligned with one Fano subalgebra."""
    centers = []
    for triple in FANO_PLANE.triples:
        i, j, k = triple
        c = torch.zeros(8, dtype=torch.float64)
        c[0] = 0.25
        c[i] = 0.75
        c[j] = 0.5
        c[k] = 0.3
        centers.append(c / c.norm())
    return centers


def _generate_samples(
    centers: list[torch.Tensor], n_per_cat: int, noise: float = 0.05, seed: int = 99
) -> tuple[list[torch.Tensor], list[int]]:
    """Generate noisy samples around category centers."""
    gen = torch.Generator().manual_seed(seed)
    samples, labels = [], []
    for cat, center in enumerate(centers):
        for _ in range(n_per_cat):
            s = center + noise * torch.randn(8, dtype=torch.float64, generator=gen)
            samples.append(s / s.norm())
            labels.append(cat)
    return samples, labels


def _accuracy(trie: OctonionTrie, samples, labels, cats=None):
    """Measure classification accuracy."""
    if cats is None:
        cats = set(labels)
    correct = total = 0
    for s, l in zip(samples, labels):
        if l not in cats:
            continue
        total += 1
        leaf = trie.query(s)
        if leaf.dominant_category == l:
            correct += 1
    return correct / max(total, 1)


# -- Test 1: GlobalPolicy defaults ------------------------------------------


def test_global_policy_defaults():
    """GlobalPolicy with default args returns assoc=0.3, sim=0.1,
    consolidation=(0.05, 3) for any node/depth."""
    policy = GlobalPolicy()
    node = TrieNode(routing_key=torch.zeros(8), content=torch.zeros(8))

    assert policy.get_assoc_threshold(node, 0) == pytest.approx(0.3)
    assert policy.get_assoc_threshold(node, 10) == pytest.approx(0.3)
    assert policy.get_sim_threshold(node, 0) == pytest.approx(0.1)
    assert policy.get_sim_threshold(node, 5) == pytest.approx(0.1)
    assert policy.get_consolidation_params(node, 0) == (pytest.approx(0.05), 3)
    assert policy.get_consolidation_params(node, 7) == (pytest.approx(0.05), 3)


# -- Test 2: Backward compatibility -----------------------------------------


def test_global_policy_backward_compat():
    """OctonionTrie(associator_threshold=0.5) produces identical results to
    OctonionTrie(policy=GlobalPolicy(assoc_threshold=0.5)) on 7-category
    classification."""
    centers = _make_aligned_centers()
    train_s, train_l = _generate_samples(centers, 20, noise=0.05, seed=99)
    test_s, test_l = _generate_samples(centers, 10, noise=0.05, seed=7777)

    # Trie 1: old-style API
    trie1 = OctonionTrie(associator_threshold=0.5, similarity_threshold=0.15, seed=42)
    for ep in range(3):
        for s, l in zip(train_s, train_l):
            trie1.insert(s, category=l)

    # Trie 2: new policy API with same parameters
    policy = GlobalPolicy(assoc_threshold=0.5, sim_threshold=0.15)
    trie2 = OctonionTrie(policy=policy, seed=42)
    for ep in range(3):
        for s, l in zip(train_s, train_l):
            trie2.insert(s, category=l)

    stats1 = trie1.stats()
    stats2 = trie2.stats()
    acc1 = _accuracy(trie1, test_s, test_l)
    acc2 = _accuracy(trie2, test_s, test_l)

    assert stats1["n_nodes"] == stats2["n_nodes"], (
        f"Node count mismatch: {stats1['n_nodes']} vs {stats2['n_nodes']}"
    )
    assert stats1["n_leaves"] == stats2["n_leaves"]
    assert stats1["max_depth"] == stats2["max_depth"]
    assert stats1["rumination_rejections"] == stats2["rumination_rejections"]
    assert acc1 == pytest.approx(acc2, abs=1e-10), (
        f"Accuracy mismatch: {acc1} vs {acc2}"
    )


# -- Test 3: PerNodeEMAPolicy fallback and adaptation -----------------------


def test_per_node_ema_adapts():
    """PerNodeEMAPolicy returns base threshold for node with < 3 observations,
    then adapts after 10+ insertions."""
    policy = PerNodeEMAPolicy(alpha=0.1, k=1.5, base_assoc=0.3, min_obs=3)
    node = TrieNode(routing_key=torch.randn(8), content=torch.randn(8))

    # Before any observations: returns base
    assert policy.get_assoc_threshold(node, 0) == pytest.approx(0.3)

    # After 1 observation: still returns base (< min_obs=3)
    x = torch.randn(8, dtype=torch.float64)
    policy.on_insert(node, x, 0.2)
    assert policy.get_assoc_threshold(node, 0) == pytest.approx(0.3)

    # After 2 observations: still returns base
    policy.on_insert(node, x, 0.25)
    assert policy.get_assoc_threshold(node, 0) == pytest.approx(0.3)

    # After 3+ observations: adapts based on EMA stats
    for _ in range(10):
        policy.on_insert(node, x, 0.2)
    threshold = policy.get_assoc_threshold(node, 0)
    # Should be mean + k*std, not the base anymore
    assert threshold != pytest.approx(0.3, abs=0.01), (
        f"Expected adapted threshold, got {threshold}"
    )


# -- Test 4: PerNodeEMAPolicy on_insert updates state -----------------------


def test_per_node_ema_state_keys():
    """PerNodeEMAPolicy.on_insert updates node._policy_state with ema_mean,
    ema_var, and ema_count keys."""
    policy = PerNodeEMAPolicy()
    node = TrieNode(routing_key=torch.randn(8), content=torch.randn(8))

    assert "ema_mean" not in node._policy_state

    x = torch.randn(8, dtype=torch.float64)
    policy.on_insert(node, x, 0.15)

    assert "ema_mean" in node._policy_state
    assert "ema_var" in node._policy_state
    assert "ema_count" in node._policy_state
    assert node._policy_state["ema_count"] == 1
    assert node._policy_state["ema_mean"] == pytest.approx(0.15)


# -- Test 5: PerNodeMeanStdPolicy convergence --------------------------------


def test_per_node_mean_std_converges():
    """PerNodeMeanStdPolicy converges to sample mean + k*std after many
    insertions."""
    k = 1.5
    policy = PerNodeMeanStdPolicy(k=k, base_assoc=0.3, min_obs=3)
    node = TrieNode(routing_key=torch.randn(8), content=torch.randn(8))
    x = torch.randn(8, dtype=torch.float64)

    # Insert many values with known distribution
    values = [0.1, 0.2, 0.3, 0.15, 0.25, 0.12, 0.28, 0.18, 0.22, 0.17]
    for v in values:
        policy.on_insert(node, x, v)

    # Compute expected threshold
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    expected = mean + k * std

    actual = policy.get_assoc_threshold(node, 0)
    assert actual == pytest.approx(expected, rel=1e-6), (
        f"Expected {expected:.6f}, got {actual:.6f}"
    )


# -- Test 6: DepthPolicy with decay < 1 ------------------------------------


def test_depth_policy_decay():
    """DepthPolicy with decay=0.8 returns base*0.8^depth."""
    policy = DepthPolicy(base_assoc=1.0, decay_factor=0.8)
    node = TrieNode(routing_key=torch.zeros(8), content=torch.zeros(8))

    assert policy.get_assoc_threshold(node, 0) == pytest.approx(1.0)
    assert policy.get_assoc_threshold(node, 1) == pytest.approx(0.8)
    assert policy.get_assoc_threshold(node, 2) == pytest.approx(0.64)
    assert policy.get_assoc_threshold(node, 3) == pytest.approx(0.512)


# -- Test 7: DepthPolicy with decay > 1 ------------------------------------


def test_depth_policy_increasing():
    """DepthPolicy with decay=1.2 returns base*1.2^depth (increases with depth)."""
    policy = DepthPolicy(base_assoc=0.3, decay_factor=1.2)
    node = TrieNode(routing_key=torch.zeros(8), content=torch.zeros(8))

    assert policy.get_assoc_threshold(node, 0) == pytest.approx(0.3)
    assert policy.get_assoc_threshold(node, 1) == pytest.approx(0.3 * 1.2)
    assert policy.get_assoc_threshold(node, 3) == pytest.approx(0.3 * 1.2 ** 3)


# -- Test 8: AlgebraicPurityPolicy with empty/filled buffer ----------------


def test_algebraic_purity_empty_and_filled():
    """AlgebraicPurityPolicy returns base threshold when buffer is empty,
    adjusts after buffer fills."""
    policy = AlgebraicPurityPolicy(base_assoc=0.3, sensitivity=2.0)

    # Node with empty buffer: returns base
    node = TrieNode(
        routing_key=torch.randn(8, dtype=torch.float64),
        content=torch.randn(8, dtype=torch.float64),
    )
    assert policy.get_assoc_threshold(node, 0) == pytest.approx(0.3)

    # Fill buffer with diverse samples
    gen = torch.Generator().manual_seed(42)
    for _ in range(10):
        sample = torch.randn(8, dtype=torch.float64, generator=gen)
        sample = sample / sample.norm()
        node.buffer.append((sample, 0))

    # After filling: threshold should differ from base (variance-adjusted)
    threshold = policy.get_assoc_threshold(node, 0)
    # With diverse random samples, variance should be nonzero, so threshold > base
    assert threshold >= 0.3, f"Expected >= base, got {threshold}"


# -- Test 9: MetaTriePolicy stub raises ------------------------------------


def test_meta_trie_stub_raises():
    """MetaTriePolicy stub raises NotImplementedError on get_assoc_threshold."""
    policy = MetaTriePolicy()
    node = TrieNode(routing_key=torch.zeros(8), content=torch.zeros(8))

    with pytest.raises(NotImplementedError, match="configure_meta_trie"):
        policy.get_assoc_threshold(node, 0)

    with pytest.raises(NotImplementedError, match="configure_meta_trie"):
        policy.get_sim_threshold(node, 0)

    with pytest.raises(NotImplementedError, match="configure_meta_trie"):
        policy.get_consolidation_params(node, 0)


# -- Test 10: HybridPolicy stub raises ------------------------------------


def test_hybrid_stub_raises():
    """HybridPolicy stub raises NotImplementedError on get_assoc_threshold."""
    policy = HybridPolicy()
    node = TrieNode(routing_key=torch.zeros(8), content=torch.zeros(8))

    with pytest.raises(NotImplementedError, match="configure.*two base policies"):
        policy.get_assoc_threshold(node, 0)

    with pytest.raises(NotImplementedError, match="configure.*two base policies"):
        policy.get_sim_threshold(node, 0)

    with pytest.raises(NotImplementedError, match="configure.*two base policies"):
        policy.get_consolidation_params(node, 0)


# -- Test 11: Adaptive changes tree structure --------------------------------


def test_adaptive_changes_structure():
    """OctonionTrie with PerNodeEMAPolicy produces different tree structure
    than GlobalPolicy on same data."""
    centers = _make_aligned_centers()
    train_s, train_l = _generate_samples(centers, 30, noise=0.05, seed=99)

    # Global policy
    trie_global = OctonionTrie(
        policy=GlobalPolicy(assoc_threshold=0.3), seed=42
    )
    for ep in range(3):
        for s, l in zip(train_s, train_l):
            trie_global.insert(s, category=l)

    # Adaptive policy with very tight k=0.5
    trie_adaptive = OctonionTrie(
        policy=PerNodeEMAPolicy(alpha=0.1, k=0.5, base_assoc=0.3, min_obs=2),
        seed=42,
    )
    for ep in range(3):
        for s, l in zip(train_s, train_l):
            trie_adaptive.insert(s, category=l)

    stats_g = trie_global.stats()
    stats_a = trie_adaptive.stats()

    # The adaptive policy should produce a different number of nodes
    # (either more or fewer, depending on the data distribution)
    assert stats_g["n_nodes"] != stats_a["n_nodes"], (
        f"Expected different node counts, both have {stats_g['n_nodes']}"
    )


# -- Test 12: Unsupervised constraint (D-02) --------------------------------


def test_unsupervised_constraint():
    """All ThresholdPolicy methods must NOT accept a 'category' parameter.
    Policies use only algebraic signals, never category labels."""
    policy_classes = [
        GlobalPolicy,
        PerNodeEMAPolicy,
        PerNodeMeanStdPolicy,
        DepthPolicy,
        AlgebraicPurityPolicy,
        MetaTriePolicy,
        HybridPolicy,
    ]

    methods_to_check = [
        "get_assoc_threshold",
        "get_sim_threshold",
        "get_consolidation_params",
        "on_insert",
    ]

    for cls in policy_classes:
        for method_name in methods_to_check:
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            sig = inspect.signature(method)
            param_names = list(sig.parameters.keys())
            assert "category" not in param_names, (
                f"{cls.__name__}.{method_name} accepts 'category' parameter, "
                f"violating D-02 unsupervised constraint. "
                f"Params: {param_names}"
            )
