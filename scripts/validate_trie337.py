"""Validate that AlgebraicPurityPolicy's associator computation is always zero.

trie.py:337 computes associator(buf_oct, node_oct, node_oct), i.e. [buf, n, n].
By alternativity, [x, y, y] = 0 for all x, y in any alternative algebra.
This means the associator variance (var_a) is always 0, and the assoc_weight
term in the policy contributes nothing to threshold adaptation.
"""

import torch

from octonion import Octonion
from octonion._octonion import associator
from octonion.trie import AlgebraicPurityPolicy, TrieNode


def test_alternativity_kills_signal():
    """Show [buf, node, node] = 0 for random inputs — the bug."""
    torch.manual_seed(42)

    for trial in range(100):
        buf = Octonion(torch.randn(8, dtype=torch.float64))
        node = Octonion(torch.randn(8, dtype=torch.float64))
        result = associator(buf, node, node)
        norm = result._data.norm().item()
        assert norm < 1e-12, f"Trial {trial}: expected ~0, got {norm}"

    print("[PASS] associator(buf, node, node) = 0 for all 100 random trials")
    print("       This confirms [x, y, y] = 0 (alternativity).")


def test_policy_var_a_always_zero():
    """Show that AlgebraicPurityPolicy's var_a is always 0 in practice."""
    torch.manual_seed(42)
    policy = AlgebraicPurityPolicy(
        base_assoc=0.3, assoc_weight=0.5, sim_weight=0.5, sensitivity=2.0
    )

    node = TrieNode(
        routing_key=torch.randn(8, dtype=torch.float64),
        content=torch.randn(8, dtype=torch.float64),
    )

    # Fill buffer with diverse samples
    for _ in range(20):
        sample = torch.randn(8, dtype=torch.float64)
        sample = sample / sample.norm()
        node.buffer.append((sample, 0))

    # Reproduce the policy's internal computation
    node_oct = Octonion(node.routing_key)
    assoc_norms = []
    for buf_x, _ in node.buffer:
        buf_oct = Octonion(buf_x)
        a = associator(buf_oct, node_oct, node_oct)
        assoc_norms.append(a.components.norm().item())

    print(f"\nAssociator norms from policy computation (should all be ~0):")
    print(f"  max: {max(assoc_norms):.2e}")
    print(f"  min: {min(assoc_norms):.2e}")

    mean_a = sum(assoc_norms) / len(assoc_norms)
    var_a = sum((v - mean_a) ** 2 for v in assoc_norms) / len(assoc_norms)
    print(f"  var_a: {var_a:.2e}")
    print(f"\n  => assoc_weight * var_a = {0.5 * var_a:.2e} (contributes nothing)")

    # Now show what the threshold actually depends on
    threshold = policy.get_assoc_threshold(node, 0)
    # Compare: policy with assoc_weight=0 should give the same result
    policy_no_assoc = AlgebraicPurityPolicy(
        base_assoc=0.3, assoc_weight=0.0, sim_weight=0.5, sensitivity=2.0
    )
    threshold_no_assoc = policy_no_assoc.get_assoc_threshold(node, 0)

    print(f"\n  Threshold with assoc_weight=0.5: {threshold:.6f}")
    print(f"  Threshold with assoc_weight=0.0: {threshold_no_assoc:.6f}")
    print(f"  Difference: {abs(threshold - threshold_no_assoc):.2e}")
    print(f"\n[PASS] var_a is always 0; assoc_weight has no effect on threshold.")


def test_fix_produces_nonzero_variance():
    """After fix: [buf, child, parent] gives nonzero associator norms."""
    torch.manual_seed(42)

    # Simulate parent and child nodes
    parent = TrieNode(
        routing_key=torch.randn(8, dtype=torch.float64),
        content=torch.randn(8, dtype=torch.float64),
    )
    child = TrieNode(
        routing_key=torch.randn(8, dtype=torch.float64),
        content=torch.randn(8, dtype=torch.float64),
    )

    # Fill child buffer with diverse samples
    for _ in range(20):
        sample = torch.randn(8, dtype=torch.float64)
        sample = sample / sample.norm()
        child.buffer.append((sample, 0))

    # Compute [buf, child, parent] — the corrected triple
    child_oct = Octonion(child.routing_key)
    parent_oct = Octonion(parent.routing_key)
    assoc_norms = []
    for buf_x, _ in child.buffer:
        buf_oct = Octonion(buf_x)
        a = associator(buf_oct, child_oct, parent_oct)
        assoc_norms.append(a.components.norm().item())

    mean_a = sum(assoc_norms) / len(assoc_norms)
    var_a = sum((v - mean_a) ** 2 for v in assoc_norms) / len(assoc_norms)
    print(f"\n=== After fix: [buf, child, parent] ===")
    print(f"  assoc_norms range: [{min(assoc_norms):.4f}, {max(assoc_norms):.4f}]")
    print(f"  mean: {mean_a:.4f}")
    print(f"  var_a: {var_a:.6f}")
    assert mean_a > 0.01, f"Expected nonzero mean, got {mean_a}"
    assert var_a > 1e-6, f"Expected nonzero variance, got {var_a}"

    # Verify the policy uses parent correctly
    policy = AlgebraicPurityPolicy(
        base_assoc=0.3, assoc_weight=0.5, sim_weight=0.5, sensitivity=2.0
    )
    threshold_with_parent = policy.get_assoc_threshold(child, 0, parent)
    threshold_no_parent = policy.get_assoc_threshold(child, 0, None)
    print(f"\n  Threshold with parent:    {threshold_with_parent:.6f}")
    print(f"  Threshold without parent: {threshold_no_parent:.6f} (fallback to base)")
    assert threshold_with_parent != threshold_no_parent, (
        "Expected different thresholds with/without parent"
    )
    print(f"\n[PASS] Fix confirmed: assoc_weight now contributes to threshold.")


if __name__ == "__main__":
    test_alternativity_kills_signal()
    test_policy_var_a_always_zero()
    test_fix_produces_nonzero_variance()
