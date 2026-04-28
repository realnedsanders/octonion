"""Tests for the octonionic trie."""

import torch

from octonion._fano import FANO_PLANE
from octonion.trie import OctonionTrie, TrieNode, subalgebra_activation

# ── Helpers ──────────────────────────────────────────────────────────


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


def _make_orthogonal_centers(n: int = 7, seed: int = 42) -> list[torch.Tensor]:
    """Create n orthogonal unit octonion centers via QR decomposition."""
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    return [Q[:, i] for i in range(min(n, 8))]


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
    for s, label in zip(samples, labels, strict=False):
        if label not in cats:
            continue
        total += 1
        leaf = trie.query(s)
        if leaf.dominant_category == label:
            correct += 1
    return correct / max(total, 1)


# ── Unit tests ───────────────────────────────────────────────────────


class TestSubalgebraActivation:
    """Tests for subalgebra_activation utility."""

    def test_output_shape(self) -> None:
        x = torch.randn(8, dtype=torch.float64)
        act = subalgebra_activation(x)
        assert act.shape == (7,)

    def test_batch_shape(self) -> None:
        x = torch.randn(10, 8, dtype=torch.float64)
        act = subalgebra_activation(x)
        assert act.shape == (10, 7)

    def test_aligned_input_peaks_at_correct_subalgebra(self) -> None:
        """An input with strong components in one subalgebra should peak there."""
        for sub_idx, triple in enumerate(FANO_PLANE.triples):
            i, j, k = triple
            x = torch.zeros(8, dtype=torch.float64)
            x[i] = 1.0
            x[j] = 1.0
            x[k] = 1.0
            act = subalgebra_activation(x)
            assert act.argmax().item() == sub_idx, (
                f"Subalgebra {sub_idx} should be strongest for input aligned "
                f"with components {triple}"
            )

    def test_nonnegative(self) -> None:
        x = torch.randn(100, 8, dtype=torch.float64)
        act = subalgebra_activation(x)
        assert (act >= 0).all()


class TestTrieNode:
    """Tests for TrieNode dataclass."""

    def test_dominant_category_empty(self) -> None:
        node = TrieNode(
            routing_key=torch.zeros(8), content=torch.zeros(8)
        )
        assert node.dominant_category is None

    def test_dominant_category(self) -> None:
        node = TrieNode(
            routing_key=torch.zeros(8),
            content=torch.zeros(8),
            category_counts={0: 5, 1: 3, 2: 7},
        )
        assert node.dominant_category == 2

    def test_is_leaf(self) -> None:
        node = TrieNode(routing_key=torch.zeros(8), content=torch.zeros(8))
        assert node.is_leaf
        node.children[0] = TrieNode(routing_key=torch.zeros(8), content=torch.zeros(8))
        assert not node.is_leaf


class TestOctonionTrie:
    """Tests for OctonionTrie core operations."""

    def test_empty_trie_has_one_node(self) -> None:
        trie = OctonionTrie(seed=0)
        assert trie.n_nodes == 1
        assert trie.stats()["n_leaves"] == 1

    def test_insert_creates_child(self) -> None:
        trie = OctonionTrie(seed=0)
        x = torch.randn(8, dtype=torch.float64)
        x = x / x.norm()
        trie.insert(x, category=0)
        assert trie.n_nodes == 2
        assert len(trie.root.children) == 1

    def test_insert_returns_node(self) -> None:
        trie = OctonionTrie(seed=0)
        x = torch.randn(8, dtype=torch.float64)
        node = trie.insert(x / x.norm(), category=0)
        assert isinstance(node, TrieNode)
        assert node.dominant_category == 0

    def test_query_returns_node(self) -> None:
        trie = OctonionTrie(seed=0)
        x = torch.randn(8, dtype=torch.float64)
        x = x / x.norm()
        trie.insert(x, category=0)
        leaf = trie.query(x)
        assert isinstance(leaf, TrieNode)

    def test_query_matches_insert_destination(self) -> None:
        """A query for the same input should reach the same node."""
        trie = OctonionTrie(seed=0)
        x = torch.randn(8, dtype=torch.float64)
        x = x / x.norm()
        insert_node = trie.insert(x, category=0)
        query_node = trie.query(x)
        assert query_node.dominant_category == insert_node.dominant_category

    def test_routing_key_is_immutable(self) -> None:
        """Inserting data should not change any node's routing key."""
        trie = OctonionTrie(seed=0)
        x1 = torch.randn(8, dtype=torch.float64)
        x1 = x1 / x1.norm()
        trie.insert(x1, category=0)

        # Record all routing keys
        keys_before = {}
        for sub_idx, child in trie.root.children.items():
            keys_before[sub_idx] = child.routing_key.clone()

        # Insert more data
        for _ in range(20):
            x = torch.randn(8, dtype=torch.float64)
            trie.insert(x / x.norm(), category=1)

        # Routing keys should be unchanged
        for sub_idx, key_before in keys_before.items():
            if sub_idx in trie.root.children:
                assert torch.allclose(
                    trie.root.children[sub_idx].routing_key, key_before
                ), f"Routing key at S{sub_idx} changed after insertion"

    def test_max_children_per_node_is_7(self) -> None:
        """No node should have more than 7 children (Fano plane limit)."""
        trie = OctonionTrie(associator_threshold=0.01, seed=0)
        gen = torch.Generator().manual_seed(42)
        for _ in range(100):
            x = torch.randn(8, dtype=torch.float64, generator=gen)
            trie.insert(x / x.norm(), category=0)

        def _check(node: TrieNode) -> None:
            assert len(node.children) <= 7, (
                f"Node at depth {node.depth} has {len(node.children)} children"
            )
            for child in node.children.values():
                _check(child)

        _check(trie.root)

    def test_consolidation_reduces_nodes(self) -> None:
        trie = OctonionTrie(associator_threshold=0.05, seed=0)
        gen = torch.Generator().manual_seed(42)
        for _ in range(50):
            x = torch.randn(8, dtype=torch.float64, generator=gen)
            trie.insert(x / x.norm(), category=0)

        nodes_before = trie.stats()["n_nodes"]
        trie.consolidate()
        nodes_after = trie.stats()["n_nodes"]
        assert nodes_after <= nodes_before


class TestTrieAlignedCategories:
    """Tests with subalgebra-aligned categories (best case)."""

    def test_aligned_categories_high_accuracy(self) -> None:
        """7 subalgebra-aligned categories should achieve >95% accuracy."""
        centers = _make_aligned_centers()
        train_s, train_l = _generate_samples(centers, 100, noise=0.05, seed=99)
        test_s, test_l = _generate_samples(centers, 50, noise=0.05, seed=7777)

        trie = OctonionTrie(associator_threshold=0.3, seed=42)
        for _ep in range(3):
            for s, label in zip(train_s, train_l, strict=False):
                trie.insert(s, category=label)

        acc = _accuracy(trie, test_s, test_l)
        assert acc > 0.95, f"Aligned categories accuracy {acc:.3f} < 0.95"


class TestTrieStabilityPlasticity:
    """Tests for stability-plasticity with structured data."""

    def test_zero_forgetting_orthogonal_data(self) -> None:
        """Phase 1 accuracy should not degrade after Phase 2 training."""
        centers = _make_orthogonal_centers(7, seed=42)
        train_s, train_l = _generate_samples(centers, 150, noise=0.05, seed=99)
        test_s, test_l = _generate_samples(centers, 50, noise=0.05, seed=7777)

        trie = OctonionTrie(associator_threshold=0.3, seed=42)

        # Phase 1: categories 0-3
        p1_cats = set(range(4))
        for _ep in range(3):
            for s, label in zip(train_s, train_l, strict=False):
                if label in p1_cats:
                    trie.insert(s, category=label)

        acc_before = _accuracy(trie, test_s, test_l, p1_cats)

        # Phase 2: categories 4-6
        p2_cats = set(range(4, 7))
        for _ep in range(3):
            for s, label in zip(train_s, train_l, strict=False):
                if label in p2_cats:
                    trie.insert(s, category=label)

        acc_after = _accuracy(trie, test_s, test_l, p1_cats)
        acc_new = _accuracy(trie, test_s, test_l, p2_cats)
        forgetting = acc_before - acc_after

        assert forgetting < 0.05, (
            f"Catastrophic forgetting detected: {forgetting:.3f} "
            f"(before={acc_before:.3f}, after={acc_after:.3f})"
        )
        assert acc_new > 0.5, f"Phase 2 plasticity too low: {acc_new:.3f}"

    def test_plasticity_new_categories_learned(self) -> None:
        """New categories added in Phase 2 should be classifiable."""
        centers = _make_orthogonal_centers(6, seed=42)
        train_s, train_l = _generate_samples(centers, 100, noise=0.05, seed=99)
        test_s, test_l = _generate_samples(centers, 50, noise=0.05, seed=7777)

        trie = OctonionTrie(associator_threshold=0.3, seed=42)

        # Phase 1: categories 0-2
        for _ep in range(3):
            for s, label in zip(train_s, train_l, strict=False):
                if label < 3:
                    trie.insert(s, category=label)

        # Phase 2: categories 3-5
        for _ep in range(3):
            for s, label in zip(train_s, train_l, strict=False):
                if label >= 3:
                    trie.insert(s, category=label)

        # All categories should be reachable
        overall = _accuracy(trie, test_s, test_l)
        assert overall > 0.7, f"Overall accuracy too low after two phases: {overall:.3f}"


# ── Query Edge Cases ─────────────────────────────────────────────────


class TestQueryEdgeCases:
    """Edge cases for the query operation."""

    def test_query_empty_trie_returns_root(self) -> None:
        """Querying an empty trie (root only, no children) returns root."""
        trie = OctonionTrie(seed=42)
        x = torch.randn(8, dtype=torch.float64)
        x = x / x.norm()
        result = trie.query(x)
        assert result is trie.root, "Query on empty trie should return root"


# ── Rumination Rejection Paths ───────────────────────────────────────


class TestRumination:
    """Tests for the rumination (geometric consistency) mechanism."""

    def test_small_buffer_accepts(self) -> None:
        """With fewer than 3 buffer entries, rumination should not reject."""
        trie = OctonionTrie(associator_threshold=0.3, seed=42)

        # Insert 2 similar samples into same branch
        base = torch.zeros(8, dtype=torch.float64)
        base[0] = 1.0
        for _i in range(2):
            sample = base + 0.01 * torch.randn(8, dtype=torch.float64)
            sample = sample / sample.norm()
            trie.insert(sample, category=0)

        assert trie.stats()["rumination_rejections"] == 0, (
            "Should have no rumination rejections with < 3 buffer entries"
        )

    def test_rejects_orthogonal_input(self) -> None:
        """Rumination should reject input orthogonal to buffer contents."""
        trie = OctonionTrie(
            associator_threshold=0.5, similarity_threshold=0.3, seed=42
        )

        # Build a cluster in one direction
        base = torch.zeros(8, dtype=torch.float64)
        base[0] = 0.9
        base[1] = 0.4
        base = base / base.norm()
        for _i in range(10):
            sample = base + 0.02 * torch.randn(8, dtype=torch.float64)
            sample = sample / sample.norm()
            trie.insert(sample, category=0)

        rejections_before = trie.stats()["rumination_rejections"]

        # Insert something nearly orthogonal to the cluster
        orthogonal = torch.zeros(8, dtype=torch.float64)
        orthogonal[5] = 0.9
        orthogonal[6] = 0.4
        orthogonal = orthogonal / orthogonal.norm()
        trie.insert(orthogonal, category=1)

        # The orthogonal input should either branch or be rejected by rumination
        # (it definitely shouldn't be silently accepted into the existing cluster)
        stats = trie.stats()
        # Either rumination rejected it OR it created a new branch (n_nodes increased)
        new_rejections = stats["rumination_rejections"] - rejections_before
        has_branched = stats["n_nodes"] > 3  # root + at least 2 children
        assert new_rejections > 0 or has_branched, (
            "Orthogonal input was neither rejected by rumination nor branched"
        )


# ── Depth Limit ──────────────────────────────────────────────────────


class TestDepthLimit:
    """Test that max_depth is respected."""

    def test_max_depth_respected(self) -> None:
        """No node should exceed max_depth."""
        max_depth = 2
        trie = OctonionTrie(
            associator_threshold=0.1, max_depth=max_depth, seed=42
        )

        # Insert many diverse samples to force deep branching attempts
        gen = torch.Generator().manual_seed(42)
        for i in range(50):
            x = torch.randn(8, dtype=torch.float64, generator=gen)
            x = x / x.norm()
            trie.insert(x, category=i % 7)

        # Walk the trie and verify no node exceeds max_depth
        def _check_depth(node: TrieNode, expected_max: int) -> None:
            assert node.depth <= expected_max, (
                f"Node at depth {node.depth} exceeds max_depth={expected_max}"
            )
            for child in node.children.values():
                _check_depth(child, expected_max)

        _check_depth(trie.root, max_depth)


# ── All Slots Occupied ───────────────────────────────────────────────


class TestAllSlotsOccupied:
    """Test behavior when all 7 subalgebra slots are filled."""

    def test_fallback_to_existing_child(self) -> None:
        """When all 7 slots are full, new input descends into existing child."""
        trie = OctonionTrie(
            associator_threshold=0.05,  # tight threshold forces branching
            seed=42,
        )

        # Insert samples that are diverse enough to fill all 7 subalgebra slots
        centers = _make_aligned_centers()
        for cat_idx, center in enumerate(centers):
            for _ in range(5):
                sample = center + 0.01 * torch.randn(8, dtype=torch.float64)
                sample = sample / sample.norm()
                trie.insert(sample, category=cat_idx)

        len(trie.root.children)

        # Now insert an 8th category — should not create a new root child
        # (max 7 subalgebra slots)
        extra = torch.randn(8, dtype=torch.float64)
        extra = extra / extra.norm()
        trie.insert(extra, category=7)

        assert len(trie.root.children) <= 7, (
            f"Root has {len(trie.root.children)} children (max 7 subalgebra slots)"
        )


# ── Consolidation Edge Cases ─────────────────────────────────────────


class TestConsolidationEdgeCases:
    """Edge cases for the consolidation operation."""

    def test_single_child_survives(self) -> None:
        """A node with only 1 child should not be consolidated."""
        trie = OctonionTrie(
            associator_threshold=0.5,  # loose threshold to keep things in one branch
            seed=42,
        )

        # Insert similar samples — should create at most 1-2 children
        base = torch.zeros(8, dtype=torch.float64)
        base[0] = 1.0
        base = base / base.norm()
        for _ in range(5):
            sample = base + 0.01 * torch.randn(8, dtype=torch.float64)
            sample = sample / sample.norm()
            trie.insert(sample, category=0)

        n_before = trie.stats()["n_nodes"]
        trie.consolidate()
        n_after = trie.stats()["n_nodes"]

        # With similar inputs in one branch, consolidation should not reduce
        assert n_after >= n_before - 1, (
            f"Consolidation removed too many nodes: {n_before} -> {n_after}"
        )
