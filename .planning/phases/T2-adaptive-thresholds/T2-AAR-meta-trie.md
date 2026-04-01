# Phase T2: After Action Report — Meta-Trie Optimizer Design Failure

**Date:** 2026-04-01
**Scope:** Investigation into why MetaTriePolicy produces results identical to GlobalPolicy despite the on_insert infrastructure fix.

## Observation

After fixing the `float("inf")` poisoning (T2-AAR.md), EMA and other per-node policies showed dramatic improvement (EMA: 99.7% vs Global: 26.0% on structured data). But MetaTriePolicy remained at 25.7-26.0% — identical to Global. The meta-trie's hyperparameters (signal encoding, feedback signal, update frequency, self-referential mode) had zero effect on accuracy.

## Root Causes

### RC-1: Insert-then-query circularity (FATAL)

The core feedback loop in `_update_thresholds` is circular:

```python
action_cat = self._compute_stability_signal(trigger_node)  # Heuristic decides action
self.meta_trie.insert(meta_input, category=action_cat)     # Insert heuristic's decision
leaf = self.meta_trie.query(meta_input)                    # Query same input
recommended = leaf.dominant_category                        # Get back heuristic's decision
```

The meta-trie inserts a sample labeled with the action that `_compute_stability_signal` already computed, then immediately queries the same input and gets that label back. **The meta-trie is an echo chamber** — it never learns anything the stability heuristic doesn't already know.

The intended design (from discuss-phase D-12 through D-19) was a feedback loop:
1. Observe classifier state → encode as octonion
2. Meta-trie recommends threshold adjustment
3. Apply adjustment → observe outcome (did stability/accuracy improve?)
4. Feed outcome back to meta-trie as the training signal

The implementation collapsed steps 2-4 into "compute action from heuristic, echo it through the trie." There is no outcome observation, no reinforcement, no learning.

### RC-2: Only trigger node gets adjusted (SIGNIFICANT)

`_update_thresholds` only adjusts the single node that happened to trigger the update frequency counter. With `update_frequency=100` and 350 samples × 3 epochs, only ~10 update events occur per epoch. Each updates ONE node.

The meta-trie was supposed to **generalize** — learn that "nodes with profile X benefit from threshold Y" and apply that to ALL similar nodes. Instead, it's a per-node lookup table that covers less than half the trie (143 of 315 nodes in our test).

### RC-3: Stability heuristic maps mostly to "keep" (MODERATE)

The CV-based heuristic in `_compute_stability_signal` maps the coefficient of variation of associator norms to one of 5 actions. In practice, 47% of nodes fall in the "keep" range (CV 0.2-0.5), producing adjustment=0.0. The distribution:

| CV Range | Action | Count | Percentage |
|----------|--------|-------|------------|
| < 0.1 | loosen 20% | 10 | 3.6% |
| 0.1-0.2 | loosen 10% | 5 | 1.8% |
| 0.2-0.5 | keep | 131 | 47.0% |
| 0.5-1.0 | tighten 10% | 90 | 32.3% |
| >= 1.0 | tighten 20% | 43 | 15.4% |

Even when the heuristic recommends a non-zero action, the echo chamber just replays it without learning whether it helped.

### RC-4: No temporal credit assignment (FUNDAMENTAL)

The meta-trie has no mechanism to attribute outcomes to past threshold changes. When a node's accuracy improves after a threshold adjustment, the meta-trie doesn't know to reinforce that adjustment. When accuracy degrades, it doesn't know to reverse course. The `feedback_signal="accuracy"` mode was supposed to address this, but the implementation just returns `action_cat=2` ("keep") as a placeholder:

```python
if self.feedback_signal == "stability":
    action_cat = self._compute_stability_signal(trigger_node)
else:
    action_cat = 2  # "keep" for accuracy mode (set externally)
```

The "accuracy" feedback path was never implemented.

## Design vs Implementation Gap

| Aspect | Discussed Design (D-12 to D-19) | Actual Implementation |
|--------|--------------------------------|----------------------|
| Meta-trie role | Learns optimal thresholds from classifier feedback | Echoes a hardcoded CV heuristic |
| Feedback loop | Observe state → adjust → measure outcome → learn | Observe state → compute heuristic → echo |
| Generalization | Learn "node profile X → threshold Y" for all nodes | Per-node lookup table for trigger nodes only |
| Accuracy feedback (D-15) | Evaluate classifier on held-out set, feed back | Placeholder `action_cat = 2` |
| Categories (D-13) | Discretized threshold actions discovered by learning | Hardcoded CV thresholds that map to actions |
| Environment feedback | Classifier's response to adjustments | Instantaneous CV measurement (no temporal lag) |

## Correct Design Requirements

A faithful meta-trie optimizer needs:

1. **Temporal feedback loop:** Apply adjustment → wait for samples to route through the adjusted node → measure outcome → insert (state, outcome) into meta-trie
2. **Outcome measurement:** Compare node's accuracy/stability BEFORE vs AFTER each adjustment
3. **Generalization:** Query the meta-trie for ALL nodes at update time, not just the trigger node. The meta-trie should recognize "this node looks like others where tightening helped"
4. **Exploration/exploitation:** Sometimes try adjustments the heuristic wouldn't suggest, to discover whether the meta-trie can learn better-than-heuristic policies
5. **Accuracy feedback path:** Actually evaluate classifier accuracy and use it as the meta-trie's training signal

## Relationship to AAR #1

This is a DESIGN failure, not an infrastructure failure. The on_insert fix (T2-AAR.md) fixed broken plumbing — data wasn't flowing through the pipes. This AAR identifies that the pipes don't connect to the right places — the meta-trie's learning loop is fundamentally absent, not just broken.

The EMA policy works well (99.7%) because it has a correct feedback loop: observe assoc_norms → compute running average → adapt threshold. It's simple but the feedback is direct and real. The meta-trie was supposed to be the sophisticated version of this, but the implementation replaced real feedback with a heuristic echo chamber.
