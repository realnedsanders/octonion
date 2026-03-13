---
status: diagnosed
trigger: "Research and fix two issues in the octonion baselines benchmark code: conv depth deviation and slow test shape mismatch"
created: 2026-03-08T00:00:00Z
updated: 2026-03-08T00:00:00Z
---

## Current Focus

hypothesis: Two independent root causes found - see Resolution
test: n/a - research-only mode
expecting: n/a
next_action: Report findings

## Symptoms

expected: |
  1. CIFAR conv networks should use depth=28 per plan (matching ResNet-style architectures from published papers)
  2. Slow CIFAR reproduction tests should train conv2d AlgebraNetworks and pass
  3. pytest --timeout=0 should work for slow tests
actual: |
  1. depth was changed to 3 because MaxPool2d at every block reduces 32x32 to zero spatial dims
  2. Slow tests fail with: RuntimeError: mat1 and mat2 shapes cannot be multiplied (12288x32 and 3x64)
     Error at _param_matching.py:83 in forward, self.input_proj(x)
  3. --timeout=0 fails because pytest-timeout is not installed
errors: |
  RuntimeError: mat1 and mat2 shapes cannot be multiplied (12288x32 and 3x64)
reproduction: |
  Run: docker compose run --rm dev uv run pytest tests/test_baselines_reproduction.py -x -v -k "slow"
started: First implementation attempt

## Eliminated

(none)

## Evidence

- timestamp: 2026-03-08
  checked: AlgebraNetwork._build_conv (lines 174-206)
  found: MaxPool2d(2,2) is added at EVERY block (line 204). For depth=28 and 32x32 CIFAR, spatial dims halved 28 times -> 0.
  implication: depth=28 is impossible with current architecture. Executor was correct to reduce depth.

- timestamp: 2026-03-08
  checked: Published paper architectures (Gaudet & Maida 2018, Trabelsi et al. 2018)
  found: Both use ResNet-style architectures with 3 stages, skip connections, and stride-2 conv for downsampling (NOT MaxPool at every block). Gaudet uses 10+9+9 residual blocks. Standard CIFAR ResNet uses stride-2 conv at stage boundaries only (3 downsampling points total).
  implication: AlgebraNetwork's pooling-at-every-block design is fundamentally incompatible with deep CIFAR architectures.

- timestamp: 2026-03-08
  checked: run_comparison in _comparison.py (lines 167-461)
  found: run_comparison ALWAYS uses _build_simple_mlp (line 322) to build models, regardless of topology override. It builds an MLP even when network_config_overrides has topology="conv2d". The topology override is only stored for config logging (line 208), never used for model construction.
  implication: Slow tests pass topology="conv2d" but get an MLP with input_dim=3. CIFAR data is [B, 3, 32, 32] = 3072 flattened features, but MLP expects input_dim=3. Shape mismatch.

- timestamp: 2026-03-08
  checked: _SimpleAlgebraMLP (lines 23-98 in _param_matching.py)
  found: input_proj is nn.Linear(input_dim, hidden * algebra.dim). With input_dim=3, it expects [B, 3]. But CIFAR data is [B, 3, 32, 32]. The model receives a 4D tensor but expects 2D.
  implication: The error "mat1 and mat2 shapes cannot be multiplied (12288x32 and 3x64)" comes from passing [B, 3, 32, 32] into nn.Linear(3, hidden*dim). PyTorch reshapes to [B*3*32, 32] vs weight [3, 64], hence 12288x32 vs 3x64.

- timestamp: 2026-03-08
  checked: pyproject.toml pytest config
  found: No pytest-timeout in dependencies. addopts = "-x --tb=short". No timeout configuration.
  implication: --timeout=0 flag will fail because pytest-timeout plugin is not installed.

## Resolution

root_cause: |
  ISSUE 1 (Conv depth): AlgebraNetwork._build_conv applies MaxPool2d(2,2) at EVERY conv block (line 204).
  Published papers (Gaudet & Maida 2018, Trabelsi et al. 2018) use ResNet-style architectures with:
  - Skip connections (residual blocks)
  - Spatial downsampling via stride-2 convolutions at stage boundaries only (typically 3 stages)
  - NO MaxPool at every block
  - Global average pooling only at the end
  The depth=3 workaround makes the network functional but produces a very shallow network that cannot
  reproduce published results (which use 28+ residual blocks). This is a fundamental architecture gap.

  ISSUE 2 (Shape mismatch): run_comparison() ALWAYS builds _SimpleAlgebraMLP regardless of topology.
  The slow tests pass network_config_overrides={"topology": "conv2d", "depth": 28, "ref_hidden": 16}
  but run_comparison ignores topology and builds an MLP. The MLP's input_proj expects [B, input_dim=3]
  but receives [B, 3, 32, 32] CIFAR image tensors, causing the shape mismatch.

  ISSUE 3 (timeout): pytest-timeout is not installed. The plan's verification command uses --timeout=0
  which requires the pytest-timeout plugin.

fix: (not applied - research only)
verification: (not performed - research only)
files_changed: []
