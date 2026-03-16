---
status: diagnosed
trigger: "profile_baseline.py --no-traces --iters 2 --warmup 1 shows Real baseline with 59,471,626 params instead of expected ~3.7M"
created: 2026-03-16T00:00:00Z
updated: 2026-03-16T00:00:00Z
---

## Current Focus

hypothesis: profile_baseline.py sets config.base_hidden = ref_hidden (4) AFTER cifar_network_config() returns a config with base_hidden=16. AlgebraNetwork._build_conv() then multiplies base_hidden by algebra.multiplier (which is 8 for REAL), so Real gets base_filters = 4 * 8 = 32 instead of the expected 4 * 1 = 4. But the multiplier design is for param-MATCHING mode, not same-width mode -- same-width should bypass the multiplier.
test: Trace the exact integer arithmetic for Real algebra with ref_hidden=4
expecting: Real base_filters = 4 * 8 = 32, stage filters = [32, 64, 128], which produces ~59M params
next_action: DIAGNOSED -- return root cause

## Symptoms

expected: Real baseline ~3.7M params when running profile_baseline.py --no-traces --iters 2 --warmup 1 with ref_hidden=4
actual: Real baseline 59,471,626 params (~59.5M)
errors: No error thrown -- just wrong param count
reproduction: Run profile_baseline.py --no-traces --iters 2 --warmup 1
started: Present in current codebase

## Eliminated

- hypothesis: cifar_network_config() doesn't get updated with ref_hidden
  evidence: Line 142 of profile_baseline.py does set config.base_hidden = ref_hidden after the call, so base_hidden is correctly set to 4
  timestamp: 2026-03-16T00:00:00Z

- hypothesis: ref_hidden arg is not passed through to build_profile_model
  evidence: profile_algebra() passes ref_hidden=args.ref_hidden correctly, build_profile_model() receives it and sets config.base_hidden = ref_hidden
  timestamp: 2026-03-16T00:00:00Z

## Evidence

- timestamp: 2026-03-16T00:00:00Z
  checked: AlgebraType enum in _config.py
  found: REAL = ("R", 1, 8) -- multiplier is 8. OCTONION = ("O", 8, 1) -- multiplier is 1.
  implication: The multiplier encodes "Real needs 8x width to match octonion param count". This is correct for param-MATCHING mode.

- timestamp: 2026-03-16T00:00:00Z
  checked: AlgebraNetwork.__init__ in _network.py line 185
  found: self.hidden = config.base_hidden * config.algebra.multiplier
  implication: For ALL topologies, including conv2d, the hidden width is always scaled by the algebra multiplier. There is no bypass for same-width mode.

- timestamp: 2026-03-16T00:00:00Z
  checked: AlgebraNetwork._build_conv() in _network.py line 304
  found: base_filters = config.base_hidden * config.algebra.multiplier
  implication: For conv2d, base_filters also scales by multiplier. Real with base_hidden=4 → base_filters = 4*8 = 32. Stage filters = [32, 64, 128]. This is a much wider network than intended.

- timestamp: 2026-03-16T00:00:00Z
  checked: Expected param count math for Real with base_filters=32, depth=28
  found: 3 stages, filters [32, 64, 128], 28 residual blocks, each block has 2x conv3x3. Conv3x3 with 128→128 channels = 128*128*9 = 147,456 weights per conv, plus BN. This easily reaches ~59M params.
  implication: Confirms that multiplier=8 on base_hidden=4 for Real is causing the explosion.

- timestamp: 2026-03-16T00:00:00Z
  checked: Expected param count math for Octonion with base_hidden=4
  found: Octonion multiplier=1, so base_filters = 4*1 = 4. Stage filters = [4, 8, 16]. Octonion conv layers store weights as [out_ch, in_ch, dim, dim, kH, kW] -- 8x8=64 real params per algebra unit pair. So actual real params per layer = out_ch * in_ch * 64 * kH * kW. With base 4 units this stays small (~495K). Real at 4 units with multiplier bypassed should be similar but smaller (1 real param per unit pair = 1x vs 64x).
  implication: The ~3.7M expected for Real at same-width is plausible IF Real runs with base_filters=4 (no multiplier), not 32.

- timestamp: 2026-03-16T00:00:00Z
  checked: profile_baseline.py build_profile_model() function (lines 124-144)
  found: Does NOT use find_matched_width. Does NOT bypass the multiplier. Simply sets config.base_hidden = ref_hidden and passes to AlgebraNetwork -- which then applies the multiplier internally.
  implication: Same-width intent (all algebras use ref_hidden=4 as their channel count) is violated because AlgebraNetwork always multiplies by algebra.multiplier.

- timestamp: 2026-03-16T00:00:00Z
  checked: run_cifar_reproduction.py network_overrides dict
  found: Passes "match_params" and "ref_hidden" to run_comparison(). That path uses find_matched_width() which properly accounts for the multiplier via binary search. The profile script skips this entirely.
  implication: The reproduction script handles param matching correctly. The profile script tries to use same-width but doesn't account for the multiplier built into AlgebraNetwork.

## Resolution

root_cause: >
  In build_profile_model() (profile_baseline.py:142), setting config.base_hidden = ref_hidden (4) and
  passing it to AlgebraNetwork does NOT produce same-width behavior. AlgebraNetwork.__init__
  (line 185 of _network.py) computes self.hidden = config.base_hidden * config.algebra.multiplier,
  and _build_conv() (line 304) computes base_filters = config.base_hidden * config.algebra.multiplier.
  For REAL, multiplier=8, so base_filters = 4*8 = 32 instead of 4. Stage filters become [32, 64, 128]
  instead of [4, 8, 16], producing ~59M params. The multiplier is designed for param-MATCHING mode
  (scale Real up to match Octonion param count), but same-width mode requires all algebras to use the
  same raw base_filters without the multiplier.
fix: empty
verification: empty
files_changed: []
