---
status: resolved
trigger: "a*inv shows 1.0 with ~1.5e-08 residuals on imaginary components instead of clean identity. float32 precision noise visible in output."
created: 2026-03-08T07:00:00Z
updated: 2026-03-08T07:30:00Z
---

## Current Focus

hypothesis: CONFIRMED -- two co-occurring root causes: (1) float32 einsum accumulation loses precision, (2) __str__ displays all non-zero values including float32 noise
test: ran float32 vs float64 comparison and inspected __str__ threshold
expecting: float32 produces ~1.5e-08 residuals (inherent to dtype), __str__ shows them because threshold is val != 0.0
next_action: return diagnosis

## Symptoms

expected: a * a.inverse() should return identity [1,0,0,0,0,0,0,0] with negligible residuals
actual: a*inv shows 1.0 with ~1.5e-08 residuals on imaginary components
errors: No error, cosmetic precision noise
reproduction: `Octonion(torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.])) * _.inverse()` -- float32 default
started: Discovered during UAT test 4

## Eliminated

- hypothesis: "STRUCTURE_CONSTANTS float64->float32 downcast is the primary error source"
  evidence: "The inverse computation itself also introduces ~1.7e-9 error per component (1/204 is not exactly representable in float32). Even with perfect structure constants, the 8-term summation per output component accumulates float32 rounding. The downcast adds negligible error since C values are exactly {-1, 0, +1} which are all perfectly representable in float32."
  timestamp: 2026-03-08T07:03:00Z

## Evidence

- timestamp: 2026-03-08T07:01:00Z
  checked: STRUCTURE_CONSTANTS dtype at module level
  found: "dtype=torch.float64, values are {-1, 0, +1}"
  implication: "Downcast to float32 is lossless for these values -- {-1, 0, +1} are exact in any float format"

- timestamp: 2026-03-08T07:02:00Z
  checked: "float32 vs float64 a*inv result"
  found: "float32 max residual: ~1.86e-08. float64 max residual: ~4.16e-17. These are consistent with respective machine epsilon (float32 eps ~1.2e-7, float64 eps ~2.2e-16)."
  implication: "The residuals are inherent to float32 accumulation arithmetic, not a bug in the algebra implementation"

- timestamp: 2026-03-08T07:02:30Z
  checked: "inverse computation precision"
  found: "inv32 differs from exact by ~1.68e-9 per component (1/204 not exactly representable). norm_sq=204.0 is exact."
  implication: "Small error enters at inverse, then the 8-term einsum sum amplifies it to ~1.5e-08 range"

- timestamp: 2026-03-08T07:03:00Z
  checked: "Octonion.__str__ display threshold"
  found: "Line 234: `elif val != 0.0` -- displays any non-zero component, no tolerance threshold"
  implication: "Even ~1e-08 noise is displayed with full scientific notation, making output look broken when it is actually within normal float32 precision"

- timestamp: 2026-03-08T07:04:00Z
  checked: "upcast strategy (compute einsum in float64 with float32 inputs)"
  found: "Still produces ~1e-08 residuals because the error is baked into the float32 inverse values before the einsum"
  implication: "Cannot fix by upcasting only the einsum; the inverse computation itself (conj/norm_sq division) already has float32 quantization"

## Resolution

root_cause: "Two co-occurring issues: (1) INHERENT FLOAT32 PRECISION -- Computing a*a.inverse() in float32 produces ~1.5e-08 residuals. This is NOT a bug; it is the expected precision limit of float32 arithmetic. The inverse involves dividing by norm_sq=204 which yields irrational float32 values, and the subsequent 8-term einsum accumulation amplifies rounding. (2) DISPLAY BUG -- Octonion.__str__ at line 234 uses `val != 0.0` as the threshold for displaying components, so float32 noise on the order of 1e-08 is rendered in full scientific notation (e.g., '- 1.4901161193847656e-08*e1'), making the output appear broken when it is mathematically correct to float32 precision."
fix:
verification:
files_changed: []
