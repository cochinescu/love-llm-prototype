# REAL-MODEL grid v2 — mechanism revision R1 'probe-guarded partiality' (scripts/reproduce_real_v2.py)

Pinned base model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` · decoding: temp 0.5, top_p 0.95 · seed `20260712` · 4 personas · 6 sessions × 8 turns · 4 seeds/arm · 2 paired generations/(probe,condition)

R1 (recorded in PROTOCOL 'W2 revision R1'): guarded
conditioning mix 0.75/0.15/0.10 + marker cap 1 on probe responses; the
quality instrument, TOST rule, margins, arms, and all other constants are
IDENTICAL to the v1 grid. The R1 analytic prediction was re-derived and
hash-frozen before any generation (sha256 `0149dfad824bed7b…`).

## Divergence vs the frozen R1 prediction (full arm)

- predicted D over depths [8, 16, 24, 32, 40, 48]: [0.116, 0.217, 0.289, 0.341, 0.378, 0.405]
- measured median D: [0.09, 0.138, 0.166, 0.219, 0.252, 0.248]
- Spearman rho: 0.943 (bar 0.8); monotone 0.800 (bar 0.80); ratio 0.886 (predicted 0.840 ± 0.20); curvature saturating (predicted saturating)
- **agreement: YES**

## Ablations (protocol-defined joint signature)

| arm | final D | rho | monotone | acc(final) | curve | accuracy | markers | JOINT |
|---|---|---|---|---|---|---|---|---|
| full | 0.248 | 0.943 | 0.80 | 0.50 | True | True | True | **PASS** |
| empty_r | 0.000 | 0.000 | 1.00 | 0.00 | False | False | False | **FAIL** |
| mismatched_r | 0.221 | 1.000 | 1.00 | 0.38 | True | True | False | **FAIL** |
| caring_only | 0.246 | 0.943 | 0.80 | 0.00 | True | False | True | **FAIL** |
| particularity_only | 0.023 | -0.200 | 0.60 | 0.50 | False | True | True | **FAIL** |
| generic_personalization | 0.007 | -0.371 | 0.40 | 0.00 | False | False | False | **FAIL** |

## Protocol checks

- **Floor:** Δ = 0.0000 vs margin 0.05 → **PASS**
- **Ordering:** full 0.248 > mismatched 0.221 > empty 0.000 (margin 0.02) → **PASS**
- **Padding sensitivity:** mean |Δq| = 0.0374, max = 0.1390
- **Marker naturalism:** density 0.0125 in band (0.0, 0.08) → PASS
- **Capability equivalence (TOST, margin ±0.05):** mean quality diff (full − base) = 0.0094, 90% CI [-0.0119, 0.0307] → **EQUIVALENT**

Total wall time: 1584s
