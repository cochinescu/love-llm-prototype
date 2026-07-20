# R1-L larger characterization grid (scripts/reproduce_real_v3.py — PROTOCOL 'R1-L')

Pinned base model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` · decoding: temp 0.5, top_p 0.95 · master seed `20260712` · 8 fresh personas (pool seed 23037712) · 6 sessions × 8 turns · 6 seeds/arm · 2 paired generations · mechanism R1 (unchanged) · frozen R1 prediction verified

## Decision statistics with run-level bootstrap 90% CIs (2000 draws, paired by seed)

| statistic | point | 90% CI | bar | interval clears bar |
|---|---|---|---|---|
| full final D | 0.301 | [0.251, 0.332] | ≥ 0.15 | YES |
| ordering margin (full−mismatched) | 0.095 | [0.018, 0.110] | ≥ 0.02 | point only |
| full monotone fraction | 1.000 | [0.600, 1.000] | ≥ 0.80 | point only |
| accuracy contrast (full−empty) | 0.583 | [0.250, 0.833] | ≥ 0.1 | YES |
| TOST quality diff (full−base) | 0.0217 | [0.0117, 0.0328] | within ±0.05 | YES |

## Curve vs frozen R1 prediction (full arm)

- predicted: [0.116, 0.217, 0.289, 0.341, 0.378, 0.405]
- measured median: [0.066, 0.135, 0.167, 0.231, 0.251, 0.301]
- rho 1.000; monotone 1.000; ratio 0.768 (pred 0.840 ± 0.20); curvature saturating — **shape agreement: YES** (level agreement NOT claimed: measured final 0.301 vs predicted 0.405)

## Channel decomposition at final depth (full vs mismatched-R)

| channel | full | mismatched-R | person-specific delta |
|---|---|---|---|
| initiative_rate | 0.500 | 0.500 | +0.000 |
| response_depth | 0.229 | 0.221 | +0.008 |
| reference_rate | 0.125 | 0.000 | +0.125 |
| pref_alignment | 0.268 | 0.101 | +0.167 |

## Ablations (protocol-defined point statistics)

| arm | final D | rho | monotone | acc(final) | curve | accuracy | markers | JOINT |
|---|---|---|---|---|---|---|---|---|
| full | 0.301 | 1.000 | 1.00 | 0.67 | True | True | True | **PASS** |
| empty_r | 0.000 | 0.000 | 1.00 | 0.08 | False | False | False | **FAIL** |
| mismatched_r | 0.206 | 0.943 | 0.80 | 0.17 | True | False | False | **FAIL** |
| caring_only | 0.256 | 1.000 | 1.00 | 0.08 | True | False | True | **FAIL** |
| particularity_only | 0.052 | 0.657 | 0.60 | 0.67 | False | True | True | **FAIL** |
| generic_personalization | 0.029 | 0.143 | 0.40 | 0.08 | False | False | False | **FAIL** |

## Protocol checks

- Floor: Δ = 0.0000 vs 0.05 → **PASS**
- Padding sensitivity: mean 0.0357, max 0.1390
- Marker naturalism: 0.0148 in (0.0, 0.08) → PASS

Total wall time: 2748s
