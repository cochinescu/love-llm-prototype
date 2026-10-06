# R1-XL expanded grid + second probe battery (scripts/reproduce_real_v4.py — PROTOCOL 'R1-XL')

Pinned base model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` · decoding: temp 0.5, top_p 0.95 · master seed `20260712` · 16 runs/arm, each a distinct fresh focal persona (pool 20, seed 23148712) · 6 sessions × 8 turns · 2 paired generations · mechanism R1 (unchanged) · frozen R1 prediction verified (0149dfad824b…)

## Decision statistics (battery A) — decided on points; intervals DESCRIPTIVE (percentile, 2000 draws, 16 clusters)

| statistic | point | boot median | 90% interval | bar |
|---|---|---|---|---|
| A_full_final_D | 0.2303 | 0.2303 | [0.2112, 0.2370] | 0.15 |
| A_ordering_margin | 0.0228 | 0.0220 | [0.0052, 0.0371] | 0.02 |
| A_full_monotone | 0.8000 | 0.8000 | [0.8000, 1.0000] | 0.8 |
| A_accuracy_contrast | 0.5000 | 0.5000 | [0.3125, 0.6875] | 0.1 |
| A_tost_quality_diff | 0.0044 | - | [-0.0118, 0.0203] | ±0.05 |
| B_tost_quality_diff_heldout_probes | 0.0826 | - | [0.0741, 0.0904] | ±0.05 |
| B_ordering_margin_heldout_probes | 0.0198 | 0.0193 | [0.0022, 0.0262] | 0.02 |
| B_floor_delta_heldout_probes | 0.0000 | - | [-, -] | 0.05 |

## Curve vs frozen R1 prediction (battery A, full arm)

- predicted: [0.116, 0.217, 0.289, 0.341, 0.378, 0.405]
- measured median: [0.094, 0.137, 0.154, 0.205, 0.242, 0.23]
- rho 0.943; monotone 0.800; ratio 0.890 (pred 0.840 ± 0.20); curvature saturating — shape agreement: YES (level not claimed: 0.230 vs 0.405)

## Ablations (battery A, protocol-defined point statistics)

| arm | final D | rho | monotone | acc(final) | curve | accuracy | markers | JOINT |
|---|---|---|---|---|---|---|---|---|
| full | 0.230 | 0.943 | 0.80 | 0.62 | True | True | True | **PASS** |
| empty_r | 0.000 | 0.000 | 1.00 | 0.12 | False | False | False | **FAIL** |
| mismatched_r | 0.207 | 1.000 | 1.00 | 0.25 | True | True | False | **FAIL** |
| caring_only | 0.224 | 0.943 | 0.80 | 0.12 | True | False | True | **FAIL** |
| particularity_only | 0.041 | 0.600 | 0.60 | 0.62 | False | True | False | **FAIL** |
| generic_personalization | 0.011 | -0.886 | 0.40 | 0.12 | False | False | False | **FAIL** |

Per-component accuracy at final depth: full valence 0.75 / need 0.50; empty_r valence 0.12 / need 0.12.
Channel decomposition (full − mismatched, final depth): initiative_rate +0.000, response_depth +0.008, reference_rate +0.031, pref_alignment +0.039.

## Battery B — probes not used in R1 calibration

- TOST (full − base): +0.0826, 90% CI [0.0741, 0.0904] vs ±0.05 → **NOT SHOWN EQUIVALENT**
- Floor: Δ = 0.0000 vs 0.05 → **PASS**
- Ordering: full 0.221 > mismatched 0.201 > empty 0.000 (margin 0.020, interval [0.002, 0.026])

## Protocol checks (battery A)

- Floor: Δ = 0.0000 vs 0.05 → **PASS**
- Padding sensitivity: mean 0.0418, max 0.1901
- Marker naturalism: 0.0124 in (0.0, 0.08) → PASS

Total wall time: 6780s

## Amendment 2026-10-06 — right-vs-wrong contrast recomputed with the pipeline's estimator stream (PROTOCOL Amendment A3)

The original `scripts/analyze_specificity.py` fed the estimator `signal(t)` only, omitting the session-opening shared-event marker that the pipeline prepends (`pipeline.estimator_input`). Corrected values (LLM-free, deterministic; original files retained unchanged):

| statistic | original (signal-only stream) | corrected (pipeline stream) |
|---|---|---|
| paired mean diff (full − mismatched) | 0.3438 | **0.3750** |
| 90% interval | [0.1250, 0.5312] | [0.1250, 0.5938] |
| 95% interval | [0.0625, 0.5938] | [0.0938, 0.6250] |
| full mean / mismatched mean | 0.5938 / 0.2500 | 0.6250 / 0.2500 |
| run-level positive / zero / negative | 11 / 3 / 2 | 11 / 3 / 2 |

The corrected full mean (0.625) equals the pipeline's own final-depth full-arm accuracy in `ablations.csv` (0.62). Files: `right_vs_wrong_contrast_pipeline_stream.csv`, `right_vs_wrong_runs_pipeline_stream.csv`.

Also recorded for this grid (PROTOCOL A1, A2, A4): the divergence comparator is the unpadded stranger response (no token parity on D); the R1 effective conditioning weights are 0.6818/0.1364/0.1818; the initiative channel is the allocated rate, not generated messages.
