# lovellm reference implementation

Companion artifact for *Love for Believable AI: Artificial Partiality and
Relationship Persistence as an Engineerable Stance*.

Release v1.1.0 of this artifact is archived at
[doi:10.5281/zenodo.21462977](https://doi.org/10.5281/zenodo.21462977);
see `CITATION.cff` for how to cite it.

The repository contains two evidence classes:

- `scripts/reproduce.py` uses a deterministic synthetic stand-in to test the
  pipeline, invariants, metrics, and analytical curve.
- `scripts/reproduce_real*.py` run the measured grids on
  `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, pinned to Hugging Face revision
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.

All relationships are constructed personas. The artifact contains no human
data and supports no claim about attachment, perceived mind, or system
interiority.

## Contents

```text
lovellm/                    mechanism, persona generator, arms, and metrics
scripts/reproduce.py        synthetic pipeline validation
scripts/reproduce_real.py   initial open-weight-model grid
scripts/reproduce_real_v2.py R1 guarded-mechanism grid
scripts/reproduce_real_v3.py R1-L fresh-persona grid
scripts/reproduce_real_v4.py R1-XL grid and battery B
scripts/analyze_specificity.py offline right-vs-wrong analysis
scripts/audit_intervals_v4.py offline near-threshold interval recomputation
scripts/sensitivity_weights_v4.py offline channel-weighting sensitivity analysis
tests/                      invariant, negative-control, and metric tests
PROTOCOL.md                 versioned analysis record
results/                    stored tables, figures, hashes, and run metadata
```

The stored result directories contain aggregate tables and selected per-run
decision inputs. Raw generated text from the historical real-model runs was
not retained. Re-executing a real-model runner regenerates the model outputs
and derived tables; it is computationally expensive and requires Apple Silicon.

## Environment

The recorded real-model environment was Python 3.12.11, MLX 0.32.0,
MLX-LM 0.31.3, NumPy 2.5.1, and Matplotlib 3.11.0. The real-model integration
uses MLX and therefore targets macOS on Apple Silicon.

```sh
# run from this directory (prototype/ inside the paper repository)
uv venv --python 3.12
uv pip install -e '.[dev,real]'
.venv/bin/pytest
```

For synthetic validation only, install `.[dev]`; MLX is not required.

## Reproduction commands

```sh
.venv/bin/python scripts/reproduce.py
.venv/bin/python scripts/reproduce_real.py
.venv/bin/python scripts/reproduce_real_v2.py
.venv/bin/python scripts/reproduce_real_v3.py
.venv/bin/python scripts/reproduce_real_v4.py
.venv/bin/python scripts/analyze_specificity.py
.venv/bin/python scripts/audit_intervals_v4.py
.venv/bin/python scripts/sensitivity_weights_v4.py
```

Each command writes to its corresponding directory under `results/`. Prediction
files are checked against recorded SHA-256 values before generation. The hashes
establish file integrity; they are not independent proof that a protocol
preceded its corresponding run. `PROTOCOL.md` is therefore described as a
versioned analysis record, not a preregistration.

## Enforced properties

- The other-model module does not import the persona generator.
- The allocation is zero-sum only over the surplus; the baseline floor is
  outside the contested budget.
- Divergence probes use exact token parity and paired generation seeds.
- The floor comparison uses the unpadded empty-history prompt against the
  unmodified base prompt.
- Spearman correlation uses average ranks for ties; constant curves return
  zero correlation.
- Quality-equivalence intervals resample the two arms independently at the
  run level. Other paired contrasts resample shared run indices.
