# Lean-JEPA: Latent World Model for Lean 4 Tactic Dynamics

A JEPA-style world model that learns — purely from observed proof traces —
how Lean 4 proof states change when tactics are applied, plus a value
function (steps-to-QED) for tactic selection. The model never generates
tactic text and never reconstructs proof states.

Phase 2 of a two-phase project; Phase 1 ([sym-jepa](https://github.com/renz011tzar/sym-jepa))
validated the architecture on propositional-logic rewriting.

**Full methodology**: [METHODOLOGY.md](METHODOLOGY.md) ·
**Change history**: [CHANGELOG.md](CHANGELOG.md) ·
**Dependency provenance**: [SECURITY.md](SECURITY.md)

## Key facts

- Data: LeanDojo Benchmark 4 (Mathlib4) — 250,814 tactic transitions
- Action space: 64 tactic names = 96.7% of all Mathlib proof steps
- Model: 34.3M params; trains on one 24 GB MacBook (MPS)
- Components from [LeWorldModel](https://github.com/lucas-maes/le-wm) (MIT):
  ARPredictor, SIGReg — see `lewm/ATTRIBUTION.md`

## Current results

| Tier | Metric | Target | v1 (dynamics-only ablation) |
|------|--------|--------|------------------------------|
| T1 | LPE / persistence | < 0.7 | **0.21 — PASS** |
| T2 | tactic top-5 | ≥ 75% | 61.7% (probe) |
| T3 | value MAE (steps) | ≤ 3.0 | 4.91 |
| T4 | pass@1 proof search | ≥ 10% | Phase 2 pending |

v2 recipe (auxiliary tactic objective) in training — see CHANGELOG.

## Usage

```bash
pip install -r requirements.txt

# Data: download benchmark (~65 MB) and preprocess
mkdir -p data && cd data
curl -L -O https://zenodo.org/record/12740403/files/leandojo_benchmark_4.tar.gz
tar -xzf leandojo_benchmark_4.tar.gz && cd ..
python3 -m lean.data data/leandojo_benchmark_4/random

# Train (v2 recipe)
python3 -u -m lean.train --seed 42 --max-epochs 12

# Probe frozen embeddings
python3 -u -m lean.probe --seed 42
```

## License

MIT
