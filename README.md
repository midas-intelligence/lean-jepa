# Lean-JEPA: A Latent World Model for Lean 4 Tactic Dynamics

A JEPA-style world model that learns — purely from observed proof traces —
how Lean 4 proof states change when tactics are applied, together with a
value function (steps-to-QED) for tactic selection during proof search. The
model never generates tactic text and never reconstructs proof states: it
predicts tactic effects in latent space and plans over a finite tactic
vocabulary.

Phase 2 of a two-phase project; Phase 1
([sym-jepa](https://github.com/renz011tzar/sym-jepa)) validated the
architecture on propositional-logic rewriting.

**How it proves theorems (start here)**: [PROVING.md](PROVING.md) ·
**Methodology**: [METHODOLOGY.md](METHODOLOGY.md) ·
**Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Key facts

- Data: LeanDojo Benchmark 4 (Mathlib4) — 250,814 tactic transitions
- Action space: 64 tactic names = 96.7% of all Mathlib proof steps
- Model: 34.3M parameters; trains on a single consumer-grade 24 GB device

## Architecture

A proof-state encoder (8-layer, 384-d transformer with learned-query pooling,
introduced here) maps Lean goal states to embeddings. Dynamics are modeled
with the AdaLN-Zero autoregressive predictor and SIGReg collapse regularizer
from LeWorldModel (Maes et al., 2026); tactic actions enter through a learned
embedding. A value head predicts remaining proof length (steps-to-QED). The
training objective combines latent next-state prediction, SIGReg, an
auxiliary tactic-classification loss, and value regression — see
[METHODOLOGY.md](METHODOLOGY.md) for the full specification and design
rationale.

## Current results

| Tier | Metric | Target | Best measured |
|------|--------|--------|---------------|
| T1 | next-state prediction vs persistence baseline | ratio < 0.7 | **0.25 — pass** (3 runs) |
| T2 | gold tactic in top-5 of 64 | ≥ 75% | 69.2% |
| T3 | steps-to-QED MAE | ≤ 3.0 | 3.52 and falling (v3 in training) |
| T4 | pass@1, value-guided search (LeanDojo test) | ≥ 10% | pending T2/T3 |

Recipe iterations (v1 → v3), failures included, are documented in
[CHANGELOG.md](CHANGELOG.md).

## Usage

```bash
pip install -r requirements.txt

# Data: download benchmark (~65 MB) and preprocess
mkdir -p data && cd data
curl -L -O https://zenodo.org/record/12740403/files/leandojo_benchmark_4.tar.gz
tar -xzf leandojo_benchmark_4.tar.gz && cd ..
python3 -m lean.data data/leandojo_benchmark_4/random

# Train (current recipe)
python3 -u -m lean.train --seed 42 --max-epochs 12 \
  --aux-weight 2.0 --value-log --value-into-encoder

# Probe frozen embeddings / refit heads
python3 -u -m lean.probe --seed 42
python3 -u -m lean.refit --ckpt checkpoints/lean_jepa_seed42_best.pt
```

## Acknowledgments

This project builds on
[LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes, Le Lidec, Scieur,
LeCun & Balestriero, 2026) — the `lewm/` directory vendors its JEPA,
ARPredictor, and SIGReg modules unmodified under MIT (see
`lewm/ATTRIBUTION.md`) — and on
[LeanDojo](https://leandojo.org) (Yang et al., NeurIPS 2023) for the
benchmark dataset and the Lean interaction toolkit.

## References

```bibtex
@inproceedings{yang2023leandojo,
  title={LeanDojo: Theorem Proving with Retrieval-Augmented Language Models},
  author={Yang, Kaiyu and Swope, Aidan and Gu, Alex and Chalamala, Rahul and
          Song, Peiyang and Yu, Shixing and Godil, Saad and Prenger, Ryan and
          Anandkumar, Anima},
  booktitle={NeurIPS},
  year={2023}
}

@article{maes2026leworldmodel,
  title={LeWorldModel},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and
          LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}

@article{lecun2022path,
  title={A Path Towards Autonomous Machine Intelligence},
  author={LeCun, Yann},
  journal={OpenReview},
  year={2022}
}
```

## License

MIT
