# Changelog

All notable changes to the Lean-JEPA experiments. Newest first.

## v2 recipe — auxiliary tactic objective (2026-06-05)

**Motivation.** v1 (dynamics-only) passed T1 decisively (LPE/persistence = 0.21)
but probing showed tactic information in the embeddings is only incidental:
linear probe top-5 = 59.2%, MLP probe top-5 = 61.7% (target ≥ 75%). Value MAE
4.91 (target ≤ 3.0). Value-based ranking of actions ≈ chance — the gold tactic
is not the steps-to-QED minimizer, so that metric was replaced.

**Changes** (`lean/train.py`):
- Added auxiliary tactic-prediction head: `nn.Linear(384, 64)` on `z_before`,
  cross-entropy vs gold tactic name, weight 1.0, **gradients flow into the
  encoder** (unlike value head) to make embeddings tactic-discriminative.
- Value loss weight raised 1.0 → 2.0 (T3 push).
- `evaluate()` now reports policy metrics `ptop1`/`ptop5` from the aux head
  (T2 metric), alongside value-ranking metrics kept for comparison.
- Checkpoints now include `aux_head` weights.
- CLI: `--aux-weight` (default 1.0), `--value-weight` (default 2.0).

**Gates** (automated in monitor loop):
- Epoch 1: LPE/persistence > 0.7 → kill, retry with aux 0.5.
- Epoch 3: ptop5 ≤ 0.55 → escalate aux to 2.0.

**v1 artifacts archived**: `logs_v1/` (training log, results JSON, probe
results, stdout) and `checkpoints/v1_ablation_seed42.pt`. v1 = "dynamics-only"
ablation row for the paper.

## v1 recipe — initial Lean-JEPA (2026-06-04)

- Data: LeanDojo Benchmark 4 `random` split → 250,814 / 4,260 / 4,506 triples;
  tactic-name action space (top-63 + other = 96.7% coverage); 8K token vocab;
  256-token states.
- Architecture: ProofStateEncoder (8L × 384) + LeWM ARPredictor (6L AdaLN-zero)
  + SIGReg (λ=0.09) + ValueHead. 34.25M params.
- Objective: latent L2 + SIGReg + value MSE (value head detached).
- Probe utility (`lean/probe.py`): linear/MLP tactic probes on frozen embeddings.
- Result (seed 42, 12 epochs): T1 PASS (0.21), probe top-5 61.7%, value MAE 4.91.

## Project structure (2026-06-04)

- `lewm/` — JEPA, ARPredictor, SIGReg vendored unmodified from LeWorldModel
  (MIT; see lewm/ATTRIBUTION.md).
- `lean/` — domain code: data pipeline, encoder, build, train, probe.
- Dataset: Zenodo 12740403 (gitignored; re-download via curl).
- Telemetry disabled (`RAY_USAGE_STATS_ENABLED=0` in train.py); see SECURITY.md.
