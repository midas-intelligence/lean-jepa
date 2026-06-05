# Lean-JEPA: A Latent World Model over a Finite Tactic Vocabulary for Lean 4

Methodology and implementation guide. Self-contained: a research engineer can implement the system from this document alone.

---

## 1. Goal

Train a world model that predicts — in latent space — how a Lean 4 proof state changes when a tactic is applied, then use that model with a learned value function to select tactics during proof search.

The model never generates tactic text; it never reconstructs proof states. It selects from a fixed vocabulary of tactic actions (Section 2). The Lean kernel executes the proof step.

## 2. Empirical foundation: 64 actions cover 96.7% of all Mathlib proof steps

Every proof in Mathlib4 is a sequence of *tactic invocations* (e.g. `rw [add_comm]`, `simp only [foo]`, `intro h`). The *tactic name* is the leading identifier of the invocation: `rw`, `simp`, `intro`.

We measured the distribution of tactic names over the 250,814 tactic invocations of LeanDojo Benchmark 4 (Mathlib4 @ 29dcec07):

| Tactic name | Share of invocations | Cumulative |
|------------|-------|------|
| `rw` | 18.9% | 18.9% |
| `simp` | 12.9% | 31.7% |
| `exact` | 11.8% | 43.6% |
| `have` | 6.5% | 50.1% |
| `refine` | 5.5% | 55.6% |
| top 20 names | — | 83.9% |
| **top 63 names + `<other>`** | — | **96.7%** |

Only 259 distinct tactic names occur in all of Mathlib4. The 63 most frequent + 1 catch-all bucket cover **96.7% of every proof step in the library**.

**Coverage** means: 242,537 of 250,814 (96.7%) training transitions have an action label that is not `<other>`. The remaining 3.3% are labeled `<other>` and are kept in training (they still teach state dynamics).

This concentration justifies treating theorem proving as a **finite-action MDP** — the structural assumption world models need. Lean's action space appears unbounded (any tactic text); empirically, mathematicians draw from a small toolbox. Phase 1 of this project validated the identical pipeline at 15 actions (propositional rewrite rules); Mathlib needs only 64.

## 3. Has a finite tactic vocabulary been used before?

Yes — fixed tactic vocabularies are an established design:

- **HOList** (Bansal et al., ICML 2019): exactly 41 tactics, policy + value networks for HOL Light.
- **TacticToe** (Gauthier et al., 2018): predicts among recorded tactics in HOL4. No deep learning.
- **CoqGym/ASTactic** (Yang & Deng, ICML 2019): generates tactics from a grammar; tactic-name prediction over a fixed set is the standard baseline.

**What has NOT been done:** learning a **latent world model** of tactic dynamics. HOList learns state → action (policy). No latent transition model f(state, tactic) → next state for theorem proving exists in the literature (verified June 2026: I-JEPA, V-JEPA 2, LeWorldModel, DWMR all target vision/control/puzzles).

The contribution: tactic effects predicted in latent space + value-guided selection. The base architecture is validated: prop-logic solve rate 33.0 ± 0.3% vs hand-designed planner 31.3%, with collapse prevention via SIGReg (LeJEPA).

## 4. Why this matters for world models

1. **Per-action inference compute.** LLM provers decode tactic text token-by-token; ~1.5B-param ReProver beam search per state. World model: one encoder pass + 64 predictor passes (34M params).
2. **Reusable dynamics.** Policy provers map state→text without modeling tactic effects. World models support planning, lookahead, value bootstrapping (Dreamer; MuZero).
3. **Hardware.** Whole system (34M params) trains on one 24 GB MacBook.

## 5. Hypotheses and performance targets

| Tier | Hypothesis | Metric | Target | Status |
|------|------------|--------|--------|--------|
| T1 | Dynamics learnable | LPE / persistence ratio | < 0.7 | epoch 1 |
| T2 | Action ranking | top-5 (64 actions) | ≥ 75% | training |
| T3 | Value calibration | MAE ≤ 3.0 steps; ρ ≥ 0.5 | training |
| T4 | Proof search | pass@1 (LeanDojo random test) | ≥ 10% | Phase 2 |

T1 fails → architecture pivot (token-level objectives). T2–T3 = workshop paper. T4 = main-conference scale: ReProver achieves 51.2% with 229M LLM + retrieval; our model is 6.7× smaller and selects from a fixed vocabulary (10% reframes search efficiency).

## 6. Data (LeanDojo Benchmark 4, `random` split)

Triples: (state_before, tactic, state_after) + steps-to-QED.
Counts: 250,814 / 4,260 / 4,506 (train/val/test, theorem-disjoint).
Tokenizer: identifiers / digits / brackets / Unicode runs; vocab 8,000; ≤256 tokens.
Action: tactic-name id ∈ {0..63}. Value: steps-to-QED clipped at 50.

## 7. Architecture (34.25M params)

ProofStateEncoder (8L×384, 8 heads, GELU, LayerScale 1e-4, learned-query pooling — ours, 17.5M); ARPredictor (6L AdaLN-Zero, LeWM, 16.4M); Tactic embedding 64×384; ValueHead 2×384 MLP + softplus.

No EMA, no stop-grad: SIGReg λ=0.09. Value head on detached embeddings.

```
L = ||predict(enc(s), a) − enc(s′)||² + 0.09·SIGReg(enc(s)⊕enc(s′)) + MSE(value(enc(s′).detach()), V(s′))
```

AdamW lr 1e-3, wd 0.01, cosine; batch 48; 12 epochs; clip 1.0; seeds 42/137/256; select by val LPE. ~102 min/epoch.

## 8. Evaluation

- LPE vs persistence ‖enc(s) − enc(s′)‖² — T1.
- Top-1/Top-5 over 64 actions by value of predicted successor — T2.
- Value MAE + Spearman — T3.
- Rank ratio (effective rank / 384) — must stay > 0.5.
- Pass@1 with greedy value-guided search ≤ 10 min/theorem (T4). Bare tactics (`rfl`, `intro`, `simp`...) execute as-is; argument tactics use canonical defaults.

## 9. Implementation order

1. Data pipeline (done): 250,814 triples; coverage 96.7%
2. Architecture (done): 34.25M; forward+backward verified
3. Smoke test (done): 1.05 s/batch ≈ 102 min/epoch
4. Train seed 42 → evaluate T1
5. Offline eval: T2, T3
6. Seeds 137, 256 → mean ± std
7. Tactic execution map: bare vs canonical defaults
8. LeanDojo greedy search → T4
9. Ablations: SIGReg off; value head off; 32 vs 64 vocab
10. Demo: theorem → tactic ranking + UMAP trajectory

## 10. Risks

| Risk | Mitigation |
|------|------------|
| Top-1 low (multiple valid tactics) | top-5 primary |
| Persistence baseline strong | T1 ratio targets 30% improvement |
| 96.7% coverage ceiling | T4 reports covered subset rate |
| Rank collapse | rank ratio monitor; raise λ if < 0.5 |
| MPS NaN on padded rows | CLS always attends |
| Val theorems share Mathlib style | acceptable for v1 |

References: Ha & Schmidhuber 2018; LeCun 2022; Assran I-JEPA 2023; V-JEPA 2 2025; Maes LeWorldModel 2026; Balestriero & LeCun LeJEPA 2025; Yang LeanDojo 2023; Bansal HOList 2019; Yang & Deng CoqGym 2019; Gauthier TacticToe 2018; Hafner Dreamer 2020.
