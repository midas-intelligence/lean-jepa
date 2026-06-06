# How Lean-JEPA Proves Theorems

A plain-language explanation of the proving process, the training feedback, the
current scope, and the road to a fully competitive prover. No Lean experience
assumed.

---

## 1. The game being played

**Lean** is a proof checker: you state a theorem, then transform the goal step
by step until nothing is left to prove. Each transformation is called a
**tactic** — a command like `intro h` ("assume the hypothesis"), `simp`
("simplify"), or `linarith` ("settle it with linear arithmetic").

A **proof state** is the current scoreboard: what you've assumed so far and
what remains to prove (the goal, written after `⊢`). Applying a tactic changes
the state. When Lean reports **"no goals"**, the theorem is proved — and this
verdict is checked by Lean's kernel, so a wrong proof is *impossible* to pass.

Proving is therefore a navigation problem: find a sequence of tactics from the
starting state to "no goals". That is exactly the kind of problem world models
are built for.

## 2. The three learned components

| Component | Question it answers | Chess analogy |
|-----------|--------------------|----------------|
| **Dynamics predictor** (T1) | "If I play tactic X here, what will the state look like?" — predicted in latent space, without running Lean | knowing how pieces move |
| **Policy signal** (T2) | "Which of the 64 tactic moves look right here?" → a ranked shortlist | move instinct |
| **Value head** (T3) | "How many steps from QED is this state?" | position evaluation |

The 64 moves are the 63 most frequent Mathlib tactic names + a catch-all
bucket — together covering 96.7% of every proof step mathematicians actually
wrote in Mathlib (the largest math library in existence).

## 3. The proving loop (T4)

```
state ──► encoder ──► shortlist (top-k of 64 moves)
                          │ keep only executable moves
                          ▼
              imagine each successor (dynamics predictor)
                          │ score with value head
                          ▼
              execute best move in REAL Lean (LeanDojo)
                          │
            Lean accepts → new state  /  Lean rejects → try next move
                          │
              repeat best-first until "no goals" or budget out
```

Two properties worth noting:

- **Lean is the referee at every step.** The model proposes; the kernel
  disposes. If the model suggests a move that doesn't apply, Lean rejects it
  and the search tries the next candidate. A completed proof is
  kernel-certified — there are no false positives in this field, ever.
- **The model never writes proof text.** It selects among 64 moves. This makes
  each decision ~1000× cheaper than LLM text generation, at the cost of not
  choosing tactic *arguments* (see §6).

## 4. Worked example

Theorem (a real Mathlib-style lemma, simplified):

```lean
theorem ex (a b : ℝ) (h : 0 < a) (h2 : 0 < b) : 0 < a + b
```

**Step 1.** State: hypotheses `h : 0 < a`, `h2 : 0 < b`, goal `⊢ 0 < a + b`.
The encoder reads this text and produces an embedding. The shortlist comes
back (illustrative ranking):

| Rank | Move | Value of imagined successor |
|------|------|------------------------------|
| 1 | `linarith` | 0.4 steps-to-go |
| 2 | `positivity` | 0.5 |
| 3 | `simp` | 4.1 |
| 4 | `norm_num` | 4.3 |
| 5 | `apply` | (not executable bare — skipped) |

**Step 2.** Search executes `linarith` (rank 1, executable). Lean's linear
arithmetic procedure finds the combination `0 < a + b` follows from the two
hypotheses.

**Step 3.** Lean reports **no goals**. Theorem proved, certificate checked by
the kernel. Total cost: one encoder pass, 64 imagined successors, one real
Lean call.

A harder theorem works the same way but with backtracking: if `linarith`
fails, Lean rejects it, search tries `positivity`; if a move succeeds but
leads to a state the value head scores poorly, the search extends a more
promising branch first (best-first search).

## 5. Training: where the feedback comes from and why it's correct

**The model never trains against its own guesses, and Lean is not run during
training.** It learns from 250,814 recorded transitions in LeanDojo
Benchmark 4 — and every single one was produced by the Lean kernel when
Mathlib itself was compiled. The dataset is a log of *verified* mathematics:

- `state_before` — a real proof state from a real Mathlib proof
- `tactic` — what the human mathematician actually played there
- `state_after` — what Lean *actually* produced (kernel-checked)

Three loss signals per training example, all against this ground truth:

1. **Dynamics loss**: predict the embedding of `state_after` from
   (`state_before`, tactic name). Truth = the recorded next state.
2. **Tactic loss**: classify which of the 64 moves was played. Truth = the
   recorded human tactic.
3. **Value loss**: predict steps remaining until that proof finished. Truth =
   counted directly from the recorded proof.

Why this cannot silently learn the wrong thing:

- **Labels are kernel-verified facts**, not model outputs and not heuristics —
  there is no reward model to hack and no self-generated data to drift on.
- **Theorem-disjoint splits**: validation/test states come from theorems the
  model never saw in training; memorizing doesn't transfer.
- **Baselines guard the metrics**: dynamics must beat the "state doesn't
  change" predictor; tactic ranking must beat the majority-vote guess; value
  must beat predicting the median. (All three checks ran on every experiment;
  v1's failure of two of them is what triggered the v2/v3 recipe revisions —
  documented in CHANGELOG.md.)
- **At proof time** the feedback is even stricter: the kernel accepts or
  rejects every single move.

## 6. Current scope — stated honestly

The model picks the tactic **verb**, not its arguments (`rw` but not *which
equation*). At search time:

- ~30 of the 64 moves execute as-is (`rfl`, `intro`, `simp`, `linarith`,
  `omega`, `ring`, `aesop`, ...) — many are proof-finishing decision
  procedures;
- `exact`/`apply`/`rw` can be mapped to Lean's own library-search variants
  (`exact?`, `apply?`, `rw?`);
- content-inventing moves (`have`, `use`, `calc`) are skipped in v1.

Reference points on the same test set: a fixed bare-tactic recipe ("tidy",
LeanDojo paper) proves 23.8%; ReProver (299M LLM + retrieval, full tactic
text) proves 51.2%. Our pre-registered target is ≥10% — the demonstration
that *learned planning over a world model* completes proofs end-to-end; the
interesting comparison is whether value-guided move selection beats the fixed
recipe over the same move space.

## 7. Road to a fully competitive prover

| Stage | Adds | Expected effect |
|-------|------|-----------------|
| **1. (this work)** verb-level world model + value-guided search | the paradigm: plan in latent space, kernel verifies | ~10–25% of theorems |
| **2. Premise retrieval** | a retriever picks the argument for `rw`/`exact`/`apply` from Mathlib's ~167K lemmas (ReProver-style) | unlocks the argument-bearing half of the move space |
| **3. Argument generation** | a small decoder writes full tactic strings, *ranked and pruned by the world model* | full action space, still cheap: the model only decodes for moves the planner already likes |
| **4. Expert iteration** | proofs found by search become new training data; retrain; repeat (AlphaZero loop) | self-improvement beyond human traces |
| **5. Scale** | larger encoder/context (the 256-token truncation is a known ceiling), more search budget | competitive pass rates |

Each stage keeps the defining property of this approach: **the expensive
generative work happens only where the world model says it's worth it**, and
the Lean kernel certifies every step.

---

*Related docs: [METHODOLOGY.md](METHODOLOGY.md) (formal spec, metrics, tiers),
[CHANGELOG.md](CHANGELOG.md) (recipe history v1→v3), [SECURITY.md](SECURITY.md)
(dependency provenance).*
