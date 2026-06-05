"""
LeanDojo Benchmark 4 data pipeline.

Loads (state_before, tactic, state_after) triples from LeanDojo Benchmark 4
and prepares them for the Lean-JEPA world model:

- Proof states are tokenized with a domain vocabulary (built from corpus).
- Tactics are normalized into action IDs via the K most frequent tactic heads
  (the first token: intro, apply, simp, rw, exact, ...).

Dataset format (Benchmark 4 JSON):
  Each theorem: {
    "url", "commit", "file_path", "full_name",
    "traced_tactics": [
      {"tactic": "intro h", "state_before": "...", "state_after": "..."},
      ...
    ]
  }
Final state is "no goals".
"""

import json
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Tokenization ────────────────────────────────────────────────────────────

MAX_STATE_TOKENS = 256
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.']*|[0-9]+|[(){}\[\],:;.]|[^\sA-Za-z0-9_(){}\[\],:;.]+"
)


def tokenize(text: str) -> list[str]:
    """Lean-aware whitespace+symbol tokenizer."""
    return _TOKEN_RE.findall(text)


def build_vocab(states, max_size: int = 8000) -> dict[str, int]:
    """Most frequent tokens across proof states."""
    counts = Counter()
    for s in states:
        counts.update(tokenize(s))
    vocab = {tok: i for i, tok in enumerate(SPECIAL)}
    for tok, _ in counts.most_common(max_size - len(SPECIAL)):
        vocab[tok] = len(vocab)
    return vocab


def encode_state(state: str, vocab: dict[str, int]) -> list[int]:
    ids = [BOS] + [vocab.get(t, UNK) for t in tokenize(state)][: MAX_STATE_TOKENS - 2] + [EOS]
    return ids + [PAD] * (MAX_STATE_TOKENS - len(ids))


# ── Tactic action space ─────────────────────────────────────────────────────

def tactic_head(tactic: str) -> str:
    """First token of the tactic — e.g. 'rw [foo] at h' -> 'rw'."""
    t = tactic.strip()
    m = re.match(r"[A-Za-z_][A-Za-z0-9_']*", t)
    return m.group(0) if m else t[:10]


def build_tactic_vocab(tactics, top_k: int = 64) -> dict[str, int]:
    counts = Counter(tactic_head(t) for t in tactics)
    vocab = {"<other>": 0}
    for head, _ in counts.most_common(top_k - 1):
        vocab[head] = len(vocab)
    return vocab


def coverage(tactics, tactic_vocab) -> float:
    n = sum(1 for t in tactics if tactic_head(t) in tactic_vocab)
    return n / max(len(tactics), 1)


# ── Dataset extraction ──────────────────────────────────────────────────────

@dataclass
class Triple:
    state_before: str
    tactic: str
    state_after: str
    theorem: str
    steps_to_qed: int  # remaining tactics until proof completes


def load_benchmark(data_dir: str, split: str) -> list[Triple]:
    """Read benchmark JSON -> list of (state_before, tactic, state_after) triples."""
    path = Path(data_dir) / f"{split}.json"
    with open(path) as f:
        theorems = json.load(f)

    triples = []
    for thm in theorems:
        tt = thm.get("traced_tactics") or []
        n = len(tt)
        for i, t in enumerate(tt):
            if not t.get("state_before") or not t.get("state_after"):
                continue
            triples.append(
                Triple(
                    state_before=t["state_before"],
                    tactic=t["tactic"],
                    state_after=t["state_after"],
                    theorem=thm["full_name"],
                    steps_to_qed=n - i,
                )
            )
    return triples


def prepare(data_dir: str, out_dir: str, vocab_size: int = 8000, top_k_tactics: int = 64):
    """One-time preprocessing: build vocabs, encode all splits, cache."""
    os.makedirs(out_dir, exist_ok=True)

    train = load_benchmark(data_dir, "train")
    val = load_benchmark(data_dir, "val")
    test = load_benchmark(data_dir, "test")
    print(f"triples: train={len(train)} val={len(val)} test={len(test)}")

    vocab = build_vocab((t.state_before for t in train), vocab_size)
    tactic_vocab = build_tactic_vocab((t.tactic for t in train), top_k_tactics)
    cov = coverage([t.tactic for t in train], tactic_vocab)
    print(f"state vocab={len(vocab)} tactic vocab={len(tactic_vocab)} coverage={cov:.3f}")

    with open(Path(out_dir) / "vocab.pkl", "wb") as f:
        pickle.dump({"state": vocab, "tactic": tactic_vocab}, f)

    for name, triples in [("train", train), ("val", val), ("test", test)]:
        rows = [
            {
                "before": encode_state(t.state_before, vocab),
                "after": encode_state(t.state_after, vocab),
                "action": tactic_vocab.get(tactic_head(t.tactic), 0),
                "steps_to_qed": min(t.steps_to_qed, 50),
            }
            for t in triples
        ]
        with open(Path(out_dir) / f"{name}.pkl", "wb") as f:
            pickle.dump(rows, f)
        print(f"cached {name}: {len(rows)} rows")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/leandojo_benchmark_4/random"
    prepare(data_dir, "data/processed")
