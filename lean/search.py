"""
T4: value-guided proof search against the live Lean kernel (LeanDojo).

For each test theorem:
  1. encode the current proof state
  2. rank the 64 tactic moves (policy head), keep the executable ones
  3. imagine successors with the dynamics predictor, score with the value head
  4. execute the best candidate in Lean; on rejection try the next
  5. best-first over the discovered states until `no goals` or budget out

Baselines run under the identical move space and budget:
  - fixed-recipe: a tidy-style constant sequence of bare tactics
  - random: uniform over executable moves

Usage:
    python3 -u -m lean.search --ckpt checkpoints/lean_jepa_seed42_best.pt \
        --num-theorems 100 --method model        # or: fixed, random
"""

import argparse
import heapq
import json
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

import torch

from lean.build import build_model
from lean.data import encode_state
from lean.train import ValueHead

ROOT = Path(__file__).resolve().parent.parent

# ── Execution map ───────────────────────────────────────────────────────────
# Moves executable without arguments. Lean rejects them where inapplicable —
# rejection is cheap feedback, not failure.
BARE = {
    "rfl", "simp", "simpa", "dsimp", "simp_all", "norm_num", "norm_cast",
    "field_simp", "ring", "linarith", "omega", "positivity", "decide",
    "aesop", "intro", "intros", "rintro", "constructor", "ext", "ext1",
    "funext", "by_contra", "classical", "infer_instance", "split_ifs",
    "gcongr", "subst", "exacts", "contrapose", "swap",
}
# Lean's built-in argument search variants (expensive; capped per proof).
LIBRARY_SEARCH = {"exact": "exact?", "apply": "apply?", "rw": "rw?"}

FIXED_RECIPE = ["intros", "simp_all", "norm_num", "aesop", "linarith", "omega",
                "ring", "positivity", "decide", "rfl"]


@dataclass(order=True)
class Node:
    score: float
    state: object = field(compare=False)
    depth: int = field(compare=False, default=0)


class Ranker:
    """Loads the trained model; ranks the 64 moves for a live proof state."""

    def __init__(self, ckpt_path, device="cpu"):
        self.device = torch.device(device)
        self.model, _, cfg = build_model("large")
        ck = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ck["model"])
        self.value_head = ValueHead(cfg["dim"], cfg["dim"])
        self.value_head.load_state_dict(ck["value_head"])
        self.aux_head = torch.nn.Linear(cfg["dim"], cfg["num_tactics"])
        self.aux_head.load_state_dict(ck["aux_head"])
        for m in (self.model, self.value_head, self.aux_head):
            m.to(self.device).eval()

        with open(ROOT / "data/processed/vocab.pkl", "rb") as f:
            v = pickle.load(f)
        self.state_vocab = v["state"]
        self.id2tactic = {i: t for t, i in v["tactic"].items()}
        self.num_actions = len(self.id2tactic)
        self.value_log = True  # v3 recipe: value head predicts log1p(steps)

    @torch.no_grad()
    def embed(self, state_text):
        ids = torch.tensor([encode_state(state_text, self.state_vocab)],
                           device=self.device)
        return self.model.encoder(ids).last_hidden_state[:, 0]

    @torch.no_grad()
    def state_value(self, state_text):
        v = self.value_head(self.embed(state_text)).squeeze().item()
        return float(torch.expm1(torch.tensor(v))) if self.value_log else v

    @torch.no_grad()
    def rank_moves(self, state_text):
        """Policy logits + imagined-successor values -> ordered move list."""
        z = self.embed(state_text)
        policy = self.aux_head(z).softmax(-1).squeeze(0)          # (64,)
        acts = torch.arange(self.num_actions, device=self.device).view(-1, 1)
        zrep = z.expand(self.num_actions, -1).unsqueeze(1)
        emb = self.model.action_encoder(acts)
        succ = self.model.predict(zrep, emb)[:, -1, :]
        val = self.value_head(succ).squeeze(-1)
        if self.value_log:
            val = torch.expm1(val)
        # Lower predicted steps-to-QED is better; break ties by policy prob.
        order = sorted(range(self.num_actions),
                       key=lambda a: (val[a].item(), -policy[a].item()))
        return [(self.id2tactic[a], policy[a].item(), val[a].item())
                for a in order]


def executable(move, ls_budget):
    if move in BARE:
        return move
    if move in LIBRARY_SEARCH and ls_budget > 0:
        return LIBRARY_SEARCH[move]
    return None


def prove(theorem, ranker=None, method="model", max_nodes=24,
          ls_per_proof=4, timeout_s=480, rng=None):
    """Attempt one theorem. Returns dict with proved flag + stats."""
    from lean_dojo import Dojo, TacticState, ProofFinished

    t0 = time.time()
    stats = {"theorem": theorem.full_name, "method": method, "proved": False,
             "nodes": 0, "lean_calls": 0, "wall_s": 0.0, "proof": None}
    try:
        with Dojo(theorem, timeout=timeout_s) as (dojo, init):
            frontier = [Node(0.0, init)]
            seen = set()
            ls_budget = ls_per_proof
            trace = {id(init): []}
            while frontier and stats["nodes"] < max_nodes:
                if time.time() - t0 > timeout_s:
                    break
                node = heapq.heappop(frontier)
                state = node.state
                if not isinstance(state, TacticState):
                    continue
                if state.pp in seen:
                    continue
                seen.add(state.pp)
                stats["nodes"] += 1

                if method == "model":
                    moves = [m for m, _, _ in ranker.rank_moves(state.pp)]
                elif method == "fixed":
                    moves = list(FIXED_RECIPE)
                else:  # random
                    moves = list(BARE)
                    rng.shuffle(moves)

                tried = 0
                for mv in moves:
                    cmd = executable(mv, ls_budget)
                    if cmd is None:
                        continue
                    if cmd in LIBRARY_SEARCH.values():
                        ls_budget -= 1
                    tried += 1
                    if tried > 8:  # per-node expansion cap
                        break
                    stats["lean_calls"] += 1
                    res = dojo.run_tac(state, cmd)
                    if isinstance(res, ProofFinished):
                        stats["proved"] = True
                        stats["proof"] = trace[id(state)] + [cmd]
                        stats["wall_s"] = time.time() - t0
                        return stats
                    if isinstance(res, TacticState) and res.pp not in seen:
                        score = (ranker.state_value(res.pp)
                                 if method == "model" else node.depth + 1)
                        trace[id(res)] = trace[id(state)] + [cmd]
                        heapq.heappush(frontier, Node(score, res, node.depth + 1))
    except Exception as e:  # Dojo init failures, timeouts, crashed workers
        stats["error"] = f"{type(e).__name__}: {e}"[:200]
    stats["wall_s"] = time.time() - t0
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/lean_jepa_seed42_best.pt")
    p.add_argument("--method", choices=["model", "fixed", "random"], default="model")
    p.add_argument("--num-theorems", type=int, default=100)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-nodes", type=int, default=24)
    p.add_argument("--timeout", type=int, default=480)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from lean_dojo import LeanGitRepo, Theorem

    with open(ROOT / "data/leandojo_benchmark_4/random/test.json") as f:
        meta = json.load(f)
    rng = random.Random(0)
    rng.shuffle(meta)  # fixed shuffle -> same theorem sample for all methods
    sample = meta[args.offset:args.offset + args.num_theorems]

    ranker = Ranker(args.ckpt) if args.method == "model" else None
    rng2 = random.Random(1)

    results, proved = [], 0
    for i, thm in enumerate(sample):
        repo = LeanGitRepo(thm["url"], thm["commit"])
        theorem = Theorem(repo, thm["file_path"], thm["full_name"])
        r = prove(theorem, ranker, args.method, args.max_nodes,
                  timeout_s=args.timeout, rng=rng2)
        results.append(r)
        proved += r["proved"]
        print(f"[{i+1}/{len(sample)}] {r['proved'] and 'PROVED' or '      '} "
              f"{thm['full_name'][:60]}  nodes={r['nodes']} "
              f"calls={r['lean_calls']} t={r['wall_s']:.0f}s "
              f"running={proved}/{i+1}")

    out = args.out or f"logs/search_{args.method}_{args.offset}_{args.num_theorems}.json"
    Path(out).write_text(json.dumps({
        "method": args.method, "ckpt": args.ckpt, "n": len(sample),
        "proved": proved, "pass_rate": proved / max(len(sample), 1),
        "results": results}, indent=2))
    print(f"\n{args.method}: {proved}/{len(sample)} = {proved/max(len(sample),1):.1%}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
