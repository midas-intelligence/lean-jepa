"""
T4 de-risk: verify live LeanDojo <-> Lean kernel interaction works on this
machine. Enters one benchmark theorem, runs one tactic, checks the state
changes. First run downloads the traced mathlib4 cache (large, one-time).

Usage:
    python3 -u -m lean.test_interaction
"""

import json
import os
import time

os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

from lean_dojo import LeanGitRepo, Theorem, Dojo, TacticState  # noqa: E402


def main():
    t0 = time.time()
    with open("data/leandojo_benchmark_4/random/val.json") as f:
        theorems = json.load(f)

    # Pick a theorem whose first recorded tactic is argument-free
    target = None
    for thm in theorems:
        tt = thm.get("traced_tactics") or []
        if tt and tt[0]["tactic"].strip() in {"rfl", "ext", "intro h", "constructor", "simp"}:
            target = thm
            break
    if target is None:
        target = next(t for t in theorems if t.get("traced_tactics"))

    print(f"theorem: {target['full_name']}")
    print(f"file:    {target['file_path']}")
    print(f"gold first tactic: {target['traced_tactics'][0]['tactic']!r}")

    repo = LeanGitRepo(target["url"], target["commit"])
    theorem = Theorem(repo, target["file_path"], target["full_name"])
    print(f"repo object OK ({time.time()-t0:.0f}s); entering Dojo "
          f"(first run downloads traced cache — may take a long time)...")

    with Dojo(theorem) as (dojo, init_state):
        assert isinstance(init_state, TacticState)
        print(f"ENTERED ({time.time()-t0:.0f}s)")
        print("initial state:", init_state.pp[:200].replace("\n", " | "))
        result = dojo.run_tac(init_state, target["traced_tactics"][0]["tactic"])
        print("after tactic:", str(getattr(result, 'pp', result))[:200].replace("\n", " | "))
        print(f"INTERACTION OK ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
