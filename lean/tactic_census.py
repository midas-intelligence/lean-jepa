"""
Tactic census from Mathlib SOURCE files (no tracing needed).

Heuristic: tactic blocks start after the `by` keyword; within an indented
block, the first identifier of each line / `;`-segment / `·` bullet is a
tactic name. Approximate — but applied identically to two snapshots, the
COMPARISON is meaningful, and we calibrate against the kernel-traced 2024
benchmark distribution.

Usage:
    python3 -m lean.tactic_census data/mathlib_head/Mathlib
"""

import re
import sys
from collections import Counter
from pathlib import Path

BY_RE = re.compile(r"\bby\b\s*$|\bby\b\s")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*")
SKIP = {"by", "fun", "let", "have", "show", "calc", "match", "if", "then",
        "else", "do", "with", "at", "this", "exact?", "fun"}
# `have`/`let`/`calc`/`show` ARE tactics too — but in source they also appear
# pervasively in term mode, which would inflate them; we exclude them from the
# census on BOTH snapshots so the comparison stays apples-to-apples.


def census(root: Path) -> Counter:
    counts = Counter()
    for f in root.rglob("*.lean"):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = text.split("\n")
        in_block = False
        block_indent = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if not stripped or stripped.startswith("--"):
                continue
            if in_block and indent <= block_indent and not stripped.startswith(("·", "|", "<;>")):
                in_block = False
            if BY_RE.search(line):
                in_block = True
                block_indent = indent
                after = line[BY_RE.search(line).end():].strip()
                segs = [after] if after else []
            elif in_block:
                segs = [stripped]
            else:
                continue
            for seg in segs:
                for part in re.split(r";|<;>|·", seg):
                    part = part.strip()
                    m = IDENT_RE.match(part)
                    if m and m.group(0) not in SKIP:
                        counts[m.group(0)] += 1
    return counts


def report(counts: Counter, total_needed=(0.90, 0.967, 0.99)):
    total = sum(counts.values())
    print(f"invocations parsed: {total:,}; distinct names: {len(counts):,}")
    freq = counts.most_common()
    cum = 0
    targets = list(total_needed)
    print("top-10:", ", ".join(f"{n} {c/total*100:.1f}%" for n, c in freq[:10]))
    cum64 = sum(c for _, c in freq[:63]) / total
    print(f"top-63 + <other> coverage: {cum64*100:.2f}%")
    for i, (n, c) in enumerate(freq, 1):
        cum += c
        while targets and cum / total >= targets[0]:
            print(f"names needed for {targets[0]*100:.1f}% coverage: {i}")
            targets.pop(0)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    report(census(root))
