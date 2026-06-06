"""
Post-hoc head refit on a frozen Lean-JEPA encoder (standard probing practice).

Motivation (measured 2026-06-05): checkpoint selection keys on LPE, freezing
value/aux heads at an early training epoch. Offline evidence: frozen
embeddings support value MAE 3.37 (log1p target, 3.4k samples) vs 4.94 from
the checkpoint's own head. This refits both heads properly:

  - ValueHead on log1p(steps-to-QED), inverse-transformed at eval
  - Policy head (MLP 384->256->64) for tactic prediction

Runs on CPU (no contention with GPU training). Uses a train subsample for
embedding speed; full val for eval.

Usage:
    python3 -u -m lean.refit --ckpt checkpoints/v2_seed42_disrupted.pt --train-n 60000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lean.build import build_model
from lean.train import TripleDataset

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def embed(model, ds, n=None, batch=64, which="after"):
    idx = range(len(ds)) if n is None else np.random.RandomState(0).choice(len(ds), n, replace=False)
    loader = DataLoader(Subset(ds, list(idx)), batch_size=batch)
    Z, S, A = [], [], []
    for b in loader:
        Z.append(model.encoder(b[which]).last_hidden_state[:, 0])
        S.append(b["steps"])
        A.append(b["action"])
    return torch.cat(Z), torch.cat(S), torch.cat(A)


def fit_value(Ztr, Str, Zva, Sva, epochs=40, log_target=True):
    head = nn.Sequential(nn.Linear(384, 384), nn.LayerNorm(384), nn.GELU(),
                         nn.Linear(384, 384), nn.LayerNorm(384), nn.GELU(),
                         nn.Linear(384, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    tgt = torch.log1p(Str) if log_target else Str
    n = len(Ztr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            j = perm[i:i + 512]
            loss = F.mse_loss(head(Ztr[j]).squeeze(-1), tgt[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = head(Zva).squeeze(-1)
        p = torch.expm1(p).clamp(0, 50) if log_target else p
    return head, (p - Sva).abs().mean().item()


def fit_policy(Ztr, Atr, Zva, Ava, epochs=40):
    head = nn.Sequential(nn.Linear(384, 256), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(256, 64))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    n = len(Ztr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            j = perm[i:i + 512]
            loss = F.cross_entropy(head(Ztr[j]), Atr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = head(Zva)
        top = logits.argsort(1, descending=True)
        t1 = (top[:, 0] == Ava).float().mean().item()
        t5 = (top[:, :5] == Ava[:, None]).any(1).float().mean().item()
    return head, t1, t5


def main(ckpt, train_n=60000):
    t0 = time.time()
    model, _, cfg = build_model("large")
    ck = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(ck["model"]); model.eval()

    tr_ds = TripleDataset(ROOT / "data/processed/train.pkl")
    va_ds = TripleDataset(ROOT / "data/processed/val.pkl")

    print(f"embedding train subsample n={train_n} (after-states for value)...")
    Ztr_a, Str, _ = embed(model, tr_ds, train_n, which="after")
    print(f"embedding train subsample (before-states for policy)... ({time.time()-t0:.0f}s)")
    Ztr_b, _, Atr = embed(model, tr_ds, train_n, which="before")
    print(f"embedding val (both)... ({time.time()-t0:.0f}s)")
    Zva_a, Sva, _ = embed(model, va_ds, which="after")
    Zva_b, _, Ava = embed(model, va_ds, which="before")

    _, mae_log = fit_value(Ztr_a, Str, Zva_a, Sva, log_target=True)
    _, mae_raw = fit_value(Ztr_a, Str, Zva_a, Sva, log_target=False)
    _, t1, t5 = fit_policy(Ztr_b, Atr, Zva_b, Ava)

    res = {"ckpt": str(ckpt), "train_n": train_n,
           "value_mae_log1p": mae_log, "value_mae_raw": mae_raw,
           "policy_top1": t1, "policy_top5": t5,
           "wall_s": time.time() - t0}
    print(json.dumps(res, indent=2))
    out = ROOT / f"logs/refit_{Path(ckpt).stem}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train-n", type=int, default=60000)
    args = p.parse_args()
    main(args.ckpt, args.train_n)
