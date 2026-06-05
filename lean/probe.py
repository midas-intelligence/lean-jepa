"""
Tactic-prediction probe on frozen Lean-JEPA embeddings (standard linear probing).

Trains 1-layer linear + 2-layer MLP probes: z_before -> tactic name (64-way),
on detached embeddings from the trained encoder. Reports top-1 / top-5
vs majority and frequency-sampling baselines.

Usage:
    python3 -u -m lean.probe --seed 42
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lean.build import build_model
from lean.train import TripleDataset, set_seed

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def embed_split(model, loader, device):
    Z, y = [], []
    model.eval()
    for batch in loader:
        z = model.encoder(batch["before"].to(device)).last_hidden_state[:, 0]
        Z.append(z.cpu())
        y.append(batch["action"])
    return torch.cat(Z), torch.cat(y)


def topk(logits, y, k):
    return (logits.topk(k, dim=1).indices == y[:, None]).any(1).float().mean().item()


def main(seed=42, epochs=10, lr=1e-3, bs=256):
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model, _, cfg = build_model("large")
    ckpt = torch.load(ROOT / f"checkpoints/lean_jepa_seed{seed}_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    tr = DataLoader(TripleDataset(ROOT / "data/processed/train.pkl"), batch_size=512)
    va = DataLoader(TripleDataset(ROOT / "data/processed/val.pkl"), batch_size=512)

    print("Embedding splits...")
    Ztr, ytr = embed_split(model, tr, device)
    Zva, yva = embed_split(model, va, device)
    print(f"train {tuple(Ztr.shape)}  val {tuple(Zva.shape)}")

    counts = Counter(ytr.tolist())
    top5 = [a for a, _ in counts.most_common(5)]
    base1 = (yva == counts.most_common(1)[0][0]).float().mean().item()
    base5 = torch.isin(yva, torch.tensor(top5)).float().mean().item()
    print(f"baseline majority: top1={base1:.3f} top5={base5:.3f}")

    results = {"baseline_top1": base1, "baseline_top5": base5, "seed": seed}

    for name, probe in [
        ("linear", nn.Linear(cfg["dim"], 64)),
        ("mlp", nn.Sequential(nn.Linear(cfg["dim"], 256), nn.GELU(), nn.Linear(256, 64))),
    ]:
        probe = probe.to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr)
        Xtr, Ytr = Ztr.to(device), ytr.to(device)
        for ep in range(epochs):
            perm = torch.randperm(len(Xtr), device=device)
            for i in range(0, len(Xtr), bs):
                idx = perm[i:i + bs]
                loss = F.cross_entropy(probe(Xtr[idx]), Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            logits = probe(Zva.to(device)).cpu()
        t1, t5 = topk(logits, yva, 1), topk(logits, yva, 5)
        print(f"{name}: top1={t1:.3f} top5={t5:.3f}")
        results[f"{name}_top1"], results[f"{name}_top5"] = t1, t5

    out = ROOT / f"logs/probe_seed{seed}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.seed)
