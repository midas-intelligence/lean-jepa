"""
Train Lean-JEPA: latent world model over Lean 4 tactic dynamics.

Follows the Sym-JEPA training recipe (validated on propositional logic):
  L = L2(pred_latent, target_latent) + lambda * SIGReg + value_MSE

ValueHead predicts steps-to-QED (remaining tactics until proof completes),
trained on detached encoder outputs.

Evaluation (offline, no Lean kernel required):
  - Latent prediction error vs identity / random baselines
  - Tactic ranking: rank all 64 tactic actions by predicted value of the
    successor state; report top-1/top-5 of the ground-truth tactic.

Usage:
    python3 -u -m lean.train --seed 42 --max-epochs 20
"""

import argparse
import copy
import json
import os
import pickle
import random
import time
from pathlib import Path

# Telemetry / network hygiene: disable phone-home in dependencies.
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")   # Ray (lean-dojo dep)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lean.build import build_model

ROOT = Path(__file__).resolve().parent.parent


class TripleDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.rows = pickle.load(f)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "before": torch.tensor(r["before"], dtype=torch.long),
            "after": torch.tensor(r["after"], dtype=torch.long),
            "action": torch.tensor(r["action"], dtype=torch.long),
            "steps": torch.tensor(float(min(r["steps_to_qed"], 50))),
        }


def _pad_to(seqs, length):
    out = torch.zeros(len(seqs), length, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def collate_pad(batch):
    """Dynamic padding, quantized to multiples of 64.

    Unquantized per-batch shapes fragment the MPS caching allocator (every
    distinct length keeps its own cached blocks -> OOM over an epoch).
    Quantizing to {64,128,...,512} bounds the shape set to 8, so blocks
    are reused. Median state is 92 tokens, so this still avoids the ~5x
    waste of padding everything to 512."""
    m = max(max(len(b["before"]) for b in batch),
            max(len(b["after"]) for b in batch))
    length = min(((m + 63) // 64) * 64, 512)
    return {
        "before": _pad_to([b["before"] for b in batch], length),
        "after": _pad_to([b["after"] for b in batch], length),
        "action": torch.stack([b["action"] for b in batch]),
        "steps": torch.stack([b["steps"] for b in batch]),
    }


class BucketSampler(torch.utils.data.Sampler):
    """Length-bucketed batches, iterated shape-contiguously.

    Batches are length-homogeneous (sorted then chunked). Iteration groups
    batches by their quantized pad shape and runs each group contiguously
    (group order and within-group order shuffled per epoch): the MPS caching
    allocator then holds ONE shape pool at a time instead of all of them —
    alternating shapes accumulated ~15 GB of cached pools and OOMed."""

    def __init__(self, dataset, batch_size, seed=0, max_tokens=None):
        # Token-budget batching: B scales inversely with the quantized shape
        # (a fixed B=32 at L=512 needs >16 GB fwd+bwd with materialized
        # attention; B=16 at 512 and B=32 at 256 cost roughly the same).
        max_tokens = max_tokens or batch_size * 256
        lengths = [len(r["before"]) for r in dataset.rows]
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        self.batches, self.shapes = [], []
        i = 0
        while i < len(order):
            shape = min(((lengths[order[i]] + 63) // 64) * 64, 512)
            bsz = max(min(max_tokens // shape, 2 * batch_size), 8)
            chunk = order[i:i + bsz]
            # shape of the chunk is set by its longest member
            m = max(lengths[j] for j in chunk)
            shape = min(((m + 63) // 64) * 64, 512)
            bsz = max(min(max_tokens // shape, 2 * batch_size), 8)
            chunk = order[i:i + bsz]
            i += len(chunk)
            if len(chunk) >= 8:
                self.batches.append(chunk)
                self.shapes.append(shape)
        self.gen = torch.Generator().manual_seed(seed)

    def __iter__(self):
        groups = {}
        for bi, s in enumerate(self.shapes):
            groups.setdefault(s, []).append(bi)
        keys = list(groups)
        for k in torch.randperm(len(keys), generator=self.gen).tolist():
            idxs = groups[keys[k]]
            for j in torch.randperm(len(idxs), generator=self.gen).tolist():
                yield self.batches[idxs[j]]

    def __len__(self):
        return len(self.batches)


class ValueHead(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return F.softplus(self.net(z))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def evaluate(model, value_head, loader, device, max_batches=None, aux_head=None,
             value_log=False):
    """Latent prediction error + tactic top-1/top-5 + value MAE.

    Tactic ranking is reported two ways:
      - value-ranking (top1/top5): rank 64 actions by value of predicted successor
      - policy (ptop1/ptop5): direct aux-head classification of z_before (T2 metric)
    """
    model.eval()
    value_head.eval()
    if aux_head is not None:
        aux_head.eval()

    pred_errs, ident_errs, val_errs = [], [], []
    top1, top5, ptop1, ptop5, n_rank = 0, 0, 0, 0, 0
    num_actions = model.action_encoder.embedding.num_embeddings

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        action = batch["action"].to(device).unsqueeze(1)
        steps = batch["steps"].to(device)

        zb = model.encoder(before).last_hidden_state[:, 0]
        za = model.encoder(after).last_hidden_state[:, 0]
        act = model.action_encoder(action)
        pred = model.predict(zb.unsqueeze(1), act)[:, -1, :]

        pred_errs += (pred - za).pow(2).mean(1).tolist()
        ident_errs += (zb - za).pow(2).mean(1).tolist()
        vpred = value_head(za).squeeze(-1)
        if value_log:
            vpred = torch.expm1(vpred).clamp(0, 50)
        val_errs += (vpred - steps).abs().tolist()

        B = zb.shape[0]
        gold = action.squeeze(1)

        # Value-based ranking
        all_a = torch.arange(num_actions, device=device).view(1, -1, 1).expand(B, -1, -1)
        zb_rep = zb.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, 1, zb.shape[-1])
        acts = model.action_encoder(all_a.reshape(-1, 1))
        preds = model.predict(zb_rep, acts)[:, -1, :].view(B, num_actions, -1)
        scores = value_head(preds).squeeze(-1)
        rank = scores.argsort(1)
        top1 += (rank[:, 0] == gold).sum().item()
        top5 += (rank[:, :5] == gold.unsqueeze(1)).any(1).sum().item()

        # Policy head (T2)
        if aux_head is not None:
            logits = aux_head(zb)
            prank = logits.argsort(1, descending=True)
            ptop1 += (prank[:, 0] == gold).sum().item()
            ptop5 += (prank[:, :5] == gold.unsqueeze(1)).any(1).sum().item()
        n_rank += B

    out = {
        "lpe": float(np.mean(pred_errs)),
        "lpe_identity": float(np.mean(ident_errs)),
        "value_mae": float(np.mean(val_errs)),
        "top1": top1 / n_rank,
        "top5": top5 / n_rank,
    }
    if aux_head is not None:
        out["ptop1"] = ptop1 / n_rank
        out["ptop5"] = ptop5 / n_rank
    return out


def train(seed=42, max_epochs=20, batch_size=48, patience=5, min_epochs=8,
          aux_tactic_weight=1.0, value_weight=2.0, resume=None,
          value_log=False, value_into_encoder=False,
          config="large", data_dir="data/processed", run_tag="", accum_steps=1):
    """v2 recipe (default): auxiliary tactic-prediction CE on z_before
    (gradients flow into encoder -> tactic-discriminative embeddings, T2)
    and value_weight=2.0 (T3). v1 ablation: aux_tactic_weight=0, value_weight=1.

    v3 recipe (--value-log --value-into-encoder --aux-weight 2.0): log1p value
    target and value gradients INTO the encoder. Justified 2026-06-05: refit on
    frozen v2 embeddings failed T2/T3 criteria (MAE 4.86 > 3.0, ptop5 0.665 <
    0.70, theorem-disjoint) -> embeddings lack value/policy info; the encoder
    must learn it in-loop. T1 margin (0.26 vs 0.7) absorbs the interference
    risk reported by Value-Guided JEPA.

    resume: path to a *_best.pt checkpoint — restores weights and continues
    at checkpoint epoch + 1 (optimizer restarts; used for MPS-hang recovery).
    """
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model, sigreg, cfg = build_model(config)
    model, sigreg = model.to(device), sigreg.to(device)
    value_head = ValueHead(cfg["dim"], cfg["dim"]).to(device)
    aux_head = nn.Linear(cfg["dim"], cfg["num_tactics"]).to(device)
    lam = cfg["sigreg_lambda"]
    tag = run_tag or seed  # artifact suffix: lean_jepa_seed{tag}_*

    start_epoch = 0
    resume_val = float("inf")
    if resume:
        ck = torch.load(resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        value_head.load_state_dict(ck["value_head"])
        aux_head.load_state_dict(ck["aux_head"])
        start_epoch, resume_val = ck["epoch"] + 1, ck["val"]
        print(f"RESUMED from {resume}: ckpt epoch {ck['epoch']} "
              f"(val={resume_val:.4f}) -> continuing at epoch {start_epoch}")

    n_model = sum(p.numel() for p in model.parameters())
    n_vh = sum(p.numel() for p in value_head.parameters())
    n_aux = sum(p.numel() for p in aux_head.parameters())
    print(f"LEAN-JEPA  model={n_model:,}  value={n_vh:,}  aux={n_aux:,}  "
          f"total={n_model + n_vh + n_aux:,}")
    print(f"SIGReg lambda={lam}  batch={batch_size}  seed={seed}  "
          f"aux_w={aux_tactic_weight}  value_w={value_weight}")
    assert n_model + n_vh + n_aux <= 50_000_000

    train_ds = TripleDataset(ROOT / data_dir / "train.pkl")
    val_ds = TripleDataset(ROOT / data_dir / "val.pkl")
    if data_dir.endswith("_v4"):
        # Clean v4b: SHUFFLED IID batches (correct SGD) + dynamic per-batch
        # padding. No length-bucketing (that broke IID in the confounded v4).
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True, collate_fn=collate_pad)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_pad)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"train={len(train_ds):,}  val={len(val_ds):,}  data={data_dir}  config={config} "
          f"micro_batch={batch_size} accum={accum_steps} eff_batch={batch_size*accum_steps}")

    params = list(model.parameters()) + list(value_head.parameters()) + list(aux_head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

    ckpt_dir, log_dir = ROOT / "checkpoints", ROOT / "logs"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    best_val, patience_ct = resume_val, 0
    # Seed best states from the resumed checkpoint so the final
    # best-model load works even if no later epoch improves.
    best_model = copy.deepcopy(model.state_dict()) if resume else None
    best_vh = copy.deepcopy(value_head.state_dict()) if resume else None
    best_aux = copy.deepcopy(aux_head.state_dict()) if resume else None
    history = []
    t0 = time.time()

    # Advance LR schedule to the resume point.
    for _ in range(start_epoch):
        sched.step()

    for epoch in range(start_epoch, max_epochs):
        te = time.time()
        model.train()
        value_head.train()
        aux_head.train()
        ep = {"pred": [], "sig": [], "val": [], "aux": []}

        opt.zero_grad()
        for bi, batch in enumerate(train_loader):
            before = batch["before"].to(device)
            after = batch["after"].to(device)
            action = batch["action"].to(device).unsqueeze(1)
            steps = batch["steps"].to(device)

            zb = model.encoder(before).last_hidden_state[:, 0]
            za = model.encoder(after).last_hidden_state[:, 0]
            act = model.action_encoder(action)
            pred = model.predict(zb.unsqueeze(1), act)[:, -1, :]

            pred_loss = (pred - za).pow(2).mean()
            sig_loss = sigreg(torch.cat([zb, za], 0).unsqueeze(0))
            val_in = za if value_into_encoder else za.detach()
            val_tgt = torch.log1p(steps) if value_log else steps
            val_loss = F.mse_loss(value_head(val_in).squeeze(-1), val_tgt)
            # Auxiliary tactic prediction: gradients flow into encoder
            # to make z_before tactic-discriminative (T2).
            aux_loss = F.cross_entropy(aux_head(zb), action.squeeze(1))

            # Gradient accumulation: micro-batch backward scaled by 1/accum,
            # optimizer step every accum_steps -> effective batch = micro*accum
            # (matches v3's batch 48 while fitting 512-ctx in memory).
            loss = (pred_loss + lam * sig_loss + value_weight * val_loss
                    + aux_tactic_weight * aux_loss) / accum_steps
            loss.backward()
            if (bi + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()

            ep["pred"].append(pred_loss.item())
            ep["sig"].append(sig_loss.item())
            ep["val"].append(val_loss.item())
            ep["aux"].append(aux_loss.item())
            if (bi + 1) % 5 == 0 and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        sched.step()

        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        metrics = evaluate(model, value_head, val_loader, device, max_batches=40,
                           aux_head=aux_head, value_log=value_log)
        entry = {"epoch": epoch, "pred": float(np.mean(ep["pred"])),
                 "sig": float(np.mean(ep["sig"])), "val_h": float(np.mean(ep["val"])),
                 "aux": float(np.mean(ep["aux"])),
                 **metrics, "time": time.time() - te}
        history.append(entry)
        print(f"  Epoch {epoch:3d} | pred={entry['pred']:.4f} sig={entry['sig']:.4f} "
              f"val_h={entry['val_h']:.3f} aux={entry['aux']:.3f} | "
              f"lpe={metrics['lpe']:.4f} ident={metrics['lpe_identity']:.4f} "
              f"vMAE={metrics['value_mae']:.2f} "
              f"ptop1={metrics.get('ptop1', 0):.3f} ptop5={metrics.get('ptop5', 0):.3f} "
              f"time={entry['time']:.0f}s")

        if metrics["lpe"] < best_val:
            best_val, patience_ct = metrics["lpe"], 0
            best_model = copy.deepcopy(model.state_dict())
            best_vh = copy.deepcopy(value_head.state_dict())
            best_aux = copy.deepcopy(aux_head.state_dict())
            torch.save({"model": best_model, "value_head": best_vh,
                        "aux_head": best_aux,
                        "epoch": epoch, "val": best_val, "seed": seed},
                       ckpt_dir / f"lean_jepa_seed{tag}_best.pt")
        else:
            patience_ct += 1
            if patience_ct >= patience and epoch >= min_epochs:
                print(f"  Early stop at {epoch}")
                break

        # Always keep the latest weights too (analysis/refit; best-LPE
        # selection can freeze heads early — measured 2026-06-05).
        torch.save({"model": model.state_dict(), "value_head": value_head.state_dict(),
                    "aux_head": aux_head.state_dict(),
                    "epoch": epoch, "val": metrics["lpe"], "seed": seed},
                   ckpt_dir / f"lean_jepa_seed{tag}_last.pt")

    with open(log_dir / f"lean_jepa_seed{tag}_log.json", "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(best_model)
    value_head.load_state_dict(best_vh)
    aux_head.load_state_dict(best_aux)
    final = evaluate(model, value_head, val_loader, device, aux_head=aux_head,
                     value_log=value_log)
    final["train_time"] = time.time() - t0
    final["seed"] = seed
    with open(log_dir / f"lean_jepa_seed{tag}_results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nFINAL  lpe={final['lpe']:.4f} vs identity={final['lpe_identity']:.4f}")
    print(f"       ptop1={final.get('ptop1', 0):.3f}  ptop5={final.get('ptop5', 0):.3f}  "
          f"value_mae={final['value_mae']:.2f}")
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=2.0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--value-log", action="store_true")
    p.add_argument("--value-into-encoder", action="store_true")
    p.add_argument("--config", default="large")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--run-tag", default="")
    p.add_argument("--accum-steps", type=int, default=1)
    args = p.parse_args()
    train(args.seed, args.max_epochs, args.batch_size,
          aux_tactic_weight=args.aux_weight, value_weight=args.value_weight,
          resume=args.resume, value_log=args.value_log,
          value_into_encoder=args.value_into_encoder,
          config=args.config, data_dir=args.data_dir, run_tag=args.run_tag,
          accum_steps=args.accum_steps)
