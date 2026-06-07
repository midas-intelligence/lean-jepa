"""
Build Lean-JEPA: a latent world model over Lean 4 proof states.

Assembles:
- ProofStateEncoder (ours) — replaces tree encoder
- TacticEncoder (ours)     — embeds tactic head ids
- ARPredictor, SIGReg, JEPA (LeWorldModel, unmodified)
"""

import torch.nn as nn

from lewm.module import ARPredictor, SIGReg
from lewm.jepa import JEPA

from .encoder import ProofStateEncoder


CONFIGS = {
    "large": {
        "dim": 384,
        "enc_layers": 8,
        "enc_heads": 8,
        "pred_depth": 6,
        "pred_heads": 8,
        "pred_dim_head": 48,
        "pred_mlp_dim": 1536,
        "sigreg_lambda": 0.09,
        "dropout": 0.1,
        "vocab_size": 8000,
        "num_tactics": 64,
        "max_len": 256,
    },
    # v4: goal-preserving truncation at 512 (fits 99.0% of states; the
    # 256 head-truncation cut the goal itself on the hardest 8%).
    "large512": {
        "dim": 384,
        "enc_layers": 8,
        "enc_heads": 8,
        "pred_depth": 6,
        "pred_heads": 8,
        "pred_dim_head": 48,
        "pred_mlp_dim": 1536,
        "sigreg_lambda": 0.09,
        "dropout": 0.1,
        "vocab_size": 8000,
        "num_tactics": 64,
        "max_len": 512,
    },
}


class TacticEncoder(nn.Module):
    """Discrete tactic-head embedding (analogous to rewrite-rule embedding)."""

    def __init__(self, num_tactics=64, emb_dim=384):
        super().__init__()
        self.embedding = nn.Embedding(num_tactics, emb_dim)

    def forward(self, tactic_ids):  # (B, T) -> (B, T, D)
        return self.embedding(tactic_ids)


def build_model(size="large"):
    cfg = CONFIGS[size]
    dim = cfg["dim"]

    encoder = ProofStateEncoder(
        vocab_size=cfg["vocab_size"],
        dim=dim,
        num_layers=cfg["enc_layers"],
        num_heads=cfg["enc_heads"],
        dropout=cfg["dropout"],
        max_len=cfg.get("max_len", 256),
    )

    predictor = ARPredictor(
        num_frames=1,
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        depth=cfg["pred_depth"],
        heads=cfg["pred_heads"],
        dim_head=cfg["pred_dim_head"],
        mlp_dim=cfg["pred_mlp_dim"],
        dropout=cfg["dropout"],
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=TacticEncoder(cfg["num_tactics"], dim),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    )

    sigreg = SIGReg(knots=17, num_proj=1024)
    return model, sigreg, cfg
