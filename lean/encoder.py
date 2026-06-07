"""
Proof state encoder for Lean 4 goal states.

Replaces Sym-JEPA's TreeEncoder. Proof states are token sequences
(MAX_STATE_TOKENS=256, vocab 8000) — encoded with a pre-norm Transformer
and aggregated via a learned CLS query, matching the interface
LeWM's JEPA expects:

    output = encoder(token_ids)
    latent = output.last_hidden_state[:, 0]
"""

import torch
import torch.nn as nn

from .data import MAX_STATE_TOKENS, PAD


class EncoderOutput:
    """Matches HuggingFace output interface used by LeWM's JEPA."""
    def __init__(self, cls_embedding):
        self.last_hidden_state = cls_embedding.unsqueeze(1)  # (B, 1, D)


class ProofStateEncoder(nn.Module):
    def __init__(self, vocab_size=8000, dim=384, num_layers=8, num_heads=8,
                 dropout=0.1, max_len=MAX_STATE_TOKENS):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.input_norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                "norm2": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim), nn.Dropout(dropout),
                ),
                "ls1": nn.ParameterList([nn.Parameter(torch.ones(dim) * 1e-4)]),
                "ls2": nn.ParameterList([nn.Parameter(torch.ones(dim) * 1e-4)]),
            }))

        self.output_norm = nn.LayerNorm(dim)
        self.cls_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cls_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cls_norm = nn.LayerNorm(dim)

        class Config:
            hidden_size = dim
        self.config = Config()

    def forward(self, token_ids, interpolate_pos_encoding=False):
        """
        token_ids: (B, MAX_STATE_TOKENS) int64
        Returns EncoderOutput with .last_hidden_state (B, 1, D)
        """
        B, T = token_ids.shape
        device = token_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.input_norm(self.tok_emb(token_ids) + self.pos_emb(pos))

        padding_mask = token_ids == PAD

        def _block(layer, x):
            normed = layer["norm1"](x)
            # need_weights=False enables the fused memory-efficient attention
            # path; the default (True) materializes B x H x L x L weight
            # matrices and retains them for backward — OOM at L=512.
            attn_out, _ = layer["attn"](normed, normed, normed,
                                        key_padding_mask=padding_mask,
                                        need_weights=False)
            x = x + layer["ls1"][0] * attn_out
            x = x + layer["ls2"][0] * layer["ffn"](layer["norm2"](x))
            return x

        for layer in self.layers:
            if self.training and T > 256:
                # Gradient checkpointing on long sequences: recompute the
                # block in backward instead of retaining activations
                # (~8x activation memory cut; ends MPS cap-chasing at 512).
                x = torch.utils.checkpoint.checkpoint(
                    lambda x, layer=layer: _block(layer, x), x,
                    use_reentrant=False)
            else:
                x = _block(layer, x)

        cls = self.cls_query.expand(B, -1, -1)
        cls_out, _ = self.cls_attn(
            self.cls_norm(cls), self.output_norm(x), x,
            key_padding_mask=padding_mask, need_weights=False,
        )
        return EncoderOutput(cls_out.squeeze(1))
