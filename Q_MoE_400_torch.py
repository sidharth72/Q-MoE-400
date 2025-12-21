import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MoEConfig:
    vocab_size: int = 50257
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 16
    max_seq_len: int = 512
    num_experts: int = 8
    top_k: int = 2
    ffn_dim: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class DenseNoBias(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.float32))
        nn.init.normal_(self.kernel, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.kernel


def causal_mask(t: int, *, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((t, t), dtype=torch.bool, device=device))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = DenseNoBias(d_model, d_model)
        self.k_proj = DenseNoBias(d_model, d_model)
        self.v_proj = DenseNoBias(d_model, d_model)
        self.out_proj = DenseNoBias(d_model, d_model)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.einsum("bthd,bshd->bhts", q, k) * scale
        att = att.masked_fill(~attn_mask.view(1, 1, t, t), -1e30)
        att = torch.softmax(att, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", att, v).contiguous()
        out = out.view(b, t, d)
        return self.out_proj(out)


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = DenseNoBias(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        denom = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        gates = topk_vals / denom
        return topk_idx, gates


class ExpertMLPBank(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim, dtype=torch.float32))
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden_dim, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model, dtype=torch.float32))
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model, dtype=torch.float32))

        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor) -> torch.Tensor:
        w1 = self.w1.index_select(0, expert_idx)
        b1 = self.b1.index_select(0, expert_idx)
        w2 = self.w2.index_select(0, expert_idx)
        b2 = self.b2.index_select(0, expert_idx)

        h = torch.einsum("nd,ndh->nh", x, w1) + b1
        h = torch.nn.functional.silu(h)
        y = torch.einsum("nh,nhd->nd", h, w2) + b2
        return y


class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = Router(d_model=d_model, num_experts=num_experts, top_k=top_k)
        self.experts = ExpertMLPBank(d_model=d_model, hidden_dim=hidden_dim, num_experts=num_experts)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        topk_idx, gates = self.router(x)

        x_flat = x.reshape(b * t, d)
        idx_flat = topk_idx.reshape(b * t, self.top_k)
        gates_flat = gates.reshape(b * t, self.top_k)

        y = torch.zeros_like(x_flat)
        for j in range(self.top_k):
            e_idx = idx_flat[:, j]
            y_j = self.experts(x_flat, e_idx)
            y = y + y_j * gates_flat[:, j : j + 1]
        return y.reshape(b, t, d)


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        self.rmsnorm_0 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.rmsnorm_1 = RMSNorm(d_model)
        self.moe = MoEFeedForward(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.rmsnorm_0(x)
        x = x + self.attn(h, attn_mask=attn_mask)
        h = self.rmsnorm_1(x)
        x = x + self.moe(h)
        return x


class MoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=cfg.d_model,
                    num_heads=cfg.num_heads,
                    hidden_dim=cfg.ffn_dim,
                    num_experts=cfg.num_experts,
                    top_k=cfg.top_k,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.rmsnorm_f = RMSNorm(cfg.d_model)
        self.lm_head = DenseNoBias(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        device = input_ids.device

        if t > self.cfg.max_seq_len:
            raise ValueError("input length exceeds max_seq_len")

        tok = self.tok_emb(input_ids)
        pos_idx = torch.arange(t, device=device).unsqueeze(0)
        pos = self.pos_emb(pos_idx)
        x = tok + pos

        attn_mask = causal_mask(t, device=device)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.rmsnorm_f(x)
        logits = self.lm_head(x)
        return logits


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def load_model_from_checkpoint(
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MoETransformer:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict: Dict[str, Any] = ckpt.get("config", {})
    cfg = MoEConfig(**cfg_dict)
    model = MoETransformer(cfg).to(torch.device(device))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(dtype=dtype)
    model.eval()
    return model
