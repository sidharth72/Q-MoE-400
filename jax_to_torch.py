from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn

VOCAB_SIZE: int = 50257
D_MODEL: int = 768
NUM_LAYERS: int = 12
NUM_HEADS: int = 16
MAX_SEQ_LEN: int = 512
NUM_EXPERTS: int = 8
TOP_K: int = 2
FFN_DIM: int = 2048


def _flatten_dict(tree: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_dict(v, new_prefix))
        return out
    out[prefix] = tree
    return out


def _to_torch(x: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    if hasattr(x, "__array__"):
        arr = np.asarray(x)
        return torch.from_numpy(arr).to(dtype=dtype)
    raise TypeError(f"Cannot convert type to torch: {type(x)}")


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


def _causal_mask(t: int, *, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((t, t), dtype=torch.bool, device=device))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = D_MODEL, num_heads: int = NUM_HEADS):
        super().__init__()
        assert d_model % num_heads == 0
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
    def __init__(self, d_model: int = D_MODEL, num_experts: int = NUM_EXPERTS, top_k: int = TOP_K):
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
    def __init__(
        self,
        d_model: int = D_MODEL,
        hidden_dim: int = FFN_DIM,
        num_experts: int = NUM_EXPERTS,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim, dtype=torch.float32)) # [8, 768, 2048]
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
    def __init__(
        self,
        d_model: int = D_MODEL,
        hidden_dim: int = FFN_DIM,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K,
    ):
        super().__init__()
        self.router = Router(d_model=d_model, num_experts=num_experts, top_k=top_k)
        self.experts = ExpertMLPBank(d_model=d_model, hidden_dim=hidden_dim, num_experts=num_experts)
        self.num_experts = num_experts
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
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        hidden_dim: int = FFN_DIM,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K,
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
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        max_seq_len: int = MAX_SEQ_LEN,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K,
        ffn_dim: int = FFN_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.top_k = top_k
        self.ffn_dim = ffn_dim

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_dim=ffn_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                )
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm_f = RMSNorm(d_model)
        self.lm_head = DenseNoBias(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        device = input_ids.device

        tok = self.tok_emb(input_ids)
        pos_idx = torch.arange(t, device=device).unsqueeze(0)
        pos = self.pos_emb(pos_idx)
        x = tok + pos

        attn_mask = _causal_mask(t, device=device)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.rmsnorm_f(x)
        logits = self.lm_head(x)
        return logits


def _get_arr(flat: Dict[str, Any], key: str) -> Any:
    if key not in flat:
        available = "\n".join(sorted(flat.keys())[:200])
        raise KeyError(f"Missing key: {key}\nFirst keys:\n{available}")
    return flat[key]


def load_orbax_params(step_dir: str) -> Dict[str, Any]:
    ckpt = Checkpointer(PyTreeCheckpointHandler()).restore(step_dir)
    ts = ckpt["train_state"]
    return ts["params"]


def inject_weights_from_jax(model: MoETransformer, jax_params: Dict[str, Any]) -> None:
    flat = _flatten_dict(jax_params)

    with torch.no_grad():
        model.tok_emb.weight.copy_(_to_torch(_get_arr(flat, "tok_emb.embedding")))
        model.pos_emb.weight.copy_(_to_torch(_get_arr(flat, "pos_emb.embedding")))
        model.lm_head.kernel.copy_(_to_torch(_get_arr(flat, "lm_head.kernel")))
        model.rmsnorm_f.scale.copy_(_to_torch(_get_arr(flat, "RMSNorm_0.scale")))

        for i, blk in enumerate(model.blocks):
            pfx = f"Block_{i}"

            blk.rmsnorm_0.scale.copy_(_to_torch(_get_arr(flat, f"{pfx}.RMSNorm_0.scale")))
            blk.rmsnorm_1.scale.copy_(_to_torch(_get_arr(flat, f"{pfx}.RMSNorm_1.scale")))

            blk.attn.q_proj.kernel.copy_(_to_torch(_get_arr(flat, f"{pfx}.MultiHeadAttention_0.q_proj.kernel")))
            blk.attn.k_proj.kernel.copy_(_to_torch(_get_arr(flat, f"{pfx}.MultiHeadAttention_0.k_proj.kernel")))
            blk.attn.v_proj.kernel.copy_(_to_torch(_get_arr(flat, f"{pfx}.MultiHeadAttention_0.v_proj.kernel")))
            blk.attn.out_proj.kernel.copy_(_to_torch(_get_arr(flat, f"{pfx}.MultiHeadAttention_0.out_proj.kernel")))

            blk.moe.router.gate.kernel.copy_(_to_torch(_get_arr(flat, f"{pfx}.Router_0.gate.kernel")))

            blk.moe.experts.w1.copy_(_to_torch(_get_arr(flat, f"{pfx}.ExpertMLP_0.w1")))
            blk.moe.experts.b1.copy_(_to_torch(_get_arr(flat, f"{pfx}.ExpertMLP_0.b1")))
            blk.moe.experts.w2.copy_(_to_torch(_get_arr(flat, f"{pfx}.ExpertMLP_0.w2")))
            blk.moe.experts.b2.copy_(_to_torch(_get_arr(flat, f"{pfx}.ExpertMLP_0.b2")))


@dataclass(frozen=True)
class ExportConfig:
    step_dir: str = "/teamspace/studios/this_studio/checkpoints/step_75000"
    out_path: str = "/teamspace/studios/this_studio/moe_torch_state_dict_75000.pt"
    device: str = "cpu"


def main() -> None:
    cfg = ExportConfig()

    jax_params = load_orbax_params(cfg.step_dir)
    model = MoETransformer().to(torch.device(cfg.device))
    model.eval()

    inject_weights_from_jax(model, jax_params)

    dummy = torch.randint(0, VOCAB_SIZE, (2, 8), dtype=torch.long, device=model.tok_emb.weight.device)
    with torch.no_grad():
        logits = model(dummy)
    print("dummy input shape:", tuple(dummy.shape))
    print("logits shape:", tuple(logits.shape))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE,
                "d_model": D_MODEL,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "max_seq_len": MAX_SEQ_LEN,
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "ffn_dim": FFN_DIM,
            },
        },
        cfg.out_path,
    )
    print("saved:", cfg.out_path)


if __name__ == "__main__":
    main()