from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int | None = 50
    top_p: float | None = 0.95
    eos_token_id: int | None = None
    pad_token_id: int | None = None


def top_k_top_p_filtering(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must be [B, V]")

    filtered = logits

    if top_k is not None and top_k > 0:
        k = min(top_k, filtered.shape[-1])
        kth = torch.topk(filtered, k=k, dim=-1).values[:, -1].unsqueeze(-1)
        filtered = torch.where(filtered < kth, torch.full_like(filtered, -float("inf")), filtered)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)

        remove = cumprobs > top_p
        remove[:, 0] = False

        sorted_logits = torch.where(remove, torch.full_like(sorted_logits, -float("inf")), sorted_logits)
        filtered = torch.empty_like(filtered).scatter(-1, sorted_idx, sorted_logits)

    return filtered


@torch.no_grad()
def generate(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cfg: GenerationConfig,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [B, T]")
    if cfg.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if cfg.temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    device = input_ids.device
    out = input_ids

    for _ in range(cfg.max_new_tokens):
        logits = model(out)
        next_logits = logits[:, -1, :]

        if cfg.eos_token_id is not None:
            eos_mask = out.eq(cfg.eos_token_id).any(dim=-1)
            if eos_mask.any():
                next_logits = torch.where(
                    eos_mask.unsqueeze(-1),
                    torch.full_like(next_logits, -float("inf")),
                    next_logits,
                )
                next_logits[eos_mask, cfg.eos_token_id] = 0.0

        if cfg.do_sample:
            scaled = next_logits / cfg.temperature
            scaled = top_k_top_p_filtering(scaled, top_k=cfg.top_k, top_p=cfg.top_p)
            probs = torch.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        out = torch.cat([out, next_token.to(device=device)], dim=-1)

        if cfg.eos_token_id is not None:
            if bool(out[:, -1].eq(cfg.eos_token_id).all()):
                break

    if cfg.pad_token_id is not None and cfg.eos_token_id is not None:
        eos_mask = out.eq(cfg.eos_token_id)
        if eos_mask.any():
            first_eos = torch.argmax(eos_mask.to(torch.int64), dim=-1)
            for i in range(out.shape[0]):
                pos = int(first_eos[i].item())
                if out[i, pos].item() == cfg.eos_token_id and pos + 1 < out.shape[1]:
                    out[i, pos + 1 :] = cfg.pad_token_id

    return out


def get_tokenizer(*, allow_download: bool = False) -> Tuple[callable, callable]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.encode, enc.decode
    except Exception:
        pass

    try:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=not allow_download)

        def encode(text: str):
            return tok.encode(text, add_special_tokens=False)

        def decode(ids):
            return tok.decode(ids)

        return encode, decode
    except Exception as e:
        raise RuntimeError(
            "Tokenizer unavailable. Install 'tiktoken' or install 'transformers' with a cached GPT-2 tokenizer."
        ) from e
