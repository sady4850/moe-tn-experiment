from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def evaluate_nll(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    batch_size: int = 1,
) -> dict[str, float | int]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, blocks.shape[0], batch_size):
        input_ids = blocks[start : start + batch_size].to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += labels.numel()
        del logits, loss, input_ids, labels
    nll = total_loss / total_tokens
    return {"nll": nll, "perplexity": math.exp(nll), "tokens": total_tokens}

