from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable

import torch

from .adapter import OlmoeAdapter, PROJECTIONS


def sample_pairs(num_experts: int, count: int, seed: int) -> list[tuple[int, int]]:
    all_pairs = list(itertools.combinations(range(num_experts), 2))
    if count > len(all_pairs):
        raise ValueError(f"Requested {count} pairs, but only {len(all_pairs)} exist")
    return sorted(random.Random(seed).sample(all_pairs, count))


def _cosine_parts(parts_a: Iterable[torch.Tensor], parts_b: Iterable[torch.Tensor]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(parts_a, parts_b, strict=True):
        left = left.detach().float()
        right = right.detach().float()
        dot += torch.sum(left * right).item()
        norm_a += torch.sum(left.square()).item()
        norm_b += torch.sum(right.square()).item()
    return dot / math.sqrt(norm_a * norm_b)


def weight_cosine(adapter: OlmoeAdapter, first: int, second: int) -> float:
    return _cosine_parts(
        (adapter.weight(first, name) for name in PROJECTIONS),
        (adapter.weight(second, name) for name in PROJECTIONS),
    )


@torch.inference_mode()
def capture_layer_inputs(
    model: torch.nn.Module,
    adapter: OlmoeAdapter,
    calibration_blocks: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    token_count = 0

    def hook(_module, args):
        nonlocal token_count
        if token_count >= max_tokens:
            return
        hidden = args[0].detach().reshape(-1, args[0].shape[-1])
        take = min(max_tokens - token_count, hidden.shape[0])
        captured.append(hidden[:take].to(device="cpu"))
        token_count += take

    handle = adapter.block.register_forward_pre_hook(hook)
    try:
        for block in calibration_blocks:
            model(input_ids=block.unsqueeze(0).to(adapter.device), use_cache=False)
            if token_count >= max_tokens:
                break
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("The target MoE layer hook captured no activations")
    return torch.cat(captured, dim=0)[:max_tokens]


@torch.inference_mode()
def activation_cosine_matrix(
    adapter: OlmoeAdapter,
    hidden_states: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    """Exact pairwise cosine over expert outputs on the same hidden states."""
    experts = adapter.num_experts
    gram = torch.zeros((experts, experts), dtype=torch.float64)
    for start in range(0, hidden_states.shape[0], chunk_tokens):
        chunk = hidden_states[start : start + chunk_tokens].to(
            device=adapter.device, dtype=adapter.dtype
        )
        outputs = torch.stack(
            [adapter.expert_forward(index, chunk) for index in range(experts)], dim=0
        )
        flat = outputs.float().reshape(experts, -1)
        gram += (flat @ flat.T).double().cpu()
        del outputs, flat, chunk
    norms = torch.sqrt(torch.clamp(torch.diag(gram), min=1e-30))
    return gram / (norms[:, None] * norms[None, :])


@torch.inference_mode()
def router_statistics(adapter: OlmoeAdapter, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden = hidden_states.to(device=adapter.device, dtype=adapter.dtype)
    logits = adapter.router_logits(hidden).float()
    probabilities = torch.softmax(logits, dim=-1)
    top_k = adapter.top_k
    selected = torch.topk(probabilities, top_k, dim=-1).indices
    mask = torch.nn.functional.one_hot(selected, num_classes=adapter.num_experts).sum(dim=1).float()
    usage = mask.mean(dim=0)
    co_routing = mask.T @ mask / mask.shape[0]
    return {
        "usage": usage.cpu().double(),
        "mean_probability": probabilities.mean(dim=0).cpu().double(),
        "co_routing": co_routing.cpu().double(),
    }


def effective_rank_from_singular_values(singular_values: torch.Tensor, error: float) -> int:
    if not 0.0 < error < 1.0:
        raise ValueError("Reconstruction error must be between zero and one")
    energy = singular_values.double().square()
    threshold = (1.0 - error * error) * energy.sum()
    retained = torch.cumsum(energy, dim=0)
    return int(torch.searchsorted(retained, threshold).item() + 1)


def _svdvals(matrix: torch.Tensor) -> torch.Tensor:
    # CUDA uses cuSOLVER; CPU is the reliable local fallback (MPS SVD support varies).
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.linalg.svdvals(matrix.detach().to(device=device, dtype=torch.float32)).cpu()


def _joint_matrix(left: torch.Tensor, right: torch.Tensor, projection: str) -> torch.Tensor:
    # gate/up share the hidden input mode -> stack rows and measure joint row space.
    # down shares the hidden output mode -> stack columns and measure joint column space.
    dimension = 1 if projection == "down_proj" else 0
    return torch.cat((left, right), dim=dimension)


def effective_rank_signals(
    adapter: OlmoeAdapter,
    pairs: list[tuple[int, int]],
    errors: list[float],
) -> dict[tuple[int, int], dict[str, float | int]]:
    unique_experts = sorted({expert for pair in pairs for expert in pair})
    individual: dict[tuple[int, str], torch.Tensor] = {}
    for expert in unique_experts:
        for projection in PROJECTIONS:
            individual[(expert, projection)] = _svdvals(adapter.weight(expert, projection))

    results: dict[tuple[int, int], dict[str, float | int]] = {}
    for first, second in pairs:
        joint_values: dict[str, torch.Tensor] = {}
        for projection in PROJECTIONS:
            joint = _joint_matrix(
                adapter.weight(first, projection),
                adapter.weight(second, projection),
                projection,
            )
            joint_values[projection] = _svdvals(joint)

        pair_result: dict[str, float | int] = {}
        for error in errors:
            label = f"{error:.4f}"
            deltas: list[int] = []
            normalized: list[float] = []
            for projection in PROJECTIONS:
                rank_first = effective_rank_from_singular_values(
                    individual[(first, projection)], error
                )
                rank_second = effective_rank_from_singular_values(
                    individual[(second, projection)], error
                )
                rank_joint = effective_rank_from_singular_values(joint_values[projection], error)
                reference = max(rank_first, rank_second)
                delta = rank_joint - reference
                pair_result[f"rank_{projection}_first_eps_{label}"] = rank_first
                pair_result[f"rank_{projection}_second_eps_{label}"] = rank_second
                pair_result[f"rank_{projection}_joint_eps_{label}"] = rank_joint
                deltas.append(delta)
                normalized.append(delta / max(reference, 1))
            pair_result[f"tn_delta_rank_eps_{label}"] = sum(deltas)
            pair_result[f"tn_delta_rank_normalized_eps_{label}"] = sum(normalized) / len(normalized)
        results[(first, second)] = pair_result
    return results


def compute_signals(
    adapter: OlmoeAdapter,
    pairs: list[tuple[int, int]],
    hidden_states: torch.Tensor,
    errors: list[float],
    activation_chunk_tokens: int,
) -> list[dict[str, float | int]]:
    activation = activation_cosine_matrix(adapter, hidden_states, activation_chunk_tokens)
    router = router_statistics(adapter, hidden_states)
    ranks = effective_rank_signals(adapter, pairs, errors)
    rows: list[dict[str, float | int]] = []
    for first, second in pairs:
        row: dict[str, float | int] = {
            "expert_first": first,
            "expert_second": second,
            "weight_cosine": weight_cosine(adapter, first, second),
            "activation_cosine": float(activation[first, second]),
            "router_co_routing": float(router["co_routing"][first, second]),
            "router_usage_first": float(router["usage"][first]),
            "router_usage_second": float(router["usage"][second]),
            "router_mean_probability_first": float(router["mean_probability"][first]),
            "router_mean_probability_second": float(router["mean_probability"][second]),
        }
        row.update(ranks[(first, second)])
        rows.append(row)
    return rows
