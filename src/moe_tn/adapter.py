from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass
class OlmoeAdapter:
    """Adapter for both ModuleList and stacked-tensor Transformers OLMoE layouts."""

    model: torch.nn.Module
    layer_index: int

    def __post_init__(self) -> None:
        try:
            self.block = self.model.model.layers[self.layer_index].mlp
        except (AttributeError, IndexError) as error:
            raise ValueError("Expected an OlmoeForCausalLM-style model") from error
        if isinstance(self.block.experts, torch.nn.ModuleList):
            self.layout = "module_list"
        elif hasattr(self.block.experts, "gate_up_proj") and hasattr(
            self.block.experts, "down_proj"
        ):
            self.layout = "stacked"
        else:
            raise ValueError(
                "Unsupported OLMoE expert layout; expected ModuleList experts or "
                "stacked gate_up_proj/down_proj tensors"
            )

    @property
    def num_experts(self) -> int:
        if self.layout == "module_list":
            return len(self.block.experts)
        return int(self.block.experts.gate_up_proj.shape[0])

    @property
    def device(self) -> torch.device:
        return next(self.block.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        if self.layout == "module_list":
            return next(self.block.experts[0].parameters()).dtype
        return self.block.experts.gate_up_proj.dtype

    @property
    def top_k(self) -> int:
        if hasattr(self.block, "top_k"):
            return int(self.block.top_k)
        return int(self.block.gate.top_k)

    def weight(self, expert: int, projection: str) -> torch.Tensor:
        if projection not in PROJECTIONS:
            raise KeyError(projection)
        if self.layout == "module_list":
            return getattr(self.block.experts[expert], projection).weight
        if projection == "down_proj":
            return self.block.experts.down_proj[expert]
        gate_up = self.block.experts.gate_up_proj[expert]
        midpoint = gate_up.shape[0] // 2
        return gate_up[:midpoint] if projection == "gate_proj" else gate_up[midpoint:]

    def expert_forward(self, expert: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layout == "module_list":
            return self.block.experts[expert](hidden_states)
        gate = F.linear(hidden_states, self.weight(expert, "gate_proj"))
        up = F.linear(hidden_states, self.weight(expert, "up_proj"))
        intermediate = self.block.experts.act_fn(gate) * up
        return F.linear(intermediate, self.weight(expert, "down_proj"))

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.block.gate(hidden_states)
        return output[0] if isinstance(output, tuple) else output

    @contextmanager
    def merged_pair(self, first: int, second: int, alpha: float = 0.5) -> Iterator[None]:
        """Alias two expert slots to the same averaged expert, then restore exactly.

        Keeping both router IDs but giving them identical parameters is functionally
        equivalent to routing both IDs to one merged expert. It isolates merge damage
        without changing the router or the rest of the model.
        """
        if first == second:
            raise ValueError("Cannot merge an expert with itself")
        originals: dict[tuple[int, str], torch.Tensor] = {}
        with torch.no_grad():
            for expert in (first, second):
                for projection in PROJECTIONS:
                    originals[(expert, projection)] = self.weight(expert, projection).detach().clone()
            for projection in PROJECTIONS:
                left = originals[(first, projection)].float()
                right = originals[(second, projection)].float()
                merged = (alpha * left + (1.0 - alpha) * right).to(self.dtype)
                self.weight(first, projection).copy_(merged)
                self.weight(second, projection).copy_(merged)
        try:
            yield
        finally:
            with torch.no_grad():
                for (expert, projection), value in originals.items():
                    self.weight(expert, projection).copy_(value)
