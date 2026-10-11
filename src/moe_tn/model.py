from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OlmoeConfig, OlmoeForCausalLM


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tiny_random_olmoe(seed: int) -> OlmoeForCausalLM:
    torch.manual_seed(seed)
    config = OlmoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=6,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    return OlmoeForCausalLM(config)


def load_model(model_config: dict[str, Any], seed: int):
    model_id = model_config["id"]
    if model_id == "__tiny_random_olmoe__":
        model = tiny_random_olmoe(seed)
        model.to(best_device())
        return model, None

    dtype = DTYPES[model_config["dtype"]]
    revision = model_config.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to(best_device())
    model.eval()
    return model, tokenizer
