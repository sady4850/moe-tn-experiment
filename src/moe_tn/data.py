from __future__ import annotations

from typing import Any

import torch


def synthetic_blocks(
    *, seed: int, vocab_size: int, count: int, block_size: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(3, vocab_size, (count, block_size), generator=generator)


def text_blocks(tokenizer, data_config: dict[str, Any], split: str, count: int) -> torch.Tensor:
    from datasets import load_dataset

    dataset = load_dataset(
        data_config["dataset"],
        data_config.get("subset"),
        split=split,
    )
    block_size = int(data_config["block_size"])
    text_field = data_config.get("text_field", "text")
    eos = tokenizer.eos_token_id
    token_buffer: list[int] = []
    blocks: list[list[int]] = []
    for row in dataset:
        text = row[text_field]
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_buffer.extend(ids)
        if eos is not None:
            token_buffer.append(eos)
        while len(token_buffer) >= block_size and len(blocks) < count:
            blocks.append(token_buffer[:block_size])
            del token_buffer[:block_size]
        if len(blocks) == count:
            break
    if len(blocks) != count:
        raise RuntimeError(f"Requested {count} blocks from {split}, got {len(blocks)}")
    return torch.tensor(blocks, dtype=torch.long)


def build_data(config: dict[str, Any], model, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    data_config = config["data"]
    if tokenizer is None:
        calibration = synthetic_blocks(
            seed=config["seed"] + 1,
            vocab_size=model.config.vocab_size,
            count=data_config["calibration_blocks"],
            block_size=data_config["block_size"],
        )
        evaluation = synthetic_blocks(
            seed=config["seed"] + 2,
            vocab_size=model.config.vocab_size,
            count=data_config["evaluation_blocks"],
            block_size=data_config["block_size"],
        )
        return calibration, evaluation

    calibration = text_blocks(
        tokenizer,
        data_config,
        data_config["calibration_split"],
        data_config["calibration_blocks"],
    )
    evaluation = text_blocks(
        tokenizer,
        data_config,
        data_config["evaluation_split"],
        data_config["evaluation_blocks"],
    )
    return calibration, evaluation

