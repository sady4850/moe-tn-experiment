"""Diagnostic (not pre-registered): is the GPT-OSS C4 degradation past token 128 an attention artefact?"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, "scripts")
from expert_removal_gptoss import load_gptoss  # noqa: E402

config = yaml.safe_load(Path("configs/expert_removal_gptoss20b.yaml").read_text())
out = Path(config["output_dir"])
model, tokenizer = load_gptoss(config)
print("attn_implementation:", model.config._attn_implementation, "sliding_window:", model.config.sliding_window, flush=True)
blocks = torch.load(out / "token_blocks.pt")
special = {name: tokenizer.convert_tokens_to_ids(name) for name in ("<|startoftext|>", "<|return|>", "<|endoftext|>")}
print("special ids:", special, flush=True)


@torch.inference_mode()
def profile(data, prefix=None):
    """NLL over the original 511 targets, by position, optionally with one token prepended."""
    nll = []
    for row in data:
        ids = row if prefix is None else torch.cat([torch.tensor([prefix]), row])
        logits = model(ids[None].to(model.device)).logits[0, :-1].float()
        values = F.cross_entropy(logits, ids[1:].to(model.device), reduction="none").cpu()
        nll.append(values[-511:])
    nll = torch.stack(nll)
    return {"all": round(nll.mean().item(), 3), "0-16": round(nll[:, :16].mean().item(), 3),
            "16-128": round(nll[:, 16:128].mean().item(), 3), "128-511": round(nll[:, 128:].mean().item(), 3)}


result = {}
for implementation in ("default", "eager"):
    if implementation == "eager":
        model.set_attn_implementation("eager")
        print("switched to:", model.config._attn_implementation, flush=True)
    for corpus in ("c4", "wikitext_test"):
        data = blocks[corpus][:32]
        for prefix_name in (None, "<|startoftext|>", "<|endoftext|>"):
            key = f"{implementation}/{corpus}/{prefix_name or 'no prefix'}"
            result[key] = profile(data, None if prefix_name is None else special[prefix_name])
            print(key, result[key], flush=True)
Path("c4_attention.json").write_text(json.dumps(result, indent=1))
