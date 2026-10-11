"""Stage 2 of PRUNING_PLAN.md on GPT-OSS-20B (amendment 2026-10-11): expert removal and merging, no healing.

Same phases, variants, scores and verdicts as scripts/expert_removal.py; only the model plumbing differs:
MXFP4 experts dequantized to BF16, transposed and interleaved gate/up weights, expert biases, the
clamped gated unit, a router that applies softmax over its top-k logits, and C4 windows re-selected with
the GPT-OSS tokenizer.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import random
import time
from types import MethodType
import urllib.request
from urllib.parse import urlsplit

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import yaml

from moe_tn.compression_run import save_json, save_tensor
from moe_tn.data import synthetic_blocks, text_blocks
from moe_tn.whitening import regularized_cholesky
from expert_removal import (
    FAMILIES, VARIANTS, cluster, is_merge, lowest, pairwise_incremental_rank, random_groups,
    write_report,
)
from layer_removal import C4_URL, digest, downstream, evaluate_set

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class GptOssAdapter:
    """Expert weights of one GPT-OSS layer as (out, in) views that write through to the parameters."""

    def __init__(self, model, layer: int):
        self.block = model.model.layers[layer].mlp
        self.experts = self.block.experts
        self.router = self.block.router
        assert type(self.router).__name__ == "GptOssTopKRouter", type(self.router)

    @property
    def num_experts(self) -> int:
        return int(self.experts.gate_up_proj.shape[0])

    @property
    def device(self):
        return self.experts.gate_up_proj.device

    @property
    def dtype(self):
        return self.experts.gate_up_proj.dtype

    @property
    def top_k(self) -> int:
        return int(self.router.top_k)

    def weight(self, expert: int, projection: str) -> torch.Tensor:
        if projection == "gate_proj":
            return self.experts.gate_up_proj[expert][:, 0::2].T
        if projection == "up_proj":
            return self.experts.gate_up_proj[expert][:, 1::2].T
        return self.experts.down_proj[expert].T

    def bias(self, expert: int, projection: str) -> torch.Tensor:
        if projection == "gate_proj":
            return self.experts.gate_up_proj_bias[expert][0::2]
        if projection == "up_proj":
            return self.experts.gate_up_proj_bias[expert][1::2]
        return self.experts.down_proj_bias[expert]

    def intermediate(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        gate_up = x @ self.experts.gate_up_proj[expert] + self.experts.gate_up_proj_bias[expert]
        return self.experts._apply_gate(gate_up)

    def output(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        return self.intermediate(x, expert) @ self.experts.down_proj[expert] + self.experts.down_proj_bias[expert]


def load_gptoss(config: dict):
    model_config = config["model"]
    if model_config["id"] == "__tiny_random_gptoss__":
        from transformers import GptOssConfig, GptOssForCausalLM
        torch.manual_seed(config["seed"])
        tiny = GptOssConfig(vocab_size=128, hidden_size=32, intermediate_size=24, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=6,
                            num_experts_per_tok=2, sliding_window=16, max_position_embeddings=128,
                            layer_types=["sliding_attention", "full_attention"], pad_token_id=0, eos_token_id=1)
        model = GptOssForCausalLM(tiny).to(config.get("device", "cpu")).eval()
        with torch.no_grad():
            for layer in model.model.layers:
                for name in ("gate_up_proj_bias", "down_proj_bias"):
                    getattr(layer.mlp.experts, name).normal_(0, 0.02)
        return model, None
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    tokenizer = AutoTokenizer.from_pretrained(model_config["id"], revision=model_config["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"], revision=model_config["revision"], dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True), device_map={"": 0})
    model.eval()
    assert model.model.layers[0].mlp.experts.gate_up_proj.dtype == torch.bfloat16
    return model, tokenizer


# ---------------------------------------------------------------------------------------------
# Data


def c4_select(tokenizer, size: int, count: int) -> tuple[torch.Tensor, list[dict]]:
    """The OLMoE C4 rule re-run with this tokenizer: first `count` eligible distinct hostnames."""
    for attempt in range(6):
        try:
            rng = random.Random(17)
            blocks, documents, hosts, texts = [], [], set(), set()
            with urllib.request.urlopen(C4_URL, timeout=120) as response, \
                    gzip.GzipFile(fileobj=response) as stream:
                for row_index, line in enumerate(stream):
                    row = json.loads(line)
                    host = (urlsplit(row["url"]).hostname or "").lower().removeprefix("www.")
                    text_hash = hashlib.sha256(row["text"].encode()).hexdigest()
                    if not host or host in hosts or text_hash in texts:
                        continue
                    ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
                    if len(ids) < size:
                        continue
                    start = rng.randrange(len(ids) - size + 1)
                    blocks.append(ids[start:start + size])
                    documents.append({"row": row_index, "url": row["url"], "hostname": host,
                                      "text_sha256": text_hash, "document_tokens": len(ids), "window_start": start})
                    hosts.add(host)
                    texts.add(text_hash)
                    if len(blocks) == count:
                        return torch.tensor(blocks, dtype=torch.long), documents
        except OSError as error:
            print(f"C4 download retry {attempt + 1}: {error}", flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("C4 selection failed")


def load_corpora(config, model, tokenizer, output: Path) -> dict[str, torch.Tensor]:
    path = output / "token_blocks.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    d, seed = config["data"], config["seed"]
    if tokenizer is None:
        def synthetic(offset, count):
            return synthetic_blocks(seed=seed + offset, vocab_size=model.config.vocab_size,
                                    count=count, block_size=d["block_size"])
        corpora = {"calibration": synthetic(1, d["calibration_blocks"]), "dev": synthetic(2, d["dev_blocks"]),
                   "wikitext_test": synthetic(3, d["test_blocks"])}
    else:
        corpora = {
            "calibration": text_blocks(tokenizer, d, d["calibration_split"], d["calibration_blocks"]),
            "dev": text_blocks(tokenizer, d, d["dev_split"], d["dev_blocks"]),
            "wikitext_test": text_blocks(tokenizer, d, d["test_split"],
                                         d["test_offset_blocks"] + d["test_blocks"])[d["test_offset_blocks"]:],
        }
        if config.get("c4_count"):
            corpora["c4"], documents = c4_select(tokenizer, d["block_size"], config["c4_count"])
            save_json(output / "c4_documents.json", documents)
    save_tensor(path, corpora)
    return corpora


# ---------------------------------------------------------------------------------------------
# Plans applied to the model


def _masked_router_forward(self, hidden_states):
    """GptOssTopKRouter.forward with removed experts' logits set to -inf before the top-k."""
    router_logits = F.linear(hidden_states, self.weight, self.bias).masked_fill(self._removed_mask, float("-inf"))
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    router_scores = F.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    return router_logits, router_scores, router_indices


def merged_parameters(adapter, members: list[dict]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Frequency-weighted average of neuron-aligned members, weights and biases."""
    merged = {}
    for member in members:
        perm = torch.as_tensor(member["perm"], device=adapter.device)
        e = member["expert"]
        aligned = {
            "gate_proj": (adapter.weight(e, "gate_proj").detach().float()[perm],
                          adapter.bias(e, "gate_proj").detach().float()[perm]),
            "up_proj": (adapter.weight(e, "up_proj").detach().float()[perm],
                        adapter.bias(e, "up_proj").detach().float()[perm]),
            "down_proj": (adapter.weight(e, "down_proj").detach().float()[:, perm],
                          adapter.bias(e, "down_proj").detach().float()),
        }
        for projection, (weight, bias) in aligned.items():
            w0, b0 = merged.get(projection, (0, 0))
            merged[projection] = (w0 + member["weight"] * weight, b0 + member["weight"] * bias)
    return {p: (w.to(adapter.dtype), b.to(adapter.dtype)) for p, (w, b) in merged.items()}


@contextmanager
def compressed(model, plan: dict | None):
    if plan is None:
        yield
        return
    saved, routers = [], []
    try:
        for layer, layer_plan in enumerate(plan["layers"]):
            adapter = GptOssAdapter(model, layer)
            if plan["operator"] == "prune":
                mask = torch.zeros(adapter.num_experts, dtype=torch.bool, device=adapter.device)
                mask[layer_plan["removed"]] = True
                adapter.router._removed_mask = mask
                adapter.router.forward = MethodType(_masked_router_forward, adapter.router)
                routers.append(adapter.router)
                continue
            for group in layer_plan["groups"]:
                if len(group) == 1:
                    continue
                parameters = merged_parameters(adapter, group)
                with torch.no_grad():
                    for member in group:
                        for projection, (weight, bias) in parameters.items():
                            for target, value in ((adapter.weight(member["expert"], projection), weight),
                                                  (adapter.bias(member["expert"], projection), bias)):
                                saved.append((target, target.detach().cpu().clone()))
                                target.copy_(value)
        yield
    finally:
        with torch.no_grad():
            for target, value in reversed(saved):
                target.copy_(value.to(target.device))
        for router in routers:
            del router.forward, router._removed_mask


# ---------------------------------------------------------------------------------------------
# Statistics and plans


@torch.inference_mode()
def capture_moe_inputs(model, blocks) -> list[torch.Tensor]:
    layers = model.model.layers
    captured: list[list[torch.Tensor]] = [[] for _ in layers]
    handles = [layer.mlp.register_forward_pre_hook(
        lambda module, args, index=i: captured[index].append(args[0].reshape(-1, args[0].shape[-1]).cpu()))
        for i, layer in enumerate(layers)]
    device = next(model.parameters()).device
    try:
        for index in range(len(blocks)):
            model(input_ids=blocks[index:index + 1].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [torch.cat(chunks) for chunks in captured]


@torch.inference_mode()
def layer_statistics(adapter, x: torch.Tensor, floor: float, chunk: int = 4096) -> dict:
    num_experts = adapter.num_experts
    _, top_values, top_indices = adapter.router(x)
    frequency = torch.bincount(top_indices.reshape(-1), minlength=num_experts).cpu()
    reap_sum = torch.zeros(num_experts, dtype=torch.float64)
    output_sum = torch.zeros(num_experts, x.shape[1], dtype=torch.float64)
    hidden_moment = torch.zeros(x.shape[1], x.shape[1], device=adapter.device)
    size = adapter.experts.down_proj.shape[1]
    down_moment = torch.zeros(size, size, device=adapter.device)
    for start in range(0, len(x), chunk):
        xf = x[start:start + chunk].float()
        hidden_moment += xf.T @ xf
    for expert in range(num_experts):
        routed_mask = top_indices == expert
        for start in range(0, len(x), chunk):
            xs = x[start:start + chunk]
            h = adapter.intermediate(xs, expert)
            y = h @ adapter.experts.down_proj[expert] + adapter.experts.down_proj_bias[expert]
            output_sum[expert] += y.float().sum(0).double().cpu()
            mask = routed_mask[start:start + chunk]
            routed = mask.any(-1)
            if routed.any():
                weight = (top_values[start:start + chunk].float() * mask).sum(-1)[routed]
                reap_sum[expert] += (weight * y[routed].float().norm(dim=-1)).sum().double().cpu()
                hr = h[routed].float()
                down_moment += hr.T @ hr
    rows = len(x)
    l_hidden, hidden_stats = regularized_cholesky(hidden_moment / rows, floor)
    l_down, down_stats = regularized_cholesky(down_moment / (rows * adapter.top_k), floor)
    return {"frequency": frequency, "reap": (reap_sum / frequency.clamp_min(1).double()).float(),
            "mean_output": (output_sum / rows).float(), "l_hidden": l_hidden.cpu(), "l_down": l_down.cpu(),
            "whitening": {"hidden": hidden_stats, "down": down_stats}}


@torch.inference_mode()
def align_group(adapter, x: torch.Tensor, group: list[int], frequency: torch.Tensor) -> list[dict]:
    anchor = min(group, key=lambda e: (-int(frequency[e]), e))
    total = float(sum(int(frequency[e]) for e in group))

    def standardized(expert):
        h = adapter.intermediate(x, expert).float()
        h = h - h.mean(0)
        return h / h.norm(dim=0).clamp_min(1e-12)

    reference = standardized(anchor)
    members = []
    for expert in group:
        weight = int(frequency[expert]) / total if total > 0 else 1.0 / len(group)
        if expert == anchor:
            perm = list(range(reference.shape[1]))
        else:
            rows, columns = linear_sum_assignment(-(reference.T @ standardized(expert)).cpu().numpy())
            perm = columns[np.argsort(rows)].tolist()
        members.append({"expert": expert, "perm": perm, "weight": weight})
    return members


def build_plans(model, inputs, stats, pairwise, keep: int, seed: int) -> dict[str, dict]:
    num_experts = len(stats[0]["frequency"])
    remove = num_experts - keep
    plans = {}
    for variant in VARIANTS:
        # One generator per random variant, Random(17 + seed), drawn layer by layer in order.
        rng = random.Random(seed + int(variant[-1])) if variant.startswith("random") else None
        layers = []
        for layer, s in enumerate(stats):
            if variant in ("tn_prune", "tn_w_prune"):
                matrix = pairwise["raw" if variant == "tn_prune" else "whitened"][layer]
                layers.append({"removed": lowest(matrix.sum(1) / (num_experts - 1), remove)})
            elif variant == "frequency":
                layers.append({"removed": lowest(s["frequency"].tolist(), remove)})
            elif variant == "reap":
                layers.append({"removed": lowest(s["reap"].tolist(), remove)})
            elif variant.startswith("random_prune"):
                layers.append({"removed": sorted(rng.sample(range(num_experts), remove))})
            else:
                if variant in ("tn_merge", "tn_w_merge"):
                    groups = cluster(pairwise["raw" if variant == "tn_merge" else "whitened"][layer], keep)
                elif variant == "output_similarity":
                    mean = F.normalize(s["mean_output"].double(), dim=1)
                    groups = cluster((1 - mean @ mean.T).clamp_min(0).fill_diagonal_(0).numpy(), keep)
                else:
                    groups = random_groups(rng, num_experts, keep)
                adapter = GptOssAdapter(model, layer)
                x = inputs[layer].to(adapter.device)
                layers.append({"groups": [align_group(adapter, x, g, s["frequency"]) for g in groups]})
        plans[variant] = {"operator": "merge" if is_merge(variant) else "prune", "layers": layers}
        print(f"[plans] {variant} built", flush=True)
    return plans


# ---------------------------------------------------------------------------------------------


def run(config: dict) -> None:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__)
    plan = {"config": config, "script_sha256": digest(here),
            "expert_removal_sha256": digest(here.with_name("expert_removal.py")),
            "layer_removal_sha256": digest(here.with_name("layer_removal.py"))}
    plan_path = output / "PLAN.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Config or scripts changed"
    else:
        save_json(plan_path, plan)

    model, tokenizer = load_gptoss(config)
    num_layers = len(model.model.layers)
    corpora = load_corpora(config, model, tokenizer, output)

    stats_path = output / "stats.pt"
    inputs = None
    if stats_path.exists():
        stats = torch.load(stats_path, weights_only=False)
    else:
        inputs = capture_moe_inputs(model, corpora["calibration"])
        stats = []
        for layer in range(num_layers):
            adapter = GptOssAdapter(model, layer)
            stats.append(layer_statistics(adapter, inputs[layer].to(adapter.device),
                                          config.get("whitening_relative_floor", 1e-3)))
            print(f"[stats] layer {layer}", flush=True)
        save_tensor(stats_path, stats)

    pairwise = {"raw": [], "whitened": []}
    started = time.time()
    for layer in range(num_layers):
        adapter = GptOssAdapter(model, layer)
        for kind in ("raw", "whitened"):
            path = output / "pairwise" / f"{kind}_layer{layer}.json"
            if not path.exists():
                matrices = {}
                for projection in PROJECTIONS:
                    factor = stats[layer]["l_down" if projection == "down_proj" else "l_hidden"].to(adapter.device)
                    matrices[projection] = [
                        adapter.weight(e, projection).detach().float() @ factor if kind == "whitened"
                        else adapter.weight(e, projection).detach().float() for e in range(adapter.num_experts)]
                path.parent.mkdir(parents=True, exist_ok=True)
                save_json(path, pairwise_incremental_rank(matrices).tolist())
                print(f"[pairwise] layer {layer} {kind} done, {(time.time() - started) / 60:.1f} min", flush=True)
            pairwise[kind].append(np.array(json.loads(path.read_text())))

    plans_dir = output / "plans"
    index_path = plans_dir / "index.json"
    if not index_path.exists():
        if inputs is None:
            inputs = capture_moe_inputs(model, corpora["calibration"])
        plans = build_plans(model, inputs, stats, pairwise, config["keep_experts"], config["seed"])
        plans_dir.mkdir(parents=True, exist_ok=True)
        for variant, value in plans.items():
            save_json(plans_dir / f"{variant}.json", value)
        save_json(index_path, {v: digest(plans_dir / f"{v}.json") for v in VARIANTS})
    index = json.loads(index_path.read_text())
    plans = {}
    for variant in VARIANTS:
        assert digest(plans_dir / f"{variant}.json") == index[variant], f"Plan {variant} changed"
        plans[variant] = json.loads((plans_dir / f"{variant}.json").read_text())
    del inputs
    print("[plans] frozen", flush=True)

    def evaluate(variant, corpus):
        with compressed(model, plans.get(variant)):
            return evaluate_set(model, corpora[corpus], [], output / corpus / f"{variant}.json")

    dev = {"base": evaluate("base", "dev")}
    for variant in VARIANTS:
        dev[variant] = evaluate(variant, "dev")
        print(f"[dev] {variant}: ΔNLL {dev[variant].mean() - dev['base'].mean():+.4f}", flush=True)
    best = {family: min(members, key=lambda v: dev[v].mean())
            for family, members in FAMILIES.items() if family != "random"}
    save_json(output / "family_best.json", best)

    chosen = ["base", *dict.fromkeys(best.values())]
    test_corpora = [c for c in ("wikitext_test", "c4") if c in corpora]
    test = {c: {v: evaluate(v, c) for v in chosen} for c in test_corpora}
    down = None
    if config.get("downstream"):
        d = config["downstream"]
        down = {}
        for variant in chosen:
            with compressed(model, plans.get(variant)):
                down[variant] = downstream(model, tokenizer, [], d["tasks"], d["batch_size"],
                                           output / "downstream" / f"{variant}.json", d.get("limit"))
    write_report(output, dev, best, test, down)
    report = output / "REPORT.md"
    num_experts = len(stats[0]["frequency"])
    title = (f"# Stage 2 on {config['model']['id']}: one-shot expert removal and merging, "
             f"{num_experts} -> {config['keep_experts']} experts in every layer (no healing)")
    lines = report.read_text().splitlines()
    report.write_text("\n".join([title, *lines[1:]]).replace(
        "(amendment 2026-10-10, stage 2 definitions)",
        "(amendments 2026-10-10 stage 2 definitions and 2026-10-11 GPT-OSS-20B replication)") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    run(yaml.safe_load(parser.parse_args().config.read_text()))


if __name__ == "__main__":
    main()
