"""Random-subspace null for pairwise incremental rank on GPT-OSS-120B experts.

Same measurement as scripts/null_rank.py, applied to the model HyperNova-60B was
compressed from. There is no merge-damage ground truth here; the question is only
whether GPT-OSS experts share more subspace than random matrices with equal spectra.

Pairs: random expert pairs per layer, plus the most-overlapping pairs found by the
all-pairs principal-subspace screen. Expert biases are ignored, as for OLMoE.

CPU only. Downloads just the packed MXFP4 expert tensors of the requested layers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import struct
import time

from huggingface_hub import hf_hub_url
import numpy as np
import requests
import torch

from moe_tn.signals import _joint_matrix, effective_rank_from_singular_values
from null_rank import ERRORS, PRIMARY, _range, haar_orthogonal, pair_score

MODEL = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
PACKED = ("gate_up_proj_blocks", "gate_up_proj_scales", "down_proj_blocks", "down_proj_scales")


def _range_retry(url: str, start: int, end: int, attempts: int = 12) -> bytes:
    for attempt in range(attempts):
        try:
            return _range(url, start, end)
        except (requests.RequestException, RuntimeError) as error:
            if attempt == attempts - 1:
                raise
            print(f"retry {attempt + 1} for bytes {start}-{end}: {error.__class__.__name__}", flush=True)
            time.sleep(min(10 * (attempt + 1), 60))


def load_packed(layer: int, cache_dir: Path) -> dict[str, torch.Tensor]:
    """Fetch each packed tensor once and cache it separately, so a dropped connection loses one tensor."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    weight_map = None
    packed = {}
    for suffix in PACKED:
        cache = cache_dir / f"layer{layer}_{suffix}.pt"
        if cache.exists():
            packed[suffix] = torch.load(cache)
            continue
        if weight_map is None:
            index_url = hf_hub_url(MODEL, "model.safetensors.index.json", revision=REVISION)
            weight_map = requests.get(index_url, timeout=60).json()["weight_map"]
        name = f"model.layers.{layer}.mlp.experts.{suffix}"
        url = hf_hub_url(MODEL, weight_map[name], revision=REVISION)
        (length,) = struct.unpack("<Q", _range_retry(url, 0, 8))
        meta = json.loads(_range_retry(url, 8, 8 + length))[name]
        assert meta["dtype"] == "U8", meta["dtype"]
        start, end = meta["data_offsets"]
        chunk = 64 * 1024 * 1024
        raw = bytearray()
        for offset in range(start, end, chunk):
            raw += _range_retry(url, 8 + length + offset, 8 + length + min(offset + chunk, end))
        packed[suffix] = torch.frombuffer(raw, dtype=torch.uint8).reshape(meta["shape"]).clone()
        torch.save(packed[suffix], cache.with_suffix(f".{os.getpid()}.tmp"))
        cache.with_suffix(f".{os.getpid()}.tmp").rename(cache)
        print(f"[layer {layer}] cached {suffix}", flush=True)
    return packed


def dequantize(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """MXFP4 [E, out, G, 16] bytes + [E, out, G] E8M0 scales -> BF16 [E, out, in].

    FP4 grid values times powers of two are exact in BF16.
    """
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32)
    *prefix, groups, width = blocks.shape
    out = torch.empty(*prefix, groups, width * 2, dtype=torch.bfloat16)
    for expert in range(prefix[0]):
        values = torch.empty(*prefix[1:], groups, width * 2, dtype=torch.float32)
        values[..., 0::2] = lut[(blocks[expert] & 0x0F).long()]
        values[..., 1::2] = lut[(blocks[expert] >> 4).long()]
        exponent = scales[expert].to(torch.int32) - 127
        out[expert] = torch.ldexp(values, exponent.unsqueeze(-1)).to(torch.bfloat16)
    return out.reshape(*prefix, groups * width * 2)


def expert_weights(packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    gate_up = dequantize(packed["gate_up_proj_blocks"], packed["gate_up_proj_scales"])
    return {
        "gate_proj": gate_up[:, 0::2].contiguous(),  # GptOssExperts._apply_gate: gate = [..., ::2]
        "up_proj": gate_up[:, 1::2].contiguous(),
        "down_proj": dequantize(packed["down_proj_blocks"], packed["down_proj_scales"]),
    }


def run_layer(layer: int, args, generator: torch.Generator) -> dict:
    started = time.time()
    weights = expert_weights(load_packed(layer, args.cache_dir))
    num_experts, _, dimension = weights["gate_proj"].shape
    print(f"[layer {layer}] {num_experts} experts loaded in {time.time() - started:.0f}s", flush=True)

    singular: dict[tuple[int, str], torch.Tensor] = {}
    overlap: dict[str, np.ndarray] = {}
    for projection in PROJECTIONS:
        projectors, ranks = [], []
        for expert in range(num_experts):
            u, s, vh = torch.linalg.svd(weights[projection][expert].float(), full_matrices=False)
            singular[(expert, projection)] = s
            rank = effective_rank_from_singular_values(s, PRIMARY)
            basis = u[:, :rank].T if projection == "down_proj" else vh[:rank]
            projectors.append((basis.T @ basis).reshape(-1))
            ranks.append(rank)
        stacked = torch.stack(projectors)
        del projectors
        traces = (stacked @ stacked.T).double().numpy()
        del stacked
        r = np.array(ranks, dtype=float)
        overlap[projection] = (traces / np.minimum.outer(r, r)) / (np.maximum.outer(r, r) / dimension)
        print(f"[layer {layer}] {projection} SVDs done, {time.time() - started:.0f}s", flush=True)

    upper = np.triu_indices(num_experts, k=1)
    mean_overlap = np.mean([overlap[p] for p in PROJECTIONS], axis=0)
    ordered = np.argsort(mean_overlap[upper])[::-1]
    top_pairs = [(int(upper[0][i]), int(upper[1][i])) for i in ordered[: args.top_pairs]]
    rng = random.Random(args.seed + layer)
    random_pairs = sorted(rng.sample([(a, b) for a in range(num_experts) for b in range(a + 1, num_experts)],
                                     args.random_pairs))

    rotations = [haar_orthogonal(dimension, generator) for _ in range(args.rotation_pool)]
    rows = []
    for index, (first, second) in enumerate(random_pairs + [p for p in top_pairs if p not in random_pairs]):
        observed = {e: [] for e in ERRORS}
        null = {e: np.zeros((args.draws, len(PROJECTIONS))) for e in ERRORS}
        for p, projection in enumerate(PROJECTIONS):
            left = weights[projection][first].float()
            right = weights[projection][second].float()
            sv_first, sv_second = singular[(first, projection)], singular[(second, projection)]
            sv_joint = torch.linalg.svdvals(_joint_matrix(left, right, projection))
            for e in ERRORS:
                observed[e].append(pair_score(sv_first, sv_second, sv_joint, e))
            choice = torch.randperm(args.rotation_pool, generator=generator)[: args.draws]
            for d, k in enumerate(choice.tolist()):
                rotated = rotations[k] @ right if projection == "down_proj" else right @ rotations[k]
                sv_null = torch.linalg.svdvals(_joint_matrix(left, rotated, projection))
                for e in ERRORS:
                    null[e][d, p] = pair_score(sv_first, sv_second, sv_null, e)
        row = {"expert_first": first, "expert_second": second,
               "group": "random" if (first, second) in random_pairs else "top_overlap",
               "overlap_ratio": float(mean_overlap[first, second])}
        for e in ERRORS:
            label = f"{e:.4f}"
            draws = null[e].mean(axis=1)
            row[f"observed_eps_{label}"] = float(np.mean(observed[e]))
            row[f"null_mean_eps_{label}"] = float(draws.mean())
            row[f"null_min_eps_{label}"] = float(draws.min())
            for p, projection in enumerate(PROJECTIONS):
                row[f"observed_{projection}_eps_{label}"] = observed[e][p]
                row[f"null_mean_{projection}_eps_{label}"] = float(null[e][:, p].mean())
        rows.append(row)
        if (index + 1) % 8 == 0:
            print(f"[layer {layer}] {index + 1} pairs, {time.time() - started:.0f}s", flush=True)

    rank_fraction = {
        projection: float(np.mean([
            effective_rank_from_singular_values(singular[(x, projection)], PRIMARY) / dimension
            for x in range(num_experts)
        ]))
        for projection in PROJECTIONS
    }
    overlap_summary = {
        projection: {
            "mean_ratio_to_random": float(overlap[projection][upper].mean()),
            "p99_ratio_to_random": float(np.quantile(overlap[projection][upper], 0.99)),
            "max_ratio_to_random": float(overlap[projection][upper].max()),
        }
        for projection in PROJECTIONS
    }
    overlap_summary["mean_over_projections_top"] = [
        {"pair": list(p), "ratio_to_random": float(mean_overlap[p])} for p in top_pairs
    ]
    print(f"[layer {layer}] done in {time.time() - started:.0f}s", flush=True)
    return {"rows": rows, "overlap": overlap_summary, "rank_fraction": rank_fraction,
            "num_experts": num_experts, "dimension": dimension}


def write_report(results: dict, args, output: Path) -> None:
    label = f"{PRIMARY:.4f}"
    lines = [
        "# Random-subspace null for pairwise incremental rank: GPT-OSS-120B",
        "",
        f"Model `{MODEL}` @ `{REVISION[:12]}`. MXFP4 expert weights dequantized exactly to BF16, "
        "then FP32 CPU SVD. Biases ignored. Same score and null as `results/null_rank` (OLMoE): "
        f"Haar rotation of the second expert in the shared hidden mode, {args.draws} draws per pair and projection.",
        "",
        "Score: normalized incremental rank, mean over gate/up/down; 0 means identical subspaces. "
        "Experts here are square (2880 x 2880), so a stacked pair can never exceed rank 2880 and "
        "the random-null score is capped well below OLMoE's. Compare observed with null, not across models.",
        "",
        "## Observed vs null, primary error 10%",
        "",
        "| Layer | Pairs | Mean effective rank / 2880 (gate, up, down) | Observed | Null | Observed − null | min per-pair diff | Below null min |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    summary = {}
    for layer, result in results.items():
        fractions = ", ".join(f"{result['rank_fraction'][p]:.2f}" for p in PROJECTIONS)
        summary[layer] = {"rank_fraction": result["rank_fraction"], "overlap": result["overlap"]}
        for group in ("random", "top_overlap"):
            rows = [r for r in result["rows"] if r["group"] == group]
            if not rows:
                continue
            observed = np.array([r[f"observed_eps_{label}"] for r in rows])
            null = np.array([r[f"null_mean_eps_{label}"] for r in rows])
            below = sum(r[f"observed_eps_{label}"] < r[f"null_min_eps_{label}"] for r in rows)
            diff = observed - null
            lines.append(f"| {layer} | {group} ({len(rows)}) | {fractions} | {observed.mean():.4f} | "
                         f"{null.mean():.4f} | {diff.mean():+.4f} | {diff.min():+.4f} | {below}/{len(rows)} |")
            summary[layer][group] = {
                "observed_mean": float(observed.mean()), "null_mean": float(null.mean()),
                "difference_mean": float(diff.mean()), "difference_min": float(diff.min()),
                "relative_to_null": float(diff.mean() / null.mean()), "pairs_below_null_min": int(below),
            }
    lines += ["", "Relative shift (observed − null) / null for random pairs, i.e. the fraction of the way from random "
              "toward identical subspaces: " + ", ".join(
                  f"{k} {v['random']['relative_to_null']:+.1%}" for k, v in summary.items()) + ".",
              "", "## Error-budget sensitivity, random pairs (mean observed − null)", "",
              "| Layer | " + " | ".join(f"{e:.0%}" for e in ERRORS) + " |", "|---|" + "---:|" * len(ERRORS)]
    for layer, result in results.items():
        rows = [r for r in result["rows"] if r["group"] == "random"]
        cells = [f"{np.mean([r[f'observed_eps_{e:.4f}'] - r[f'null_mean_eps_{e:.4f}'] for r in rows]):+.4f}"
                 for e in ERRORS]
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    lines += ["", "## Principal-subspace overlap, all pairs", "",
              "Ratio of ||P_i P_j||_F^2 / min(r_i, r_j) to the random expectation max(r_i, r_j)/n at 10% error. "
              "With near-full-rank square experts this ratio is bounded by n / max(r_i, r_j).", "",
              "| Layer | Projection | Mean | 99th pct | Max |", "|---|---|---:|---:|---:|"]
    for layer, result in results.items():
        for projection in PROJECTIONS:
            o = result["overlap"][projection]
            lines.append(f"| {layer} | {projection} | {o['mean_ratio_to_random']:.3f} | "
                         f"{o['p99_ratio_to_random']:.3f} | {o['max_ratio_to_random']:.3f} |")
        top = ", ".join(f"{tuple(t['pair'])} {t['ratio_to_random']:.3f}"
                        for t in result["overlap"]["mean_over_projections_top"])
        lines.append(f"| {layer} | top pairs (mean of projections) | {top} | | |")
    lines += ["", "No merge-damage ground truth is measured for GPT-OSS. Raw weights only, no whitening or "
              "activation weighting, no neuron alignment."]
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    for layer, result in results.items():
        (output / f"{layer}_pairs.json").write_text(json.dumps(result["rows"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 18, 29])
    parser.add_argument("--random-pairs", type=int, default=32)
    parser.add_argument("--top-pairs", type=int, default=5)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--rotation-pool", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/gptoss_experts"))
    parser.add_argument("--output", type=Path, default=Path("results/null_rank_gptoss"))
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    results = {f"layer{layer}": run_layer(layer, args, generator) for layer in args.layers}
    write_report(results, args, args.output)
    print((args.output / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
