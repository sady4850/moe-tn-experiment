"""Random-subspace null for the pairwise incremental-rank signal of Experiment 0.

For each saved expert pair, the second expert is rotated by a Haar-random orthogonal
matrix in the shared mode (input space for gate/up, output space for down). This keeps
both spectra exactly and destroys any subspace alignment. If the observed incremental
rank equals the rotated one, the two experts overlap no more than random matrices.

A second, cheaper analysis covers all 2016 pairs per layer: overlap of the rank-r
principal subspaces at the primary error, relative to the random-subspace expectation.

CPU only. Downloads just the expert tensors of the requested layers via HTTP ranges.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import time

from huggingface_hub import hf_hub_url
import numpy as np
import requests
import torch

from moe_tn.adapter import PROJECTIONS
from moe_tn.signals import _joint_matrix, effective_rank_from_singular_values

MODEL = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
NUM_EXPERTS = 64
ERRORS = (0.01, 0.05, 0.10, 0.20)
PRIMARY = 0.10


def _range(url: str, start: int, end: int) -> bytes:
    response = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
    response.raise_for_status()
    if len(response.content) != end - start:
        raise RuntimeError(f"Short read from {url}: {len(response.content)} != {end - start}")
    return response.content


def load_layer_experts(layer: int, cache_dir: Path) -> dict[tuple[int, str], torch.Tensor]:
    cache = cache_dir / f"layer{layer}_experts.pt"
    if cache.exists():
        return torch.load(cache)
    index_url = hf_hub_url(MODEL, "model.safetensors.index.json", revision=REVISION)
    weight_map = requests.get(index_url, timeout=60).json()["weight_map"]
    names = {
        (expert, projection): f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    headers: dict[str, tuple[str, int, dict]] = {}
    for shard in sorted({weight_map[name] for name in names.values()}):
        url = hf_hub_url(MODEL, shard, revision=REVISION)
        (length,) = struct.unpack("<Q", _range(url, 0, 8))
        headers[shard] = (url, 8 + length, json.loads(_range(url, 8, 8 + length)))

    def fetch(key: tuple[int, str]) -> tuple[tuple[int, str], torch.Tensor]:
        url, base, header = headers[weight_map[names[key]]]
        meta = header[names[key]]
        assert meta["dtype"] == "BF16", meta["dtype"]
        start, end = meta["data_offsets"]
        raw = bytearray(_range(url, base + start, base + end))
        return key, torch.frombuffer(raw, dtype=torch.bfloat16).reshape(meta["shape"]).clone()

    with ThreadPoolExecutor(max_workers=8) as pool:
        weights = dict(pool.map(fetch, names))
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(weights, cache)
    return weights


def haar_orthogonal(size: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(size, size, generator=generator, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()


def pair_score(sv_first, sv_second, sv_joint, error: float) -> float:
    """Normalized incremental rank for one projection, exactly as in Experiment 0."""
    reference = max(
        effective_rank_from_singular_values(sv_first, error),
        effective_rank_from_singular_values(sv_second, error),
    )
    joint = effective_rank_from_singular_values(sv_joint, error)
    return (joint - reference) / max(reference, 1)


def subspace_overlaps(weights, layer_svd, error: float) -> dict[str, np.ndarray]:
    """Pairwise ||P_i P_j||_F^2 / min(r_i, r_j) for rank-r principal subspaces in the shared mode."""
    out = {}
    for projection in PROJECTIONS:
        projectors, ranks = [], []
        for expert in range(NUM_EXPERTS):
            u, s, vh = layer_svd[(expert, projection)]
            rank = effective_rank_from_singular_values(s, error)
            basis = u[:, :rank].T if projection == "down_proj" else vh[:rank]
            projectors.append((basis.T @ basis).reshape(-1))
            ranks.append(rank)
        stacked = torch.stack(projectors)
        traces = (stacked @ stacked.T).double().numpy()
        ranks_arr = np.array(ranks, dtype=float)
        dimension = basis.shape[1]
        observed = traces / np.minimum.outer(ranks_arr, ranks_arr)
        expected = np.maximum.outer(ranks_arr, ranks_arr) / dimension
        out[projection] = observed / expected
        del stacked, projectors
    return out


def run_layer(layer: int, args, generator: torch.Generator) -> dict:
    started = time.time()
    weights = {key: value.float() for key, value in load_layer_experts(layer, args.cache_dir).items()}
    print(f"[layer {layer}] weights loaded in {time.time() - started:.0f}s", flush=True)

    layer_svd = {
        key: torch.linalg.svd(value, full_matrices=False) for key, value in weights.items()
    }
    singular = {key: value[1] for key, value in layer_svd.items()}

    source = Path(f"results/olmoe_layer{layer}")
    pairs = [tuple(pair) for pair in json.loads((source / "pairs.json").read_text())]
    stored = {
        (int(row["expert_first"]), int(row["expert_second"])): row
        for row in csv.DictReader((source / "results.csv").open())
    }

    dimension = weights[(0, "gate_proj")].shape[1]
    rotations = [haar_orthogonal(dimension, generator) for _ in range(args.rotation_pool)]

    rows = []
    rank_mismatch = 0
    for index, (first, second) in enumerate(pairs):
        observed = {error: [] for error in ERRORS}
        null = {error: np.zeros((args.draws, len(PROJECTIONS))) for error in ERRORS}
        for p, projection in enumerate(PROJECTIONS):
            left, right = weights[(first, projection)], weights[(second, projection)]
            sv_joint = torch.linalg.svdvals(_joint_matrix(left, right, projection))
            stored_joint = int(stored[(first, second)][f"rank_{projection}_joint_eps_{PRIMARY:.4f}"])
            rank_mismatch += abs(effective_rank_from_singular_values(sv_joint, PRIMARY) - stored_joint) > 1
            for error in ERRORS:
                observed[error].append(
                    pair_score(singular[(first, projection)], singular[(second, projection)], sv_joint, error)
                )
            choice = torch.randperm(args.rotation_pool, generator=generator)[: args.draws]
            for d, r in enumerate(choice.tolist()):
                rotated = rotations[r] @ right if projection == "down_proj" else right @ rotations[r]
                sv_null = torch.linalg.svdvals(_joint_matrix(left, rotated, projection))
                for error in ERRORS:
                    null[error][d, p] = pair_score(
                        singular[(first, projection)], singular[(second, projection)], sv_null, error
                    )
        row = {"expert_first": first, "expert_second": second,
               "delta_nll": float(stored[(first, second)]["delta_nll"])}
        for error in ERRORS:
            label = f"{error:.4f}"
            draws = null[error].mean(axis=1)
            row[f"observed_eps_{label}"] = float(np.mean(observed[error]))
            row[f"null_mean_eps_{label}"] = float(draws.mean())
            row[f"null_min_eps_{label}"] = float(draws.min())
            row[f"null_max_eps_{label}"] = float(draws.max())
            for p, projection in enumerate(PROJECTIONS):
                row[f"observed_{projection}_eps_{label}"] = observed[error][p]
                row[f"null_mean_{projection}_eps_{label}"] = float(null[error][:, p].mean())
        rows.append(row)
        if (index + 1) % 8 == 0:
            print(f"[layer {layer}] {index + 1}/{len(pairs)} pairs, {time.time() - started:.0f}s", flush=True)

    overlaps = subspace_overlaps(weights, layer_svd, PRIMARY)
    upper = np.triu_indices(NUM_EXPERTS, k=1)
    overlap_summary = {}
    for projection, matrix in overlaps.items():
        values = matrix[upper]
        best = int(np.argmax(values))
        overlap_summary[projection] = {
            "mean_ratio_to_random": float(values.mean()),
            "p99_ratio_to_random": float(np.quantile(values, 0.99)),
            "max_ratio_to_random": float(values.max()),
            "max_pair": [int(upper[0][best]), int(upper[1][best])],
        }
    mean_overlap = np.mean([overlaps[p] for p in PROJECTIONS], axis=0)[upper]
    top = np.argsort(mean_overlap)[::-1][:5]
    overlap_summary["mean_over_projections_top5"] = [
        {"pair": [int(upper[0][i]), int(upper[1][i])], "ratio_to_random": float(mean_overlap[i])}
        for i in top
    ]
    print(f"[layer {layer}] done in {time.time() - started:.0f}s", flush=True)
    return {"rows": rows, "overlap": overlap_summary, "rank_mismatch_gt1": int(rank_mismatch)}


def write_report(results: dict, args, output: Path) -> None:
    label = f"{PRIMARY:.4f}"
    lines = [
        "# Random-subspace null for pairwise incremental rank",
        "",
        f"Model `{MODEL}` @ `{REVISION[:12]}`, raw BF16 expert weights cast to FP32, CPU SVD.",
        f"Pairs: the saved Experiment 0 pairs. Null: second expert rotated by a Haar-random orthogonal "
        f"matrix in the shared mode, {args.draws} draws per pair and projection (pool of {args.rotation_pool}). "
        "Spectra of both experts are preserved exactly.",
        "",
        "Score: normalized incremental rank averaged over gate/up/down, as in Experiment 0. "
        "Lower means more shared subspace.",
        "",
        "## Observed vs null, primary error 10%",
        "",
        "| Layer | Observed mean | Null mean | Observed − null (mean) | min / max per-pair difference | Pairs below their null min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary = {}
    for layer, result in results.items():
        rows = result["rows"]
        observed = np.array([r[f"observed_eps_{label}"] for r in rows])
        null = np.array([r[f"null_mean_eps_{label}"] for r in rows])
        below = sum(r[f"observed_eps_{label}"] < r[f"null_min_eps_{label}"] for r in rows)
        diff = observed - null
        lines.append(
            f"| {layer} | {observed.mean():.4f} | {null.mean():.4f} | {diff.mean():+.4f} | "
            f"{diff.min():+.4f} / {diff.max():+.4f} | {below}/{len(rows)} |"
        )
        summary[layer] = {
            "observed_mean": float(observed.mean()), "null_mean": float(null.mean()),
            "difference_mean": float(diff.mean()), "difference_min": float(diff.min()),
            "difference_max": float(diff.max()), "pairs_below_null_min": int(below),
            "rank_mismatch_gt1_vs_stored": result["rank_mismatch_gt1"],
            "overlap": result["overlap"],
        }
    lines += ["", "## Error-budget sensitivity (mean observed − null over pairs)", "",
              "| Layer | " + " | ".join(f"{e:.0%}" for e in ERRORS) + " |",
              "|---|" + "---:|" * len(ERRORS)]
    for layer, result in results.items():
        cells = []
        for error in ERRORS:
            l = f"{error:.4f}"
            cells.append(f"{np.mean([r[f'observed_eps_{l}'] - r[f'null_mean_eps_{l}'] for r in result['rows']]):+.4f}")
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-projection difference at 10% (mean observed − null)", "",
              "| Layer | gate | up | down |", "|---|---:|---:|---:|"]
    for layer, result in results.items():
        cells = [f"{np.mean([r[f'observed_{p}_eps_{label}'] - r[f'null_mean_{p}_eps_{label}'] for r in result['rows']]):+.4f}"
                 for p in PROJECTIONS]
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    lines += ["", "## Principal-subspace overlap, all 2016 pairs per layer", "",
              "Ratio of ||P_i P_j||_F^2 / min(r_i, r_j) to its random-subspace expectation max(r_i, r_j)/n, "
              "rank-r subspaces at 10% error. 1.0 means no more overlap than random subspaces.", "",
              "| Layer | Projection | Mean | 99th pct | Max | Max pair |", "|---|---|---:|---:|---:|---|"]
    for layer, result in results.items():
        for projection in PROJECTIONS:
            o = result["overlap"][projection]
            lines.append(f"| {layer} | {projection} | {o['mean_ratio_to_random']:.3f} | "
                         f"{o['p99_ratio_to_random']:.3f} | {o['max_ratio_to_random']:.3f} | {tuple(o['max_pair'])} |")
    lines += ["", "Recomputed joint ranks are checked against the stored Experiment 0 values "
              "(GPU vs CPU SVD may differ by one): mismatches greater than one per layer: "
              + ", ".join(f"{k}={v['rank_mismatch_gt1_vs_stored']}" for k, v in summary.items()) + ".",
              "", "Raw weights only, no whitening or activation weighting. The null tests subspace "
              "alignment, not functional similarity or mergeability after neuron alignment."]
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    for layer, result in results.items():
        with (output / f"{layer}_pairs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 8, 13])
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--rotation-pool", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/olmoe_experts"))
    parser.add_argument("--output", type=Path, default=Path("results/null_rank"))
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    results = {f"layer{layer}": run_layer(layer, args, generator) for layer in args.layers}
    write_report(results, args, args.output)
    print((args.output / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
