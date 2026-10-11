"""Stage 0 of PRUNING_PLAN.md: TN layer density for every OLMoE layer (CPU, no quality data).

For each expert projection, the raw tensor [expert, out, in] is unfolded along each mode; the
effective rank at a fixed relative error is divided by the mode size. The layer density is the
mean of the 9 normalized ranks (3 projections x 3 modes).

Layers not already cached by scripts/null_rank.py are downloaded, used and deleted again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import requests
import torch

from moe_tn.adapter import PROJECTIONS
from moe_tn.signals import effective_rank_from_singular_values
from null_rank import NUM_EXPERTS, load_layer_experts

ERRORS = (0.05, 0.10, 0.20)
PRIMARY = 0.10
MODES = ("expert", "out", "in")
KEEP = {3, 8, 13}


def mode_singular_values(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Singular values of the three mode unfoldings via their Gram matrices (FP64 eigensolver)."""
    experts = tensor.double()
    grams = {
        "expert": (lambda flat: flat @ flat.T)(experts.reshape(experts.shape[0], -1)),
        "out": torch.einsum("eoi,epi->op", experts, experts),
        "in": torch.einsum("eoi,eoj->ij", experts, experts),
    }
    return {mode: torch.linalg.eigvalsh(gram).clamp_min(0).flip(0).sqrt() for mode, gram in grams.items()}


def layer_density(layer: int, cache_dir: Path) -> dict:
    cached = (cache_dir / f"layer{layer}_experts.pt").exists()
    weights = load_layer_experts(layer, cache_dir)
    result: dict = {"normalized_rank": {}}
    for projection in PROJECTIONS:
        tensor = torch.stack([weights[(e, projection)].float() for e in range(NUM_EXPERTS)])
        for mode, values in mode_singular_values(tensor).items():
            size = {"expert": tensor.shape[0], "out": tensor.shape[1], "in": tensor.shape[2]}[mode]
            for error in ERRORS:
                rank = effective_rank_from_singular_values(values, error)
                result["normalized_rank"][f"{projection}/{mode}/{error:.2f}"] = rank / size
    for error in ERRORS:
        result[f"density_{error:.2f}"] = float(np.mean([
            result["normalized_rank"][f"{p}/{m}/{error:.2f}"] for p in PROJECTIONS for m in MODES
        ]))
    if not cached and layer not in KEEP:
        (cache_dir / f"layer{layer}_experts.pt").unlink()
    return result


def write_report(results: dict[int, dict], output: Path) -> None:
    label = f"{PRIMARY:.2f}"
    densities = {layer: r[f"density_{label}"] for layer, r in results.items()}
    order = sorted(densities, key=densities.get)
    lines = [
        "# Stage 0: TN layer density (OLMoE-1B-7B)",
        "",
        "Raw expert weights, FP32 -> FP64 Gram eigensolver. Normalized effective mode rank of the "
        "`[expert, out, in]` tensor per projection; density = mean of the 9 values. See PRUNING_PLAN.md.",
        "",
        "| Layer | Density 5% | Density 10% (primary) | Density 20% | " +
        " | ".join(f"{p.split('_')[0]} {m}" for p in PROJECTIONS for m in MODES) + " (10%) |",
        "|---:|---:|---:|---:|" + "---:|" * 9,
    ]
    for layer, r in results.items():
        cells = [f"{r['normalized_rank'][f'{p}/{m}/{label}']:.3f}" for p in PROJECTIONS for m in MODES]
        lines.append(f"| {layer} | {r['density_0.05']:.4f} | {r['density_0.10']:.4f} | "
                     f"{r['density_0.20']:.4f} | " + " | ".join(cells) + " |")
    values = np.array(list(densities.values()))
    lines += [
        "",
        f"Primary density range: {values.min():.4f} to {values.max():.4f} "
        f"(spread {values.max() - values.min():.4f}, sd {values.std(ddof=1):.4f}).",
        "",
        "Stage 1 TN removal order (lowest density first): " + ", ".join(str(l) for l in order) + ".",
        f"Frozen TN choice for k = 1: {order[:1]}, k = 2: {sorted(order[:2])}, k = 4: {sorted(order[:4])}.",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "density.json").write_text(json.dumps(
        {"layers": {str(k): v for k, v in results.items()}, "removal_order": order,
         "tn_sets": {"1": order[:1], "2": sorted(order[:2]), "4": sorted(order[:4])}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/olmoe_experts"))
    parser.add_argument("--output", type=Path, default=Path("results/layer_tn_density"))
    args = parser.parse_args()
    partial = args.output / "partial"
    partial.mkdir(parents=True, exist_ok=True)
    results = {}
    for layer in args.layers:
        done = partial / f"layer{layer}.json"
        if done.exists():
            results[layer] = json.loads(done.read_text())
            continue
        started = time.time()
        for attempt in range(10):
            try:
                results[layer] = layer_density(layer, args.cache_dir)
                break
            except requests.RequestException as error:
                print(f"[layer {layer}] retry {attempt + 1}: {error.__class__.__name__}", flush=True)
                time.sleep(min(30 * (attempt + 1), 180))
        else:
            raise RuntimeError(f"layer {layer}: download kept failing")
        done.write_text(json.dumps(results[layer]))
        print(f"[layer {layer}] density {results[layer]['density_0.10']:.4f} "
              f"in {time.time() - started:.0f}s", flush=True)
    write_report(results, args.output)
    print((args.output / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
