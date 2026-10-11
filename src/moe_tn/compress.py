"""Direct compression with optional shared input whitening (FP32 decomposition)."""
from __future__ import annotations

import math

import torch


METHODS = ("svd", "shared", "tucker")


def balanced_ranks(shape, budget, expert_rank=None):
    """TD-MoE's fixed expert rank / balanced spatial ranks, with a strict ceiling.

    Adapted algorithmically from calculate_tucker_ranks_balanced in TD-MoE;
    unlike upstream, candidates exceeding the payload budget are excluded.
    """
    e, o, i = shape
    re = e if expert_rank is None else expert_rank
    if not isinstance(re, int) or not 1 <= re <= e:
        raise ValueError("Expert rank must be between 1 and the number of experts")
    best = None
    for ro in range(max(1, o // 50), o + 1, max(1, o // 200)):
        for ri in range(max(1, i // 50), i + 1, max(1, i // 200)):
            count = re * ro * ri + e * re + o * ro + i * ri
            if count > budget:
                continue
            score = 10 * (budget - count) / budget + abs(ro / o - ri / i)
            if best is None or score < best[0]:
                best = (score, (re, ro, ri))
    if best is None:
        raise ValueError("Budget too small for balanced Tucker")
    return best[1]


def eigenbasis(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(gram)
    return values.flip(0).clamp_min(0), vectors.flip(1)


def tucker_ranks(shape: tuple[int, int, int], budget: int,
                 spectra: list[torch.Tensor]) -> tuple[int, int, int]:
    """Minimize the HOSVD discarded-energy upper bound on a fixed rank grid.

    Expert ranks are exhaustive; output ranks use steps of 16 for large matrices.
    For each pair choose the largest input rank fitting the exact factor budget.
    This is a deterministic weight-space heuristic, not validation-set tuning.
    """
    e, o, i = shape
    retained = [torch.cat([torch.zeros(1), s.double().cpu().cumsum(0)]) for s in spectra]
    output_ranks = sorted(set(range(1, o + 1, 1 if o <= 64 else 16)) | {o})
    best = None
    for re in range(1, e + 1):
        for ro in output_ranks:
            ri = min(i, (budget - e * re - o * ro) // (i + re * ro))
            if ri < 1:
                continue
            score = float(retained[0][re] + retained[1][ro] + retained[2][ri])
            if best is None or score > best[0]:
                best = (score, (re, ro, ri))
    if best is None:
        raise ValueError("Budget too small for Tucker factors")
    return best[1]


@torch.no_grad()
def decompose(weights: torch.Tensor, method: str, fraction: float,
              storage_dtype: torch.dtype, input_cholesky=None,
              rank_policy="energy", expert_rank=None) -> dict:
    if method not in METHODS or not 0 < fraction <= 1:
        raise ValueError("Unknown method or invalid retained fraction")
    w = weights.float()
    if input_cholesky is not None:
        input_cholesky = input_cholesky.to(w.device, torch.float32)
        w = w @ input_cholesky
    e, o, i = w.shape
    budget = math.floor(w.numel() * fraction)
    if method == "svd":
        r = min(o, i, budget // (e * (o + i)))
        if r < 1:
            raise ValueError("Budget too small for SVD factors")
        left, right = [], []
        for expert in w:
            u, s, vh = torch.linalg.svd(expert, full_matrices=False)
            left.append((u[:, :r] * s[:r]).cpu())
            right.append(vh[:r].cpu())
        factors = {"left": torch.stack(left), "right": torch.stack(right)}
        ranks = [r]
        orientation = None
    else:
        # Gram matrices avoid explicitly copying the large unfolded weight tensor.
        so, uo = eigenbasis(torch.einsum("eoi,epi->op", w, w))
        si, ui = eigenbasis(torch.einsum("eoi,eoj->ij", w, w))
        if method == "shared":
            ro = min(o, budget // (o + e * i))
            ri = min(i, budget // (i + e * o))
            if max(ro, ri) < 1:
                raise ValueError("Budget too small for shared factors")
            if ro >= 1 and (ri < 1 or so[:ro].sum() >= si[:ri].sum()):
                factors = {"basis": uo[:, :ro],
                           "coeff": torch.einsum("oa,eoi->eai", uo[:, :ro], w)}
                ranks, orientation = [ro], "output"
            else:
                factors = {"basis": ui[:, :ri], "coeff": w @ ui[:, :ri]}
                ranks, orientation = [ri], "input"
        else:
            se, ue = eigenbasis(w.flatten(1) @ w.flatten(1).T)
            if rank_policy not in ("energy", "td_balanced"):
                raise ValueError("Unknown Tucker rank policy")
            if expert_rank is not None and rank_policy != "td_balanced":
                raise ValueError("Explicit expert rank requires td_balanced policy")
            ranks = list(balanced_ranks((e, o, i), budget, expert_rank) if rank_policy == "td_balanced"
                         else tucker_ranks((e, o, i), budget, [se, so, si]))
            re, ro, ri = ranks
            # Explicit contraction order bounds intermediates by the original tensor size.
            core = torch.einsum("ea,eoi->aoi", ue[:, :re], w)
            core = torch.einsum("ob,aoi->abi", uo[:, :ro], core)
            core = core @ ui[:, :ri]
            factors = {"expert": ue[:, :re], "output": uo[:, :ro],
                       "input": ui[:, :ri], "core": core}
            orientation = None
    if input_cholesky is not None:
        # Absorb L^-1 into existing factors BEFORE storage rounding. No extra
        # dense whitening matrix is required by the compressed representation.
        if method == "svd" or (method == "shared" and orientation == "output"):
            key = "right" if method == "svd" else "coeff"
            value = factors[key].to(w.device)
            factors[key] = torch.linalg.solve_triangular(
                input_cholesky.T, value.transpose(-1, -2), upper=True).transpose(-1, -2)
        else:
            key = "input" if method == "tucker" else "basis"
            factors[key] = torch.linalg.solve_triangular(
                input_cholesky.T, factors[key].to(w.device), upper=True)
    factors = {key: value.to(storage_dtype).cpu() for key, value in factors.items()}
    count = sum(value.numel() for value in factors.values())
    assert count <= budget
    return {"method": method, "shape": list(w.shape), "ranks": ranks,
            "whitening": "input" if input_cholesky is not None else "none",
            "rank_policy": rank_policy if method == "tucker" else None,
            "orientation": orientation, "factors": factors,
            "original_parameters": w.numel(), "stored_parameters": count,
            "stored_bytes": sum(v.numel() * v.element_size() for v in factors.values()),
            "retained_fraction": count / w.numel()}


def reconstruct_expert(artifact: dict, expert: int, device: torch.device) -> torch.Tensor:
    # Reconstruct rounded stored factors in FP32; caller casts to model dtype.
    f = {key: value.to(device=device, dtype=torch.float32)
         for key, value in artifact["factors"].items()}
    if artifact["method"] == "svd":
        return f["left"][expert] @ f["right"][expert]
    if artifact["method"] == "shared":
        if artifact["orientation"] == "output":
            return f["basis"] @ f["coeff"][expert]
        return f["coeff"][expert] @ f["basis"].T
    middle = torch.einsum("a,abc->bc", f["expert"][expert], f["core"])
    return f["output"] @ middle @ f["input"].T
