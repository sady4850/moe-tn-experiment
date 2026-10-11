"""Shared input moments for OLMoE; no backward pass or weight updates."""
import torch
import torch.nn.functional as F


def regularized_cholesky(moment, relative_floor=1e-3):
    """Normalize by sample count upstream; floor eigenvalues relative to mean variance."""
    if relative_floor <= 0 or not torch.isfinite(moment).all():
        raise ValueError("Invalid covariance or regularization")
    covariance = (moment.float() + moment.float().T) / 2
    values, vectors = torch.linalg.eigh(covariance)
    floor = max(float(covariance.diag().mean()) * relative_floor, 1e-12)
    regularized = (vectors * values.clamp_min(floor)) @ vectors.T
    regularized = (regularized + regularized.T) / 2
    return torch.linalg.cholesky(regularized), {
        "eigenvalue_floor": floor, "clamped_eigenvalues": int((values < floor).sum()),
        "min_eigenvalue": float(values.min()), "max_eigenvalue": float(values.max())}


@torch.inference_mode()
def collect_moments(model, adapter, blocks, device, state=None, checkpoint=None):
    """Pool X^T X over block inputs and actual routed expert intermediate inputs.

    All experts share each projection's metric. Down-projection samples are
    pooled per token/expert assignment, without router-probability weighting.
    This matches a common linear-input metric, not the full MoE output loss.
    """
    model.eval()
    if state is None:
        hidden = adapter.weight(0, "up_proj").shape[1]
        intermediate = adapter.weight(0, "down_proj").shape[1]
        state = {"processed_blocks": 0, "hidden_rows": 0, "down_rows": 0,
                 "expert_counts": torch.zeros(adapter.num_experts, dtype=torch.long),
                 "hidden_sum": torch.zeros(hidden, hidden),
                 "down_sum": torch.zeros(intermediate, intermediate)}
    hidden_sum = state["hidden_sum"].to(device)
    down_sum = state["down_sum"].to(device)
    for index in range(state["processed_blocks"], len(blocks)):
        captured = []
        def capture(module, args):
            captured.append(args[0].detach().reshape(-1, args[0].shape[-1]).clone())
        hook = adapter.block.register_forward_pre_hook(capture)
        try:
            model(input_ids=blocks[index:index + 1].to(adapter.device), use_cache=False)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one invocation of the target MoE block")
        x = captured.pop()
        xf = x.to(device, torch.float32)
        hidden_sum += xf.T @ xf
        state["hidden_rows"] += len(x)
        router = adapter.block.gate(x)
        if isinstance(router, tuple) and len(router) == 3:
            selected = router[2]
        else:
            logits = router[0] if isinstance(router, tuple) else router
            selected = logits.float().softmax(-1).topk(adapter.top_k, dim=-1).indices
        for expert in range(adapter.num_experts):
            routed = x[(selected == expert).any(dim=-1)]
            if len(routed) == 0:
                continue
            act = (adapter.block.experts[expert].act_fn if adapter.layout == "module_list"
                   else adapter.block.experts.act_fn)
            if adapter.layout == "stacked":
                gate, up = F.linear(routed, adapter.block.experts.gate_up_proj[expert]).chunk(2, dim=-1)
            else:
                gate = F.linear(routed, adapter.weight(expert, "gate_proj"))
                up = F.linear(routed, adapter.weight(expert, "up_proj"))
            intermediate = (act(gate) * up).to(device, torch.float32)
            down_sum += intermediate.T @ intermediate
            state["expert_counts"][expert] += len(routed)
            state["down_rows"] += len(routed)
        state["processed_blocks"] = index + 1
        state["hidden_sum"] = hidden_sum.cpu()
        state["down_sum"] = down_sum.cpu()
        if checkpoint is not None:
            checkpoint(state)
        print(f"  calibration {index + 1}/{len(blocks)}", flush=True)
    if state["down_rows"] != state["hidden_rows"] * adapter.top_k:
        raise RuntimeError("Routed calibration coverage does not match top-k")
    return state
