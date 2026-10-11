"""Diagnostic (not pre-registered): why does pruning lower GPT-OSS-20B NLL on C4 but not on WikiText?"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, "scripts")
from expert_removal_gptoss import compressed, load_gptoss  # noqa: E402

config = yaml.safe_load(Path("configs/expert_removal_gptoss20b.yaml").read_text())
out = Path(config["output_dir"])
model, tokenizer = load_gptoss(config)
model.eval()
blocks = torch.load(out / "token_blocks.pt")
plans = {"base": None, **{v: json.loads((out / "plans" / f"{v}.json").read_text()) for v in ("tn_prune", "frequency")}}
bins = [(0, 16), (16, 128), (128, 511)]
result = {}
for name, plan in plans.items():
    with compressed(model, plan), torch.inference_mode():
        for corpus in ("c4", "wikitext_test"):
            data = blocks[corpus][:48].to(model.device)
            nll, ent, top1, maxp = [], [], [], []
            for i in range(len(data)):
                logits = model(data[i:i + 1]).logits[0, :-1].float()
                logp = F.log_softmax(logits, -1)
                target = data[i, 1:]
                nll.append(-logp.gather(1, target[:, None])[:, 0].cpu())
                ent.append(-(logp.exp() * logp).sum(-1).cpu())
                top1.append((logits.argmax(-1) == target).float().cpu())
                maxp.append(logp.max(-1).values.exp().cpu())
            nll, ent, top1, maxp = (torch.stack(x) for x in (nll, ent, top1, maxp))
            r = {"nll": nll.mean().item(), "entropy": ent.mean().item(), "top1": top1.mean().item(),
                 "mean_max_prob": maxp.mean().item(),
                 "nll_by_position": {f"{a}-{b}": nll[:, a:b].mean().item() for a, b in bins},
                 "share_tokens_p_below_1e-4": (nll > 9.21).float().mean().item(),
                 "nll_from_p_below_1e-4": (nll * (nll > 9.21)).sum().item() / nll.sum().item()}
            result[f"{name}/{corpus}"] = r
            print(name, corpus, json.dumps(r), flush=True)
Path("c4_diagnostic.json").write_text(json.dumps(result, indent=1))
