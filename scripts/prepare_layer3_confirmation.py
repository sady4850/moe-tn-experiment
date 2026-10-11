"""Freeze the existing layer-3 candidates; copy no development losses to test."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import yaml

source = Path("results/compression_crosslayer_3")
config = yaml.safe_load(Path("configs/compression_layer3_confirmation.yaml").read_text())
signature = json.loads((source / "metadata.json").read_text())["signature"]
expected = json.loads(json.dumps(signature["config"]))
expected["data"].update(evaluation_split="test", evaluation_blocks=64, evaluation_offset_blocks=32)
expected["output_dir"] = "results/compression_layer3_confirmation"
assert config == expected, "Confirmation must change only evaluation text and output directory"
code_hash = hashlib.sha256()
for path in sorted(Path("src/moe_tn").glob("*.py")):
    code_hash.update(path.name.encode() + path.read_bytes())
assert code_hash.hexdigest() == signature["code_sha256"], "Source changed"
for package, version in signature["versions"].items():
    assert importlib.metadata.version(package) == version, f"Version changed: {package}"

output = Path(config["output_dir"])
output.mkdir()  # Deliberately refuse accidental overwrite of an existing confirmation.
files = [Path("calibration_blocks.pt"), Path("calibration_moments.pt")]
for method in config["methods"]:
    directory = Path(f"{method}_0.4")
    (output / directory).mkdir()
    files += [directory / f"{projection}.pt" for projection in ("gate_proj", "up_proj", "down_proj")]
hashes = {}
for relative in files:
    shutil.copy2(source / relative, output / relative)
    original = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == original
    hashes[str(relative)] = original
(output / "frozen_candidates.json").write_text(json.dumps({
    "source": str(source), "sha256": hashes,
    "primary_comparison": "tucker_input_r48 minus svd_input on test blocks 32..95",
    "success_rule": "negative mean paired NLL difference with paired block-bootstrap 95% CI below zero",
    "source_signature": signature,
}, indent=2) + "\n")
print("Frozen candidate factors and calibration verified; evaluation losses will be recomputed on new test text.")
