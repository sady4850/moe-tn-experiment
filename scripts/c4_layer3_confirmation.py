"""Frozen cross-corpus evaluation; source compression implementation stays unchanged."""
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import urllib.request
from urllib.parse import urlsplit

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer
import moe_tn.compression_run as runner

SOURCE = Path('results/compression_crosslayer_3')
OUTPUT = Path('results/compression_layer3_c4')
REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
SHARD = 'en/c4-validation.00000-of-00008.json.gz'
COUNT = 256
SIZE = 512


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    signature = json.loads((SOURCE / 'metadata.json').read_text())['signature']
    h = hashlib.sha256()
    for path in sorted(Path('src/moe_tn').glob('*.py')):
        h.update(path.name.encode() + path.read_bytes())
    assert h.hexdigest() == signature['code_sha256'], 'Compression source changed'
    for name, version in signature['versions'].items():
        assert importlib.metadata.version(name) == version, name
    config = json.loads(json.dumps(signature['config']))
    config['output_dir'] = str(OUTPUT)
    config['data'].update(dataset='allenai/c4', subset='en', evaluation_blocks=COUNT,
                          evaluation_offset_blocks=0, evaluation_split='validation')
    plan = {
        'source_signature': signature, 'evaluation_config': config,
        'dataset_revision': REVISION, 'shard': SHARD,
        'selection': 'First 256 eligible distinct hostnames in shard order, unique text SHA256; at least 512 tokens; one uniformly sampled 512-token window per document using Python Random(17); no cross-document concatenation or added EOS.',
        'calibration': 'Unchanged WikiText train tokens and moments copied from source; config dataset describes evaluation only.',
        'primary': 'Mean paired NLL: tucker_input_r48 minus svd_input',
        'success': 'Mean below zero and paired document-bootstrap 95% percentile interval wholly below zero; 2000 replicates, NumPy default_rng(17).',
        'scope': 'English C4 validation shard 0, long pages, one page per hostname; fixed layer, rank, factors, calibration and payload. No model-pretraining disjointness claim.',
        'script_sha256': digest(Path(__file__)),
    }
    OUTPUT.mkdir(exist_ok=True)
    plan_path = OUTPUT / 'PLAN.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'Protocol changed'
    else:
        runner.save_json(plan_path, plan)
    Path('configs/compression_layer3_c4.yaml').write_text(yaml.safe_dump(config))
    files = [Path('calibration_blocks.pt'), Path('calibration_moments.pt')]
    for method in config['methods']:
        directory = Path(f'{method}_0.4')
        (OUTPUT / directory).mkdir(exist_ok=True)
        files.extend(directory / f'{p}.pt' for p in runner.PROJECTIONS)
    hashes = {}
    for relative in files:
        hashes[str(relative)] = digest(SOURCE / relative)
        if not (OUTPUT / relative).exists():
            shutil.copy2(SOURCE / relative, OUTPUT / relative)
        assert digest(OUTPUT / relative) == hashes[str(relative)], str(relative)
    runner.save_json(OUTPUT / 'frozen_candidates.json', {'source': str(SOURCE), 'sha256': hashes})
    print('Protocol and all factor/calibration checksums frozen before evaluation.', flush=True)
    token_path = OUTPUT / 'evaluation_blocks.pt'
    if not token_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(config['model']['id'], revision=config['model']['revision'])
        rng = random.Random(17)
        blocks, documents, hosts, texts = [], [], set(), set()
        url = f'https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/{SHARD}'
        with urllib.request.urlopen(url, timeout=60) as response, gzip.GzipFile(fileobj=response) as stream:
            for row_index, line in enumerate(stream):
                row = json.loads(line)
                host = (urlsplit(row['url']).hostname or '').lower().removeprefix('www.')
                text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
                if not host or host in hosts or text_hash in texts:
                    continue
                ids = tokenizer(row['text'], add_special_tokens=False)['input_ids']
                if len(ids) < SIZE:
                    continue
                start = rng.randrange(len(ids) - SIZE + 1)
                blocks.append(ids[start:start + SIZE])
                documents.append({'row': row_index, 'url': row['url'], 'hostname': host,
                                  'text_sha256': text_hash, 'document_tokens': len(ids), 'window_start': start})
                hosts.add(host)
                texts.add(text_hash)
                if len(blocks) == COUNT:
                    break
        assert len(blocks) == COUNT
        runner.save_json(OUTPUT / 'documents.json', documents)
        runner.save_tensor(token_path, torch.tensor(blocks, dtype=torch.long))
    documents = json.loads((OUTPUT / 'documents.json').read_text())
    blocks = torch.load(token_path, weights_only=True)
    assert blocks.shape == (COUNT, SIZE)
    assert len(documents) == len({d['hostname'] for d in documents}) == COUNT
    data_hashes = {'evaluation_blocks.pt': digest(token_path), 'documents.json': digest(OUTPUT / 'documents.json')}
    manifest = OUTPUT / 'evaluation_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == data_hashes
    else:
        runner.save_json(manifest, data_hashes)
    # Refuse any accidental factor regeneration or evaluation/calibration data fallback.
    def forbidden(*args, **kwargs):
        raise RuntimeError('Frozen evaluation may not regenerate factors or data')
    runner.decompose = forbidden
    runner.text_blocks = forbidden
    runner.run(config)
    for relative, expected in hashes.items():
        assert digest(OUTPUT / relative) == expected
    for relative, expected in data_hashes.items():
        assert digest(OUTPUT / relative) == expected
    baseline = np.array([r['nll'] for r in json.loads((OUTPUT / 'baseline_blocks.json').read_text())])
    results = {m: json.loads((OUTPUT / f'{m}_0.4/result.json').read_text()) for m in config['methods']}
    assert len(baseline) == COUNT
    for method, result in results.items():
        assert len(result['block_nll']) == COUNT
        previous = json.loads((SOURCE / f'{method}_0.4/result.json').read_text())
        assert result['projections'] == previous['projections'], 'Projection diagnostics changed'
        assert result['stored_bytes'] == previous['stored_bytes']
    delta = np.array(results['tucker_input_r48']['block_nll']) - np.array(results['svd_input']['block_nll'])
    ci = runner.paired_interval(delta, 17)
    summary = {'primary_difference': float(delta.mean()), 'paired_document_95_ci': ci,
               'criterion_passed': bool(delta.mean() < 0 and ci[1] < 0),
               'documents_better_than_svd': int((delta < 0).sum()), 'documents': COUNT,
               'scored_tokens': COUNT * (SIZE - 1), 'baseline_nll': float(baseline.mean()),
               'delta_nll': {m: r['delta_nll'] for m, r in results.items()},
               'checks': 'Factor/calibration/data SHA256 unchanged; projection diagnostics and payload exactly match development.'}
    runner.save_json(OUTPUT / 'confirmation_summary.json', summary)
    lines = ['# Frozen layer-3 cross-corpus confirmation: C4', '',
             '256 distinct-hostname documents; one 512-token window each; fixed WikiText calibration and factors.', '',
             '| Method | Delta NLL |', '|---|---:|']
    lines += [f"| {m} | {r['delta_nll']:+.8f} |" for m, r in results.items()]
    lines += ['', f'Primary Tucker r48 minus SVD: {delta.mean():+.8f}; paired document-bootstrap 95% CI {ci}.',
              f"Frozen success criterion passed: {summary['criterion_passed']}. Wins: {summary['documents_better_than_svd']}/256.",
              '', 'Scope: first eligible long pages from one C4 shard, one per hostname. Different sites can still be dependent.',
              'No recovery or recalibration; dense reconstruction measures quality, not inference memory/speed.',
              'Exact dataset revision, sampling, provenance and primary criterion are in PLAN.json.', '']
    (OUTPUT / 'CONFIRMATION.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
