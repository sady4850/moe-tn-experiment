"""Fixed layer/rank/budget calibration-domain diagnostic on reused evaluation sets."""
import gc
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
from c4_layer3_confirmation import digest, REVISION

ROOT = Path('results/compression_layer3_calibration_crossover')
SOURCES = {'c4': Path('results/compression_layer3_c4'),
           'wikitext': Path('results/compression_layer3_confirmation')}
TRAIN_SHARD = 'en/c4-train.00000-of-01024.json.gz'
METHODS = ['svd_input', 'tucker_input_r64', 'tucker_input_r48']


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Frozen artifact changed: {path}'
    else:
        runner.save_json(path, value)


def verified_copy(source, target):
    if not target.exists():
        shutil.copy2(source, target)
    assert digest(source) == digest(target), str(target)


def contrast(values):
    values = np.asarray(values)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    return {'mean': float(values.mean()), 'paired_95_ci': runner.paired_interval(values, 17)}


def main():
    signature = json.loads((SOURCES['c4'] / 'metadata.json').read_text())['signature']
    h = hashlib.sha256()
    for path in sorted(Path('src/moe_tn').glob('*.py')):
        h.update(path.name.encode() + path.read_bytes())
    assert h.hexdigest() == signature['code_sha256']
    for package, version in signature['versions'].items():
        assert importlib.metadata.version(package) == version
    ROOT.mkdir(exist_ok=True)
    plan = {
        'source_signature': signature,
        'script_sha256': digest(Path(__file__)),
        'helper_sha256': digest(Path(__file__).with_name('c4_layer3_confirmation.py')),
        'calibration': {'dataset': 'allenai/c4', 'revision': REVISION, 'shard': TRAIN_SHARD,
                        'blocks': 64, 'block_size': 512, 'seed': 17,
                        'selection': 'First eligible distinct hostnames and unique text hashes; exclude all C4 evaluation hostnames/text hashes. One random 512-token window per document >=512 tokens; no EOS or concatenation.'},
        'fixed': 'OLMoE layer 3, BF16, fraction 0.4, SVD and Tucker r64/r48, td_balanced ranks, whitening floor 0.001; no recovery or rank selection.',
        'evaluation': 'Reuse exact C4 256-document windows and WikiText test blocks 32..95. Recompute original baseline and compare exactly with prior baseline.',
        'primary': '(Tucker48 - SVD) under C4 calibration minus (Tucker48 - SVD) under WikiText calibration, on C4 evaluation; negative means relative improvement from recalibration.',
        'analysis': 'Paired percentile bootstrap of per-item contrasts, 2000 resamples, NumPy default_rng(17). Also report new method gaps, each method calibration effect, and same contrasts on WikiText.',
        'interpretation': 'Diagnostic reuse of observed evaluations, one calibration sample per corpus. Corpus, sampling/packing and sample effects are not separated. Intervals conditional on factors and evaluated items; WikiText adjacent blocks may be correlated. Not a fresh confirmation test.'}
    freeze(ROOT / 'PLAN.json', plan)
    originals = {}
    for name, source in SOURCES.items():
        for relative in ['evaluation_blocks.pt', 'baseline_blocks.json', 'metadata.json'] + [f'{m}_0.4/result.json' for m in METHODS]:
            originals[str(source / relative)] = digest(source / relative)
    originals[str(SOURCES['c4'] / 'documents.json')] = digest(SOURCES['c4'] / 'documents.json')
    freeze(ROOT / 'source_manifest.json', originals)
    print('Frozen calibration diagnostic protocol and existing evaluation/result checksums.', flush=True)
    calibration_path = ROOT / 'calibration_blocks.pt'
    if not calibration_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(signature['config']['model']['id'], revision=signature['config']['model']['revision'])
        evaluation_docs = json.loads((SOURCES['c4'] / 'documents.json').read_text())
        hosts = {d['hostname'] for d in evaluation_docs}
        texts = {d['text_sha256'] for d in evaluation_docs}
        blocks, documents = [], []
        rng = random.Random(17)
        url = f'https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/{TRAIN_SHARD}'
        with urllib.request.urlopen(url, timeout=60) as response, gzip.GzipFile(fileobj=response) as stream:
            for row_index, line in enumerate(stream):
                row = json.loads(line)
                host = (urlsplit(row['url']).hostname or '').lower().removeprefix('www.')
                text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
                if not host or host in hosts or text_hash in texts:
                    continue
                ids = tokenizer(row['text'], add_special_tokens=False)['input_ids']
                if len(ids) < 512:
                    continue
                start = rng.randrange(len(ids) - 511)
                blocks.append(ids[start:start + 512])
                documents.append({'row': row_index, 'url': row['url'], 'hostname': host,
                                  'text_sha256': text_hash, 'document_tokens': len(ids), 'window_start': start})
                hosts.add(host)
                texts.add(text_hash)
                if len(blocks) == 64:
                    break
        assert len(blocks) == 64
        runner.save_json(ROOT / 'calibration_documents.json', documents)
        runner.save_tensor(calibration_path, torch.tensor(blocks, dtype=torch.long))
    docs = json.loads((ROOT / 'calibration_documents.json').read_text())
    eval_docs = json.loads((SOURCES['c4'] / 'documents.json').read_text())
    assert len(docs) == len({d['hostname'] for d in docs}) == 64
    assert not ({d['hostname'] for d in docs} & {d['hostname'] for d in eval_docs})
    assert not ({d['text_sha256'] for d in docs} & {d['text_sha256'] for d in eval_docs})
    assert torch.load(calibration_path, weights_only=True).shape == (64, 512)
    freeze(ROOT / 'calibration_manifest.json', {str(p): digest(p) for p in [calibration_path, ROOT / 'calibration_documents.json']})

    def forbidden(*args, **kwargs):
        raise RuntimeError('Cached data required; second evaluation must reuse new factors')
    runner.text_blocks = forbidden
    for corpus, source in SOURCES.items():
        output = ROOT / corpus
        output.mkdir(exist_ok=True)
        config = json.loads(json.dumps(signature['config']))
        config['output_dir'] = str(output)
        config['methods'] = METHODS
        config['data'].update(dataset='allenai/c4', subset='en', calibration_split='train',
                              evaluation_split='validation' if corpus == 'c4' else 'test',
                              evaluation_blocks=256 if corpus == 'c4' else 64, evaluation_offset_blocks=0)
        # Token file provenance, not generic loader fields, defines each evaluation corpus.
        Path(f'configs/calibration_crossover_{corpus}.yaml').write_text(yaml.safe_dump(config))
        verified_copy(calibration_path, output / 'calibration_blocks.pt')
        verified_copy(source / 'evaluation_blocks.pt', output / 'evaluation_blocks.pt')
        if corpus == 'wikitext':
            verified_copy(ROOT / 'c4/calibration_moments.pt', output / 'calibration_moments.pt')
            for method in METHODS:
                (output / f'{method}_0.4').mkdir(exist_ok=True)
                for projection in runner.PROJECTIONS:
                    relative = Path(f'{method}_0.4/{projection}.pt')
                    verified_copy(ROOT / 'c4' / relative, output / relative)
            runner.decompose = forbidden
        print(f'Running C4-calibrated candidates on {corpus}', flush=True)
        runner.run(config)
        baseline = json.loads((output / 'baseline_blocks.json').read_text())
        assert baseline == json.loads((source / 'baseline_blocks.json').read_text()), 'Baseline changed'
        gc.collect()
        torch.cuda.empty_cache()
    summary = {'scope': plan['interpretation'], 'corpora': {}}
    lines = ['# Calibration-domain crossover diagnostic', '',
             'Fixed layer 3, retained payload 40%, no rank selection or recovery. Same evaluation tokens in both calibration conditions.', '',
             '| Evaluation | Calibration | SVD delta NLL | Tucker r64 delta NLL | Tucker r48 delta NLL |',
             '|---|---|---:|---:|---:|']
    for corpus, source in SOURCES.items():
        old = {m: json.loads((source / f'{m}_0.4/result.json').read_text()) for m in METHODS}
        new = {m: json.loads((ROOT / corpus / f'{m}_0.4/result.json').read_text()) for m in METHODS}
        expected_count = 256 if corpus == 'c4' else 64
        for method in METHODS:
            assert len(new[method]['block_nll']) == len(old[method]['block_nll']) == expected_count
            assert new[method]['stored_bytes'] == old[method]['stored_bytes']
            for projection in runner.PROJECTIONS:
                assert new[method]['projections'][projection]['ranks'] == old[method]['projections'][projection]['ranks']
        old_gap = np.array(old['tucker_input_r48']['block_nll']) - np.array(old['svd_input']['block_nll'])
        new_gap = np.array(new['tucker_input_r48']['block_nll']) - np.array(new['svd_input']['block_nll'])
        summary['corpora'][corpus] = {
            'wiki_calibration_delta_nll': {m: old[m]['delta_nll'] for m in METHODS},
            'c4_calibration_delta_nll': {m: new[m]['delta_nll'] for m in METHODS},
            'old_tucker48_minus_svd': contrast(old_gap), 'new_tucker48_minus_svd': contrast(new_gap),
            'relative_calibration_effect': contrast(new_gap - old_gap),
            'per_method_calibration_effect': {m: contrast(np.array(new[m]['block_nll']) - np.array(old[m]['block_nll'])) for m in METHODS}}
        for label, results in [('WikiText', old), ('C4', new)]:
            lines.append(f"| {corpus} | {label} | " + ' | '.join(f"{results[m]['delta_nll']:+.8f}" for m in METHODS) + ' |')
    for path, expected in originals.items():
        assert digest(Path(path)) == expected, path
    new_hashes = {}
    for method in METHODS:
        for projection in runner.PROJECTIONS:
            relative = Path(f'{method}_0.4/{projection}.pt')
            assert digest(ROOT / 'c4' / relative) == digest(ROOT / 'wikitext' / relative)
            new_hashes[str(relative)] = digest(ROOT / 'c4' / relative)
        a = json.loads((ROOT / 'c4' / f'{method}_0.4/result.json').read_text())
        b = json.loads((ROOT / 'wikitext' / f'{method}_0.4/result.json').read_text())
        assert a['projections'] == b['projections']
    freeze(ROOT / 'new_factor_manifest.json', new_hashes)
    for corpus in SOURCES:
        row = summary['corpora'][corpus]
        lines += ['', f"{corpus}: relative calibration effect {row['relative_calibration_effect']}; new Tucker48 minus SVD {row['new_tucker48_minus_svd']}."]
    lines += ['', plan['interpretation'], '', 'Baselines match previous runs exactly; ranks and payload unchanged; new factors identical across both evaluations. No inference memory/speed claim.', '']
    summary['checks'] = 'Unchanged old source artifacts; exact baseline match; unchanged ranks and bytes; same new factors/projection statistics across evaluations; disjoint calibration/evaluation hostnames and text hashes.'
    runner.save_json(ROOT / 'summary.json', summary)
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
