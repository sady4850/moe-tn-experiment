"""Reuse frozen WikiText-calibrated factors to replace one SVD projection with Tucker."""
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from moe_tn.adapter import OlmoeAdapter, PROJECTIONS
from moe_tn.compress import reconstruct_expert
from moe_tn.compression_run import evaluate_cached, original_layer, save_json
from moe_tn.evaluate import evaluate_nll
from moe_tn.model import load_model
from c4_layer3_confirmation import digest
from calibration_crossover import contrast, freeze

ROOT = Path('results/compression_layer3_projection_ablation')
FACTORS = Path('results/compression_crosslayer_3')
SOURCES = {'wikitext': Path('results/compression_layer3_confirmation'),
           'c4': Path('results/compression_layer3_c4')}
SVD = 'svd_input'
TUCKER = 'tucker_input_r48'
MIXES = {f'tucker_{p.removesuffix("_proj")}': {q: TUCKER if q == p else SVD for q in PROJECTIONS}
         for p in PROJECTIONS}


@torch.no_grad()
def install(adapter, mapping):
    assert set(mapping) == set(PROJECTIONS)
    stats = {}
    for projection, method in mapping.items():
        path = FACTORS / f'{method}_0.4/{projection}.pt'
        artifact = torch.load(path, weights_only=True)
        assert artifact['method'] == ('svd' if method == SVD else 'tucker')
        expected = json.loads((FACTORS / f'{method}_0.4/result.json').read_text())['projections'][projection]
        for key, value in artifact.items():
            if key != 'factors':
                assert value == expected[key], (projection, key)
        assert artifact['stored_bytes'] == sum(t.numel() * t.element_size() for t in artifact['factors'].values())
        assert artifact['retained_fraction'] <= 0.4
        artifact['factors'] = {k: v.to(adapter.device, torch.float32) for k, v in artifact['factors'].items()}
        for expert in range(adapter.num_experts):
            adapter.weight(expert, projection).copy_(reconstruct_expert(artifact, expert, adapter.device).to(adapter.dtype))
        stats[projection] = {'source_method': method, 'source_file': str(path),
                             'ranks': artifact['ranks'], 'stored_bytes': artifact['stored_bytes'],
                             'stored_parameters': artifact['stored_parameters'],
                             'original_parameters': artifact['original_parameters']}
    return stats


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()


def run():
    signature = json.loads((FACTORS / 'metadata.json').read_text())['signature']
    h = hashlib.sha256()
    for path in sorted(Path('src/moe_tn').glob('*.py')):
        h.update(path.name.encode() + path.read_bytes())
    assert h.hexdigest() == signature['code_sha256']
    for package, version in signature['versions'].items():
        assert importlib.metadata.version(package) == version
    plan = {'source_signature': signature, 'projection_mixes': MIXES,
            'script_sha256': digest(Path(__file__)),
            'helper_sha256': {p: digest(Path(__file__).with_name(p)) for p in ['c4_layer3_confirmation.py', 'calibration_crossover.py']},
            'fixed': 'Layer 3, original WikiText calibration, saved BF16 factors, fraction 0.4 per projection, unchanged expert/spatial ranks and whitening. No decomposition/calibration/training.',
            'evaluation': 'Reused WikiText test blocks 32..95 (64) and C4 documents (256); no data selection.',
            'comparisons': 'Each hybrid minus all-SVD per evaluated item. Lower is better. Also full-Tucker minus SVD and nonadditivity residual = full-Tucker minus SVD minus sum of three hybrid-minus-SVD effects.',
            'intervals': 'Paired item percentile bootstrap 2000 resamples, NumPy default_rng(17); exploratory, no multiplicity correction. WikiText blocks can be correlated.',
            'candidate_rule': 'A single-projection hybrid is a candidate only if its mean effect vs SVD is negative on both corpora; report intervals without claiming independent confirmation.',
            'controls': 'Recompute original baselines exactly; reconstruct full SVD and full Tucker with this installer and match prior first-block NLL on both corpora. Verify source hashes before/after.',
            'scope': 'Diagnostic reuse of known evaluations, single layer/calibration/budget. Nonadditivity combines pairwise and higher-order interactions; it does not identify each interaction. New-data confirmation required after selection. Dense reconstruction, no runtime memory/speed claim.'}
    freeze(ROOT / 'PLAN.json', plan)
    manifest = {}
    for method in (SVD, TUCKER):
        for relative in [f'{method}_0.4/{p}.pt' for p in PROJECTIONS] + [f'{method}_0.4/result.json']:
            manifest[str(FACTORS / relative)] = digest(FACTORS / relative)
    for corpus, source in SOURCES.items():
        frozen = json.loads((source / 'frozen_candidates.json').read_text())['sha256']
        for method in (SVD, TUCKER):
            for projection in PROJECTIONS:
                relative = f'{method}_0.4/{projection}.pt'
                assert frozen[relative] == manifest[str(FACTORS / relative)]
        for relative in ['evaluation_blocks.pt', 'baseline_blocks.json', 'metadata.json'] + [f'{m}_0.4/result.json' for m in (SVD, TUCKER)]:
            manifest[str(source / relative)] = digest(source / relative)
    freeze(ROOT / 'source_manifest.json', manifest)
    print('Frozen projection assignments, analysis and factor/data hashes.', flush=True)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load_model(signature['config']['model'], 17)
    model.eval()
    adapter = OlmoeAdapter(model, 3)
    save_json(ROOT / 'metadata.json', {'signature': signature, 'device': str(adapter.device),
              'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              'resolved_model_commit': getattr(model.config, '_commit_hash', None)})
    blocks, baseline, previous = {}, {}, {}
    for corpus, source in SOURCES.items():
        output = ROOT / corpus
        output.mkdir(exist_ok=True)
        blocks[corpus] = torch.load(source / 'evaluation_blocks.pt', weights_only=True)
        assert blocks[corpus].shape == (64 if corpus == 'wikitext' else 256, 512)
        print(f'Original baseline: {corpus}', flush=True)
        baseline[corpus] = evaluate_cached(model, blocks[corpus], output / 'baseline_blocks.json')
        assert json.loads((output / 'baseline_blocks.json').read_text()) == json.loads((source / 'baseline_blocks.json').read_text())
        previous[corpus] = {m: json.loads((source / f'{m}_0.4/result.json').read_text()) for m in (SVD, TUCKER)}
    controls = {}
    for method in (SVD, TUCKER):
        with original_layer(adapter):
            install(adapter, {p: method for p in PROJECTIONS})
            for corpus in SOURCES:
                measured = evaluate_nll(model, blocks[corpus][:1])['nll']
                assert measured == previous[corpus][method]['block_nll'][0], (method, corpus, measured)
                controls[f'{corpus}/{method}'] = measured
    save_json(ROOT / 'reconstruction_controls.json', controls)
    print('Original baselines and four reconstructed-control NLL values match exactly.', flush=True)
    results = {c: {} for c in SOURCES}
    for name, mapping in MIXES.items():
        with original_layer(adapter):
            stats = install(adapter, mapping)
            for corpus in SOURCES:
                print(f'Hybrid {name}: {corpus}', flush=True)
                trial = ROOT / corpus / name
                trial.mkdir(exist_ok=True)
                nll = evaluate_cached(model, blocks[corpus], trial / 'nll_blocks.json')
                assert len(nll) == len(baseline[corpus])
                result = {'variant': name, 'projection_sources': mapping, 'projections': stats,
                          'stored_bytes': sum(s['stored_bytes'] for s in stats.values()),
                          'retained_fraction': sum(s['stored_parameters'] for s in stats.values()) / sum(s['original_parameters'] for s in stats.values()),
                          'block_nll': nll.tolist(), 'delta_nll': float((nll - baseline[corpus]).mean()),
                          'minus_svd': contrast(nll - np.array(previous[corpus][SVD]['block_nll']))}
                save_json(trial / 'result.json', result)
                results[corpus][name] = result
    summary = {'corpora': {}, 'candidates_negative_on_both': [], 'scope': plan['scope']}
    lines = ['# Single-projection Tucker ablation', '',
             'Fixed WikiText-calibrated factors, layer 3, 40% per-projection payload ceiling; no training.', '',
             '| Evaluation | Variant | Retained payload | Delta NLL | Variant minus SVD | Paired 95% CI |',
             '|---|---|---:|---:|---:|---|']
    for corpus in SOURCES:
        svd = np.array(previous[corpus][SVD]['block_nll'])
        full_effect = np.array(previous[corpus][TUCKER]['block_nll']) - svd
        sum_effects = sum(np.array(results[corpus][n]['block_nll']) - svd for n in MIXES)
        summary['corpora'][corpus] = {'hybrids': {n: {k: v for k, v in results[corpus][n].items() if k != 'block_nll'} for n in MIXES},
                                      'full_tucker_minus_svd': contrast(full_effect),
                                      'sum_single_projection_effects': contrast(sum_effects),
                                      'nonadditivity_residual': contrast(full_effect - sum_effects)}
        for method in (SVD,):
            r = previous[corpus][method]
            lines.append(f"| {corpus} | all SVD | {r['retained_fraction']:.4%} | {r['delta_nll']:+.8f} | 0 | — |")
        for name, r in results[corpus].items():
            c = r['minus_svd']
            lo, hi = c['paired_95_ci']
            lines.append(f"| {corpus} | {name} | {r['retained_fraction']:.4%} | {r['delta_nll']:+.8f} | {c['mean']:+.8f} | [{lo:+.8f}, {hi:+.8f}] |")
        r = previous[corpus][TUCKER]
        c = contrast(full_effect)
        lo, hi = c['paired_95_ci']
        lines.append(f"| {corpus} | all Tucker r48 | {r['retained_fraction']:.4%} | {r['delta_nll']:+.8f} | {c['mean']:+.8f} | [{lo:+.8f}, {hi:+.8f}] |")
    summary['candidates_negative_on_both'] = [n for n in MIXES if all(results[c][n]['minus_svd']['mean'] < 0 for c in SOURCES)]
    for corpus in SOURCES:
        r = summary['corpora'][corpus]['nonadditivity_residual']
        lines += ['', f"{corpus} nonadditivity (full effect minus sum of single effects): {r['mean']:+.8f}, CI {r['paired_95_ci']}."]
    for path, expected in manifest.items():
        assert digest(Path(path)) == expected, path
    for corpus in SOURCES:
        assert evaluate_nll(model, blocks[corpus][:1])['nll'] == baseline[corpus][0], 'Original restoration failed'
    summary['checks'] = 'Source SHA256 unchanged, exact original baselines, exact four full-method reconstruction controls, exact final original-model restoration checks. All six hybrids use declared saved projection factors.'
    lines += ['', f"Candidates with negative mean on both corpora: {summary['candidates_negative_on_both']}", '', plan['intervals'], '', plan['scope'], '', summary['checks'], '']
    save_json(ROOT / 'summary.json', summary)
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
