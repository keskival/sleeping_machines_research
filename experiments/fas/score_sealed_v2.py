"""B3 Stage 4: source-bound, once-only inference for the selected configurations.

An immutable admission binds result, checkpoint, sources and frozen dataset.
Validation reconstruction passes before a durable TEST reservation. All TEST
runs are scored, in generator order. No optimizer or model fitting is used.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/fas'))
sys.path.insert(0, str(ROOT/'experiments/tpp'))
from sealed_registry import atomic_json, complete, reserve, sha, sync_directory


def verify_admission(admission, root=ROOT):
    """Metadata/hash checks only: no NPZ arrays are loaded."""
    if admission['kind'] not in ('native', 'lstm', 'transformer', 'classical'):
        raise ValueError('Only the selected information-matched families are eligible')
    for path, expected in admission['source_sha256'].items():
        if sha(root/path) != expected:
            raise ValueError('Pinned scoring/model source changed: '+path)
    data = root/'experiments/data/fas'/admission['data']
    if sha(data/'manifest.json') != admission['data_manifest_sha256']:
        raise ValueError('Frozen dataset manifest changed')
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest['name'] != admission['data'] or manifest['args']['K'] != 2:
        raise ValueError('Selected K=2 dataset required')
    for split in ('train_clean', 'val_clean', 'val_faulty', 'test_clean', 'test_faulty'):
        if sha(data/(split+'.npz')) != manifest['splits'][split]['sha256']:
            raise ValueError('Frozen dataset bytes changed: '+split)
    if admission['kind'] == 'classical':
        if admission['fit_runs'] != 2000:
            raise ValueError('Classical calibration recipe must remain at 2000 clean TRAIN runs')
        if 'result' in admission:
            path = root/admission['result']
            if sha(path) != admission['result_sha256']:
                raise ValueError('Classical validation result changed')
            result = json.loads(path.read_text())
            if result['status'] != 'completed' or result['split'] != 'val' or result['data'] != admission['data'] or result['fit_runs'] != 2000:
                raise ValueError('Selected anonymous classical validation recipe required')
            if any(admission['source_sha256'].get(path) != expected for path,expected in result['source_sha256'].items()):
                raise ValueError('Classical fitting/scoring source changed')
            return result, manifest
        return None, manifest
    path = root/admission['result']
    if sha(path) != admission['result_sha256']:
        raise ValueError('Selected validation result changed')
    result = json.loads(path.read_text()); args = result['args']
    if result['status'] != 'completed' or args['data'] != admission['data']:
        raise ValueError('Completed selected result on the frozen dataset required')
    if args['seed'] != admission['seed'] or args['tag'] != admission['trained_tag']:
        raise ValueError('Selected seed/tag differs')
    if any(admission['source_sha256'].get(path) != expected
           for path, expected in result['source_sha256'].items()):
        raise ValueError('Selected model sources differ from scoring sources')
    if sha(root/admission['checkpoint']) != admission['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    if admission['kind'] == 'native':
        frozen = dict(pred_window=64, keyed=0, epochs=3, d=32, modes=16, layers=2,
                      n_exp=2, n_lognormal=8, n_window=4, dv=4, dropout=.1, lr=.003,
                      batch=16, fit_runs=5000, train_max_events=1024, max_events=1100, eval_runs=1000)
        if args['seed'] not in (6, 7, 8) or any(args.get(k) != v for k,v in frozen.items()):
            raise ValueError('Frozen C10 native confirmation recipe required')
        if result['checkpoint'] != admission['checkpoint'] or 'not scored' not in str(result['test_auroc']):
            raise ValueError('Native selected checkpoint differs or result already contains TEST')
    else:
        if args['model'] != admission['kind'] or args['seed'] not in (0, 1, 2):
            raise ValueError('Selected reference family/seeds required')
        if args['d'] != 128 or args['layers'] != 2 or args['lr'] != .003 or not args['no_test']:
            raise ValueError('Frozen reference grid winner and validation-only result required')
        cp = path.parent/result['selected_weights']['path']
        if cp.resolve() != (root/admission['checkpoint']).resolve() or result['selected_weights']['sha256'] != admission['checkpoint_sha256']:
            raise ValueError('Reference selected checkpoint differs')
        if result['data_manifest_sha256'] != admission['data_manifest_sha256']:
            raise ValueError('Reference fit used different frozen data')
    return result, manifest


def native_model(result, checkpoint):
    import numpy as np
    import torch
    a = result['args']; torch.set_default_dtype(torch.float64)
    module = importlib.import_module(next(Path(k).stem for k in result['source_sha256'] if 'race_tpp_fas' in k))
    d = ROOT/'experiments/data/fas'/a['data']
    train, _ = module.load(d/'train_clean.npz', a['train_max_events']); train = train[:a['fit_runs']]
    gaps = np.concatenate([np.diff(t) for _, t in train[:500]]); pos = gaps[gaps > 0]
    scale = float(np.median(pos))
    if not math.isclose(scale, result['scale_s'], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Native training scale reconstruction differs')
    qs = np.log(np.quantile(pos, np.linspace(.1, .9, a['n_lognormal']))).tolist()
    edges = module.cluster_windows(pos, a['n_window'], a['seed']) if a['n_window'] else None
    args = (46, a['d'], a['modes'], a['layers'], a['n_exp'], a['n_lognormal'], a['dv'], a['dropout'],
            scale, qs, a['cell_s'], a['n_window'], edges, 0)
    kw = dict(dk=a['dk'], local=a['keyed'] == 2)
    if a.get('pred_window'): kw.update(pred_window=a['pred_window'], keyed=bool(a['keyed']))
    if a.get('consume'): kw['consume'] = True
    model = module.KeyedRaceFAS(*args, **kw) if a['keyed'] or a.get('pred_window') else module.RaceTPP(*args)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)); model.eval()
    def score(runs):
        pt, pg, nll = module.per_position(model, runs, a.get('eval_batch', 8))
        return module.prefix_scores(pt, pg, [len(r[0]) for r in runs]), nll
    return module, score


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(); parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True); parser.add_argument('--tag', required=True)
    args = parser.parse_args(); start = time.monotonic()
    if sha(ROOT/args.admission) != args.admission_sha256:
        raise ValueError('Scoring admission changed')
    admission = json.loads((ROOT/args.admission).read_text())
    result, manifest = verify_admission(admission)
    torch.set_num_threads(1)
    kind = admission['kind']; data = ROOT/'experiments/data/fas'/admission['data']
    validation = None
    if kind == 'classical':
        module = importlib.import_module('classical_scores_v2')
        train, _ = module.runs(data/'train_clean.npz'); fns = module.fit(train[:admission['fit_runs']])
        del train
        score = lambda runs: (module.score_runs(fns, runs), None)
        load = lambda path: module.runs(path)
        if result is None:
            raise ValueError('Classical scoring requires the completed validation result')
        vc, _ = load(data/'val_clean.npz'); vf, _ = load(data/'val_faulty.npz')
        cc, _ = score(vc); ff, _ = score(vf)
        validation = max(abs(module.auroc(cc[name][:,j],ff[name][:,j])-result['auroc'][name][str(n)])
                         for name in fns for j,n in enumerate(module.PREFIXES))
        if not math.isfinite(validation) or validation > 1e-12:
            raise ValueError('Classical validation AUROC fails reconstruction')
    else:
        a = result['args']
        if kind == 'native':
            module, score = native_model(result, ROOT/admission['checkpoint'])
        else:
            torch.set_default_dtype(torch.float32)
            module = importlib.import_module('dense'); model = module.build(kind, a['d'], a['layers'])
            model.load_state_dict(torch.load(ROOT/admission['checkpoint'], map_location='cpu', weights_only=True)); model.eval()
            score = lambda runs: module.scores(model, runs, a['eval_lanes'])
        load = lambda path: module.load(path, a['max_events'])
        clean, _ = load(data/'val_clean.npz')
        _, validation = score(clean[:a['eval_runs']])
        expected = result.get('val_clean_nll', result.get('selected_val_clean_nll'))
        # dense.py calls this field validation_clean_nll in some recorded versions.
        if expected is None: expected = result.get('validation_clean_nll')
        if expected is None:
            expected = min(row['val_clean_nll'] for row in result['curve'])
        tolerance = 1e-9 if kind == 'native' else 1e-5
        if not math.isfinite(validation) or abs(validation-expected) > tolerance:
            raise ValueError('Selected validation likelihood fails reconstruction')

    directory = ROOT/'experiments/results/fas'; tag = admission['trained_tag']
    config = result['args'] if kind != 'classical' else dict(data=admission['data'], fit_runs=admission['fit_runs'], kind='classical')
    config = {k:v for k,v in config.items() if k != 'tag'}
    identity = hashlib.sha256(json.dumps(dict(kind=kind, config=config,
        checkpoint=admission.get('checkpoint_sha256'), data=admission['data_manifest_sha256']), sort_keys=True).encode()).hexdigest()
    provenance = dict(kind=kind, seed=admission.get('seed'), data=admission['data'],
                      data_manifest_sha256=admission['data_manifest_sha256'], config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                      checkpoint_sha256=admission.get('checkpoint_sha256'), admission_sha256=args.admission_sha256,
                      source_sha256=admission['source_sha256'], reason='Registered Stage 4 first and only scoring')
    reservation = reserve(directory, tag, identity, provenance)
    # The first access to TEST arrays occurs only after the durable reservation.
    clean, _ = load(data/'test_clean.npz'); faulty, kinds = load(data/'test_faulty.npz')
    if len(clean) != manifest['splits']['test_clean']['runs'] or len(faulty) != manifest['splits']['test_faulty']['runs']:
        raise ValueError('All frozen test runs must be scored')
    with torch.no_grad():
        clean_scores, _ = score(clean); faulty_scores, _ = score(faulty)
    prefix = module.PREFIXES
    arrays = dict(prefixes=np.array(prefix), kinds=np.asarray(kinds),
                  clean_sample_ids=np.arange(len(clean)), faulty_sample_ids=np.arange(len(faulty)))
    for rule in clean_scores:
        arrays['clean_'+rule] = clean_scores[rule]; arrays['faulty_'+rule] = faulty_scores[rule]
    score_path = directory/f'{tag}_TEST_scores.npz'; temporary = score_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(score_path)
    sync_directory(directory)
    record = dict(provenance, status='completed', battle='B3', tag=args.tag, trained_tag=tag, identity=identity,
                  validation_reconstruction=dict(metric='maximum_validation_auroc_error' if kind == 'classical' else 'clean_nll',
                                                 value=validation), prefixes=prefix,
                  primary_rule='declared detectors' if kind == 'classical' else 'total',
                  per_run_scores=dict(path=str(score_path.relative_to(ROOT)),sha256=sha(score_path)),
                  wall_s=time.monotonic()-start)
    result_path = directory/f'{tag}_TEST.json'; atomic_json(result_path, record)
    complete(reservation, result_path, score_path)
    print(json.dumps(dict(status='completed',tag=args.tag,trained_tag=tag,wall_s=record['wall_s'])), flush=True)


if __name__ == '__main__': main()
