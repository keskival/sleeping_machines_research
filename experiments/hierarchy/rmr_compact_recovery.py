"""Validate zero-write elimination, then score the completed K8 checkpoint.

No optimizer steps. Historical model code is reconstructed from the immutable
training admission; the compact execution law is the only substitution.
"""
import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    import pytest
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--mode', choices=['gate', 'recover'], required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    files = ['experiments/hierarchy/rmr_compact_recovery.py',
             'experiments/r1_rmr_compact.py', 'tests/test_rmr_compact.py',
             'experiments/r1_rmr.py', 'experiments/r1_cerl.py',
             'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py']
    sources = {name: sha((ROOT / name).read_bytes()) for name in files}
    assert pytest.main(['-q', 'tests/test_rmr_compact.py']) == 0
    result = dict(status='completed', battle='R1', gate='PASS',
                  stage='compact residual-memory numerical gate',
                  source_sha256=sources)
    if a.mode == 'recover':
        import numpy as np
        packet = ROOT / 'experiments/queue/aws_model_improvement_repair_20261005T161000Z/addenda/000_aws_r1_curie_ordered_program_20261016_v1.json'
        admission = json.loads(packet.read_text())
        tag = 'aws_r1_rmr_dense_K8_4M_s0_20261016_v1'
        job = next(j for jobs in admission['slots'].values() for j in jobs if j['name'] == tag)
        loaded = {}
        # Keep the exact trained prediction law, including its historical CERL
        # dependency. Never relabel today's source as the old loaded source.
        for module_name, name in [('r1_cerl', 'experiments/r1_cerl.py'),
                                  ('experiments.r1_rmr', 'experiments/r1_rmr.py')]:
            code = subprocess.check_output(['git', 'show', '64de80ffa:' + name])
            assert sha(code) == job['source_sha256'][name], name
            module = types.ModuleType(module_name)
            module.__file__ = str(ROOT / name)
            sys.modules[module_name] = module
            exec(compile(code, module.__file__, 'exec'), module.__dict__)
            loaded[name] = sha(code)
        for name in ['experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py']:
            assert sources[name] == job['source_sha256'][name], name
            loaded[name] = sources[name]
        sys.modules.pop('experiments.r1_rmr_compact', None)
        compact_class = importlib.import_module('experiments.r1_rmr_compact').CompactRMR
        old = sys.modules['r1_cerl']
        path = ROOT / f'experiments/results/token_language/{tag}.resume.pt'
        raw = path.read_bytes()
        digest = sha(raw)
        saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        del raw
        settings = saved['args']
        assert settings['levels'] == 8 and settings['top_m'] == 0
        expected_steps = int(np.ceil((settings['train_tokens'] // settings['lane']) / settings['batch']))
        assert saved['step'] == expected_steps, 'evaluation recovery cannot refit unfinished training'
        weights = saved['model']
        cerl = {k: settings[k] for k in ['d', 'modes', 'layers', 'dk', 'units', 'topk', 'explore']}
        model = compact_class(50257, weights['base.class_of'], weights['base.class_words'],
                              cerl, levels=8, dk=settings['read_dk'], top_m=0,
                              chunk=settings['chunk'], read_topk=settings['read_topk'],
                              rezero=bool(settings['rezero']))
        model.load_state_dict(weights)
        training = dict(steps=saved['step'], wall_s=saved['wall'],
                        milestones=saved['milestones'], targets=saved['step']*settings['batch']*(settings['lane']-1))
        del saved, weights
        val = old.load_fineweb(ROOT / 'data/fineweb_gpt2/fineweb_val_000000.bin')
        # Emit the short flagship first so a long-slice interruption cannot hide it.
        windows = np.stack([np.asarray(val[q:q+1025]) for q in range(25165824, 25165824+8192, 1024)])
        flagship = old.lanes_nll(model, windows, 128, stats=True)
        print(json.dumps(dict(flagship=flagship)), flush=True)
        result.update(stage='completed checkpoint compact residual-memory evaluation',
                      parent_run=tag, parent_checkpoint_sha256=digest,
                      training_loaded_source_sha256=loaded,
                      training_admission_sha256=sha(packet.read_bytes()),
                      additional_training_targets=0, training=training,
                      flagship=flagship,
                      score_65528=old.lanes_nll(model, np.asarray(val[20971520:20971520+8*8192]).reshape(8,8192), 128, stats=True),
                      scope='Same completed weights and scored targets/history. Zero old-write matrices removed; no retraining, TEST, context reduction or new model-quality claim from numerical parity alone.')
    result.update(elapsed_s=time.monotonic()-started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    folder = 'diagnostics' if a.mode == 'gate' else 'token_language'
    out = ROOT / f'experiments/results/{folder}/{a.tag}.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
