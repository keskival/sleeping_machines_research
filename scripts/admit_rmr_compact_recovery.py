"""Admit numerical parity and evaluation-only recovery through the AWS gym."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/queue/aws_model_improvement_repair_20261005T161000Z'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.uname().nodename == 'ip-172-31-47-132'
    sources = ['experiments/hierarchy/rmr_compact_recovery.py',
               'experiments/r1_rmr_compact.py', 'tests/test_rmr_compact.py',
               'experiments/r1_rmr.py', 'experiments/r1_cerl.py',
               'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py']
    jobs = []
    gate = 'aws_r1_rmr_compact_gate_20261016_v1'
    for mode, tag in [('gate', gate), ('recover', 'aws_r1_rmr_dense_K8_4M_compact_eval_20261016_v1')]:
        purpose = dict(battle='R1', decision='Remove identically zero quadratic old-write state and recover a completed K8 evaluation without training again.',
                       stationarity='Existing fixed-weight development protocol. No online comparison or new fitting.',
                       fighter='Exact saved eight-level residual-memory race model; top_m=0 algebra only. Old-slot-writing variants use original execution.',
                       state_history='Original full 1024/8192-token scored windows and independent lane policy. No shortening, reset change, optimizer steps or TEST.',
                       mechanisms='Same temporal carrier, cause races, concept capacity, nonlinear refinement, keys/values and exploration-trained parameters. This carrier-backed member is not the event-local flagship.',
                       restrictions='Only zero old-slot write matrices/products eliminated. Gate requires prediction and all-parameter full-objective gradient parity, causal/chunk checks and old-write fallback.',
                       resources='One existing AWS slot, one thread, 3M KiB RSS, 12M KiB VMS, 8 GiB host reserve. Counts evaluation process cost and RSS; no energy or training-work win inferred.')
        queue = ROOT / f'experiments/queue/r1/{tag}/{tag}.txt'
        queue.parent.mkdir(parents=True, exist_ok=True)
        content = '# ' + json.dumps(purpose) + '\n' + f'{tag} experiments/hierarchy/rmr_compact_recovery.py --mode {mode} --tag {tag}\n'
        if queue.exists():
            assert queue.read_text() == content
        else:
            queue.write_text(content)
        folder = 'diagnostics' if mode == 'gate' else 'token_language'
        jobs.append(dict(name=tag, kind='r1', queue=str(queue.relative_to(ROOT)),
                         queue_sha256=sha(queue), source_sha256={n: sha(ROOT/n) for n in sources},
                         result=f'experiments/results/{folder}/{tag}.json',
                         requires=[] if mode == 'gate' else [dict(result=f'experiments/results/diagnostics/{gate}.json', stage='compact residual-memory numerical gate')],
                         timeout_s=1800 if mode == 'gate' else 14400, rss_kb=3000000, vms_kb=12000000))
    packet = dict(host=os.uname().nodename, battle='R1', purpose='Gated zero-write algebra and completed-checkpoint evaluation recovery. No fitting.',
                  parent_manifest_sha256=sha(PLAN/'manifest.json'), slots={'2': jobs})
    path = PLAN/'addenda/000_aws_r1_00000_compact_recovery_20261016_v1.json'
    if path.exists():
        assert json.loads(path.read_text()) == packet
    else:
        tmp = path.with_suffix('.writing')
        tmp.write_text(json.dumps(packet, indent=2)+'\n')
        tmp.replace(path)
    # Product order: data-scaled residual-memory depth, after the same algebra
    # gate. Existing source-bound trainers finish untouched.
    fit_sources = ['experiments/r1_rmr_compact_train.py', 'experiments/r1_rmr_compact.py',
                   'experiments/r1_rmr.py', 'experiments/r1_cerl.py',
                   'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py']
    slots = {}
    for slot, levels in [('6', 4), ('5', 8)]:
        old_tag = f'aws_r1_rmr_dense_K{levels}_16M_s0_20261016_v1'
        tag = f'aws_r1_rmr_compact_K{levels}_16M_s0_20261016_v2'
        original = ROOT/f'experiments/queue/r1/{old_tag}/{old_tag}.txt'
        content = original.read_text()
        content = '\n'.join(line for line in content.splitlines() if line.startswith('#')) + '\n'
        content += '# Execution repair: identical current-slot model without zero old-write matrices. Complete-objective gradient/prediction gate required. Source-bound active jobs unchanged.\n'
        content += f'{tag} experiments/r1_rmr_compact_train.py --tag {tag} --levels {levels} --top-m 0 --read-topk 0 --train-tokens 16777216 --seed 0\n'
        queue = ROOT/f'experiments/queue/r1/{tag}/{tag}.txt'
        queue.parent.mkdir(parents=True, exist_ok=True)
        if queue.exists():
            assert queue.read_text() == content
        else:
            queue.write_text(content)
        slots[slot] = [dict(name=tag, kind='r1', queue=str(queue.relative_to(ROOT)),
                            queue_sha256=sha(queue), source_sha256={n:sha(ROOT/n) for n in fit_sources},
                            result=f'experiments/results/token_language/{tag}.json',
                            requires=[dict(result=f'experiments/results/diagnostics/{gate}.json', stage='compact residual-memory numerical gate')],
                            timeout_s=86400, rss_kb=3000000, vms_kb=12000000)]
    packet = dict(host=os.uname().nodename, battle='R1', purpose='Product-ordered 16M residual-memory depth fits, identical models with zero-write storage removed. Saved dense references only; fixed-weight development protocol and existing restrictions retained.',
                  parent_manifest_sha256=sha(PLAN/'manifest.json'), slots=slots)
    path = PLAN/'addenda/000_aws_r1_00001_compact_depth16M_20261016_v2.json'
    if path.exists():
        assert json.loads(path.read_text()) == packet
    else:
        tmp = path.with_suffix('.writing'); tmp.write_text(json.dumps(packet, indent=2)+'\n'); tmp.replace(path)
    print('Compact parity gate, evaluation-only recovery and conditional product-ordered 16M fits admitted.')


if __name__ == '__main__':
    main()
