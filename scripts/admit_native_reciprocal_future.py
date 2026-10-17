"""R1 numerical and acquired-model gates for native future-directed learning."""
import hashlib
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v3')
    args = parser.parse_args()
    assert args.version.startswith('v') and args.version[1:].isdigit()
    assert os.uname().nodename == 'ip-172-31-47-132'
    sources = [str(p.relative_to(ROOT)) for p in (ROOT/'sleeping_machines').glob('*.py')]
    sources += ['experiments/credit/r1_native_reciprocal_future.py',
                'tests/test_deferred_episodic_writer_credit.py', 'tests/test_native_future_update_credit.py',
                'tests/test_native_query_migration.py', 'tests/test_native_indexed_cause_head.py',
                'experiments/native_query_migration.py', 'experiments/r1_composition_credit_pair.py',
                'experiments/r1_local_native_deep.py', 'experiments/r1_cerl.py']
    declaration = dict(battle='R1', stationarity='Chronological TRAIN integration; no DEV/TEST selection. Current source-bound fits finish unchanged.',
        fighter='Acquired fully native eight-layer local learner with persistent indexed causes, joint memory queries, full-depth sampled producer replay and deferred episodic writer credit; a separate eight-layer native race credit machine ranks addressed updates.',
        online='Every forecast is scored before its outcome changes weights/state used for that forecast. Action selection precedes its continuation; actual and audit branches perform their own eligible ordinary AdamW updates.',
        state_history='Reuse the acquired checkpoint, Adam moments, numerical memory and RNGs. Branches share exactly the same starting learner state; their subsequent states/learning evolve under their actions. Commit the issued branch regardless of audit outcome. Retain final actor and credit learner for reuse.',
        retained='Native temporal races, receiver eligibility, independent clocks, persistent versioned memories, indexed episodic causes, separate keys/values, legal joint queries, counterfactual alternatives, sampled lower-producer content credit and future-valued learned update selection. No dense carrier or recurrent BPTT.',
        restrictions='Bounded numerical gate, not competition. Two episodes; four already-revealed support targets then eight future targets. Initial action interface is a single addressed mix-block step with both signs and no-op. One declared precommitted bootstrap audit, then native policy with positive support. No whole-history exact meta-gradient or arbitrary-action coverage claim.',
        accounting='One existing gym slot, one thread, 3M KiB RSS/12M KiB VMS, 8 GiB available-memory reserve. Count 24 actual TRAIN targets plus 16 additional counterfactual targets, critic training, replay/snapshot counts, RSS and elapsed time. No hardware, full-work or iso-compute win.',
        decision='Do future-directed reciprocal learning and delayed episodic producer credit compose correctly in the acquired native learner, preserve causal action selection/forecast parity, deliver finite producer/critic updates and resume exactly? Numerical pass does not establish future prediction benefit.')
    contract = f'aws_r1_native_reciprocal_contract_20261016_{args.version}'
    jobs = []
    for mode, tag in [('contract', contract), ('acquired', f'aws_r1_native_reciprocal_acquired_20261016_{args.version}')]:
        queue = ROOT/f'experiments/queue/r1/{tag}/{tag}.txt'
        queue.parent.mkdir(parents=True, exist_ok=True)
        text = '# '+json.dumps(declaration)+'\n'+f'{tag} experiments/credit/r1_native_reciprocal_future.py --mode {mode} --tag {tag}\n'
        if queue.exists():
            assert queue.read_text() == text
        else:
            queue.write_text(text)
        jobs.append(dict(name=tag, kind='r1', queue=str(queue.relative_to(ROOT)), queue_sha256=sha(queue),
                         source_sha256={n:sha(ROOT/n) for n in sorted(set(sources))},
                         result=f'experiments/results/diagnostics/{tag}.json',
                         requires=[] if mode=='contract' else [dict(result=f'experiments/results/diagnostics/{contract}.json', stage='native reciprocal future credit numerical gate')],
                         timeout_s=1800 if mode=='contract' else 7200, rss_kb=3000000, vms_kb=12000000))
    acquired_tag=f'aws_r1_native_reciprocal_acquired_20261016_{args.version}'
    parent=f'experiments/results/diagnostics/{acquired_tag}.json'
    tag=f'aws_r1_native_reciprocal_online_1152_20261016_{args.version}'
    prospective=dict(declaration,
        restrictions='Prospective one-parent, one-seed TRAIN development comparison: 32 episodes, four revealed support targets and 32 future targets each. Addressed mix-block update actions with both signs and no-op; all other actor parameters still adapt ordinarily. Native credit machine and actor retain numerical state. No TEST or hindsight rollback.',
        accounting='1152 actual trajectory targets,1024 additional same-state no-action audit targets,1152 targets for the continuously adapting complete-native reference and1152 for an immediate-loss-greedy native policy. All credit learning, snapshots, replay and checkpoint costs paid. One existing gym slot,one thread,3M KiB RSS/12M KiB VMS,8 GiB host reserve. Atomic joint resume package binds all three actors after every completed episode.',
        decision='Does the integrated future-valued native update policy improve chronological prequential NLL against ordinary causal adaptation and an immediate-loss-greedy action policy in the same complete native model? The greedy policy takes the largest-gradient-norm addressed descent step under the same RMS action budget. Do audited actions improve subsequent loss? Report prediction and calibration separately. Positive developmental result is not a public or matched-work win.')
    queue=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';queue.parent.mkdir(parents=True,exist_ok=True)
    text='# '+json.dumps(prospective)+'\n'+f'{tag} experiments/credit/r1_native_reciprocal_future.py --mode continue --parent {parent} --tag {tag}\n'
    if queue.exists():assert queue.read_text()==text
    else:queue.write_text(text)
    jobs.append(dict(name=tag,kind='r1',queue=str(queue.relative_to(ROOT)),queue_sha256=sha(queue),
                     source_sha256={n:sha(ROOT/n) for n in sorted(set(sources))},
                     result=f'experiments/results/diagnostics/{tag}.json',
                     requires=[dict(result=parent,stage='acquired native reciprocal future credit integration gate')],
                     timeout_s=14400,rss_kb=3000000,vms_kb=12000000))
    packet = dict(host=os.uname().nodename, battle='R1', purpose=declaration,
                  parent_manifest_sha256=sha(PLAN/'manifest.json'), slots={'3':jobs})
    path = PLAN/f'addenda/000_aws_r1_00000_reciprocal_future_20261016_{args.version}.json'
    if path.exists():
        assert json.loads(path.read_text()) == packet
    else:
        tmp = path.with_suffix('.writing'); tmp.write_text(json.dumps(packet, indent=2)+'\n'); tmp.replace(path)
    print('Numerical gate, acquired-native integration and conditional prospective 1152-target comparison admitted in slot 3.')


if __name__ == '__main__':
    main()
