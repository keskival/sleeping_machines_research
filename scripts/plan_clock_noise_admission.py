"""Prepare frozen one-job native contracts then bounded tau19 DEV integration.

No execution or numerical imports; existing NeuroBench owners retain priority.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from plan_public_speech_admission import local_dependencies
from experiments.clock_noise_admission import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp', required=True)
    args = parser.parse_args()
    if not args.stamp.isalnum():
        raise ValueError('Unique alphanumeric stamp required')
    prefix = 'native_clock_noise_' + args.stamp
    directory = ROOT / 'experiments/queue' / prefix
    directory.mkdir()
    sources = local_dependencies(['experiments/clock_noise_admission.py',
        'sleeping_machines/clock_noise_law.py', 'sleeping_machines/clock_noise_episodes.py',
        'scripts/plan_clock_noise_admission.py', 'tests/test_clock_noise_law_stdlib.py'])
    for name in ['experiments/queue/run_safe.sh',
                 'experiments/theory/151_normalized_clock_noise_and_precision_credit.md',
                 'experiments/public_benchmarks/neurobench_mg_data_manifest.json']:
        sources[name] = sha(ROOT / name)
    stages = {}
    for name in ('contracts', 'pilot'):
        tag = prefix + '_' + name
        stages[name] = dict(tag=tag, temperatures=[1., .5, 0.], rss_cap_kb=2_000_000,
            vms_cap_kb=6_000_000, min_available_mb=8192, timeout_s=900,
            queue=str((directory / (tag + '.txt')).relative_to(ROOT)),
            output='experiments/results/clock_noise_admission/' + tag + '.json')
    stages['pilot'].update(seed=6, steps=32, lanes=8, segment=64, repeat=4)
    data_manifest = json.loads((ROOT / 'experiments/public_benchmarks/neurobench_mg_data_manifest.json').read_text())
    data = next(x for x in data_manifest['files'] if Path(x['path']).name == 'mg_19.npy')
    record = dict(status='prepared_unrun', source_sha256=sources, stages=stages,
                  data=dict(path=data['path'], sha256=data['sha256'], tau=19),
                  official_tau17_read=False, numerical_scores=None,
                  admission='Physical run_safe reservation only; no container waiter or host-lock bypass.',
                  priority='Existing AWS six-session primate and official MG, curie MG/primate/SHD and active replay owners first.')
    path = directory / 'manifest.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    digest = sha(path)
    for name, cfg in stages.items():
        command = cfg['tag'] + ' experiments/clock_noise_admission.py --manifest ' + str(path.relative_to(ROOT))
        command += ' --manifest-sha256 ' + digest + ' --stage ' + name
        (ROOT / cfg['queue']).write_text('# UNRUN: physical owner after prioritized queues; one job, one CPU thread.\n'
            '# MEM_CAP_KB=6000000 MEM_CAP_RSS_KB=2000000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=900\n' + command + '\n')
    print(json.dumps(dict(status='prepared_unrun', manifest=str(path.relative_to(ROOT)),
                         manifest_sha256=digest, sources=len(sources), jobs=2)))


if __name__ == '__main__':
    main()
