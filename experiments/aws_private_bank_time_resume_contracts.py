"""Guarded real-driver optimizer interruption/resume for positive time scales."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--tag', required=True)
    cli = p.parse_args()
    packet = json.loads(Path(cli.manifest).read_text())
    pins = packet['contract_source_sha256']
    def check():
        for name, digest in pins.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    check()
    out = ROOT / 'experiments/results/diagnostics' / (cli.tag + '.json')
    if out.exists(): raise FileExistsError(out)
    import torch
    import horizon_token_language_engine as engine
    from aws_private_bank_time_model import TimeScaledPrivateBankModel
    from aws_private_bank_work import assert_same
    torch.set_num_threads(1)
    folder = ROOT / 'experiments/results/token_language'
    base = ['--readout-init', 'frequency', '--input-init', 'balanced', '--steps', '6',
        '--eval-every', '3', '--train-tokens', '512', '--dev-tokens', '128',
        '--payload', '24', '--heads', '2', '--depth', '2', '--pool', '4', '--tie-pools',
        '--lanes', '2', '--eval-lanes', '2', '--chunk', '8', '--credit-window', '16',
        '--future-every', '1', '--future-site', 'uniform', '--future-score-positions', '4',
        '--minimum-tail-width', '0', '--seed', '6']
    for label, value in packet['resume_data'].items():
        base += ['--' + label.replace('_', '-'), value]
    rows = []
    parent, original_argv = engine.Model, sys.argv
    try:
        for scale in (1., .25, .0625):
            class ResumeModel(TimeScaledPrivateBankModel):
                def __init__(self, args, order, counts):
                    super().__init__(args, order, counts, scale)
            engine.Model = ResumeModel
            stem = cli.tag + '_scale' + str(scale).replace('.', 'p')
            tags = [stem + '_' + label for label in ('full', 'partial', 'restored')]
            for tag in tags:
                if any(folder.glob(tag + '.*')): raise FileExistsError(tag)
            for tag, extra in zip(tags, ([], ['--stop-after-step', '3'],
                    ['--resume', str(folder / (tags[1] + '.pt'))])):
                sys.argv = ['horizon_token_language_engine.py', '--tag', tag, *base, *extra]
                engine.main()
                gc.collect()
            full = torch.load(folder / (tags[0] + '.pt'), weights_only=False)
            restored = torch.load(folder / (tags[2] + '.pt'), weights_only=False)
            fields = ('model', 'optimizer', 'state', 'cursor', 'step', 'site_generator',
                'position_generator', 'local_generator', 'alternative_generator',
                'generator', 'torch_rng', 'writes', 'total_presentations')
            for field in fields: assert_same(full[field], restored[field], field)
            assert full['total_presentations'] == restored['total_presentations'] == 96
            assert len(full['curve']) == len(restored['curve'])
            for a, b in zip(full['curve'], restored['curve']):
                for field in ('step', 'train_nll', 'dev_nll'):
                    assert_same(a[field], b[field], 'curve.' + field)
            rows.append(dict(scale=scale, exact_fields=list(fields), fitting_targets=96,
                full_tag=tags[0], resumed_tag=tags[2], quality_curve_exact=True))
    finally:
        engine.Model, sys.argv = parent, original_argv
    check()
    record = dict(status='completed', optimizer_continuation_exact=True, rows=rows,
        source_sha256=pins, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Actual eager uniform-site K4 counterfactual training driver, six updates interrupted after three, scales1,1/4,1/16. Exact optimizer/model/persistent-state/cursor/all teacher-route RNG and curve continuation; synthetic-size real token interval, no benchmark quality claim.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
