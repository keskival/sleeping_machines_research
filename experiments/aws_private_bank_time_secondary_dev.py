"""Guarded secondary DEV scoring of frozen, previously selected checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--tag', required=True)
    cli = p.parse_args()
    packet = json.loads(Path(cli.manifest).read_text())
    pins = packet['secondary_source_sha256']
    def check():
        for name, digest in pins.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    check()
    out = ROOT / 'experiments/results/diagnostics' / (cli.tag + '.json')
    if out.exists(): raise FileExistsError(out)
    import torch
    import horizon_token_language_engine as lab
    from aws_private_bank_time_model import TimeScaledPrivateBankModel
    torch.set_num_threads(1)
    rows = []
    populations = set()
    folder = ROOT / 'experiments/results/token_language'
    interval = packet['secondary_interval']
    for tag in packet['fit_tags']:
        path = folder / (tag + '.json')
        fit = json.loads(path.read_text())
        assert fit['status'] == 'completed' and fit['source_sha256'] == packet['fit_source_sha256']
        args = SimpleNamespace(**fit['args'])
        assert interval['offset'] >= 10485760
        assert interval['tokens'] % args.eval_lanes == 0
        assert interval['offset'] >= args.dev_offset + args.dev_tokens
        for key, name in (('train_sha256', args.train_file), ('dev_sha256', args.dev_file)):
            assert lab.sha(ROOT / name) == fit['identity'][key]
        train, dev = lab.load_tokens(args.train_file), lab.load_tokens(args.dev_file)
        counts = lab.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
        torch.manual_seed(args.seed)
        model = TimeScaledPrivateBankModel(args, torch.argsort(counts, descending=True, stable=True),
                                          counts, fit['memory_time_scale'])
        selected = json.loads((folder / (tag + '.selection.json')).read_text())['selected']
        checkpoint = Path(selected['checkpoint'])
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        primary = lab.interval_tensor(dev, args.dev_offset, args.dev_tokens, args.eval_lanes)
        primary_nll = lab.evaluate(model, primary, args)
        assert math.isclose(primary_nll, selected['dev_nll'], rel_tol=0., abs_tol=2e-6)
        secondary = lab.interval_tensor(dev, interval['offset'], interval['tokens'], args.eval_lanes)
        secondary_nll = lab.evaluate(model, secondary, args)
        assert math.isfinite(secondary_nll)
        targets = secondary.numel() - args.eval_lanes
        populations.add((fit['identity']['dev_sha256'], interval['offset'], interval['tokens'], args.eval_lanes))
        rows.append(dict(tag=tag, seed=args.seed, memory_time_scale=fit['memory_time_scale'],
            primary_selected=selected, primary_parity=True, secondary_dev_nll=secondary_nll,
            secondary_dev_targets=targets, secondary_interval=interval,
            checkpoint_sha256=lab.sha(checkpoint), fit_sha256=lab.sha(path)))
        print(json.dumps(rows[-1]), flush=True)
    assert len(populations) == 1
    check()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(status='completed', rows=rows, source_sha256=pins,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Secondary development interval; checkpoints frozen from primary DEV selection, no refitting or reselection. Same corpus hash/lane/chunk/EOS/routing protocol across all arms. Fresh state per scored interval. Not official public validation or an independent training-seed confirmation; no matched-compute claim.'), indent=2) + '\n')


if __name__ == '__main__': main()
