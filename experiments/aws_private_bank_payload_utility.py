"""Guarded selected-checkpoint payload/metadata interventions; no fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))


def intervene(state, mode, zeros_like):
    """Copy the container; preserve every unselected field by identity."""
    fields = {'intact': (), 'payload': ('mem',),
              'metadata': ('arr', 'seen'), 'memory': ('mem', 'arr', 'seen')}
    selected = fields[mode]
    result = dict(state)
    for key in selected:
        result[key] = [zeros_like(value) for value in state[key]]
    for key in state:
        if key not in selected:
            assert result[key] is state[key], ('Changed unselected field', key)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--tag', required=True)
    cli = parser.parse_args()
    packet = json.loads(Path(cli.manifest).read_text())
    pins = packet['payload_source_sha256']

    def check_sources():
        for name, digest in pins.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    check_sources()
    out = ROOT / 'experiments/results/diagnostics' / (cli.tag + '.json')
    if out.exists():
        raise FileExistsError(out)
    import torch
    import horizon_token_language_engine as lab
    from aws_private_bank_model import PrivateBankModel
    from token_stage_utility import features
    torch.set_num_threads(1)
    rows = []
    folder = ROOT / 'experiments/results/token_language'
    for tag in packet['fit_tags']:
        path = folder / (tag + '.json')
        raw = json.loads(path.read_text())
        assert raw['status'] == 'completed', tag
        assert raw['source_sha256'] == packet['fit_source_sha256'], tag
        args = SimpleNamespace(**raw['args'])
        for key, name in [('train_sha256', args.train_file), ('dev_sha256', args.dev_file)]:
            assert lab.sha(ROOT / name) == raw['identity'][key], name
        train, dev = lab.load_tokens(args.train_file), lab.load_tokens(args.dev_file)
        counts = lab.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
        torch.manual_seed(args.seed)
        model = PrivateBankModel(args, torch.argsort(counts, descending=True, stable=True), counts)
        selected = json.loads((folder / (tag + '.selection.json')).read_text())['selected']
        checkpoint = Path(selected['checkpoint'])
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model.eval()
        data = lab.interval_tensor(dev, args.dev_offset, args.dev_tokens, args.eval_lanes)
        losses, ref_rng = {}, None
        contracts = {}
        with torch.no_grad():
            x, y = features(model, data, args)
            partition_nll = float(model.readout.nll(x, y).mean())
            assert math.isclose(partition_nll, selected['dev_nll'], abs_tol=2e-6)
            for mode in ('intact', 'payload', 'metadata', 'memory'):
                state = None
                generator = torch.Generator().manual_seed(args.seed + 100000)
                total, targets, checked = 0., 0, False
                for begin in range(data.shape[1] - 1):
                    if state is not None:
                        previous = state
                        snapshots = {key: [v.clone() for v in previous[key]]
                                     for key in ('mem', 'arr', 'seen')} if not checked else None
                        state = intervene(previous, mode, torch.zeros_like)
                        if not checked:
                            for key, values in snapshots.items():
                                assert all(torch.equal(a, b) for a, b in zip(values, previous[key]))
                            erased = {'intact': (), 'payload': ('mem',), 'metadata': ('arr', 'seen'),
                                      'memory': ('mem', 'arr', 'seen')}[mode]
                            for key in erased:
                                assert all(torch.count_nonzero(v).item() == 0 for v in state[key])
                            checked = True
                    x, state, _ = model(data[:, begin:begin+1], state, generator, args,
                                        deterministic=args.deterministic_eval)
                    loss = model.readout.nll(x, data[:, begin+1:begin+2])
                    total += float(loss.sum())
                    targets += loss.numel()
                assert checked and targets == y.numel()
                rng = generator.get_state()
                if ref_rng is None:
                    ref_rng = rng
                else:
                    assert torch.equal(ref_rng, rng), (tag, mode)
                losses[mode] = total / targets
                contracts[mode] = True
        assert math.isclose(losses['intact'], partition_nll, abs_tol=2e-6)
        rows.append(dict(tag=tag, seed=args.seed, pool=args.pool, selected=selected,
                         dev_targets=y.numel(), erasure_nll=losses,
                         erasure_increase={k: v-losses['intact'] for k, v in losses.items() if k != 'intact'},
                         matched_rng=True, partition_parity=True, intervention_contracts=contracts,
                         checkpoint_sha256=lab.sha(checkpoint), fit_sha256=lab.sha(path)))
        print(json.dumps(rows[-1]), flush=True)
    check_sources()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(status='completed', rows=rows, source_sha256=pins,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Selected DEV checkpoints; payload-only clears mem while preserving arrival/seen and message state; metadata-only clears arr/seen while preserving payload. Frozen interventions before each subsequent token, no retraining. Changed payload can subsequently change routing. All other state fields preserved by identity; original tensors unchanged. No pure-route causal attribution or public benchmark claim.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
