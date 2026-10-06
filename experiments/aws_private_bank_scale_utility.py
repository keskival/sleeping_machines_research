"""Guarded positive memory-time-scale diagnosis; no fitting."""
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
    fields = {mode: () for mode in ('1', '0.5', '0.25', '0.0625')}
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
    pins = packet['scale_source_sha256']

    def check_sources():
        for name, digest in pins.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    check_sources()
    out = ROOT / 'experiments/results/diagnostics' / (cli.tag + '.json')
    if out.exists():
        raise FileExistsError(out)
    import torch
    import importlib
    from aws_memory_time_scale import installed_time_scale
    kernel = importlib.import_module('sleeping_machines.sparse_counterfactual_layer')
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
        first_features = None
        versions = {name: value._version for name, value in model.named_parameters()}
        contracts = {}
        with torch.no_grad():
            x, y = features(model, data, args)
            partition_nll = float(model.readout.nll(x, y).mean())
            assert math.isclose(partition_nll, selected['dev_nll'], abs_tol=2e-6)
            for mode in ('1', '0.5', '0.25', '0.0625'):
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
                            erased = ()
                            for key in erased:
                                assert all(torch.count_nonzero(v).item() == 0 for v in state[key])
                            checked = True
                    with installed_time_scale(kernel, float(mode)):
                        x, state, _ = model(data[:, begin:begin+1], state, generator, args,
                                            deterministic=args.deterministic_eval)
                    if begin == 0:
                        if first_features is None: first_features = x.clone()
                        else: assert torch.equal(first_features, x), (tag, mode, 'cold-start parity')
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
        assert math.isclose(losses['1'], partition_nll, abs_tol=2e-6)
        assert versions == {name: value._version for name, value in model.named_parameters()}
        rows.append(dict(tag=tag, seed=args.seed, pool=args.pool, selected=selected,
                         dev_targets=y.numel(), erasure_nll=losses,
                         erasure_increase={k: v-losses['1'] for k, v in losses.items() if k != '1'},
                         matched_rng=True, partition_parity=True, cold_start_parity=True, parameters_unmutated=True, intervention_contracts=contracts,
                         checkpoint_sha256=lab.sha(checkpoint), fit_sha256=lab.sha(path)))
        print(json.dumps(rows[-1]), flush=True)
    check_sources()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(status='completed', rows=rows, source_sha256=pins,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen selected DEV checkpoints, positive memory-time scales1,0.5,0.25,0.0625. Scale multiplies selected-unit decay rates and rotation frequencies, equivalent to scaling memory age; race delays, message transport, sparse writes and payload/key separation retained. Later routes can change. Intact, cold-start, matched-RNG, target and parameter nonmutation checks required. No retraining or public-reference claim.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
