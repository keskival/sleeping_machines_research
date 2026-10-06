"""Frozen causal DEV message-gain interventions; no fitting or model substitution."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import horizon_token_language_engine as lab


def main():
    started = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--tags', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    rows = []
    sources = {}
    for tag in a.tags:
        base = ROOT/'experiments/results/token_language'
        parent_path = base/(tag+'.json')
        parent = json.loads(parent_path.read_text())
        assert parent['status'] == 'completed'
        for name, digest in parent['identity']['source_sha256'].items():
            assert lab.sha(ROOT/name) == digest, name
        selected = json.loads((base/(tag+'.selection.json')).read_text())['selected']
        args = SimpleNamespace(**parent['args'])
        train, dev = lab.load_tokens(args.train_file), lab.load_tokens(args.dev_file)
        assert lab.sha(args.train_file) == parent['identity']['train_sha256']
        assert lab.sha(args.dev_file) == parent['identity']['dev_sha256']
        counts = lab.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
        torch.manual_seed(args.seed)
        model = lab.Model(args, torch.argsort(counts, descending=True, stable=True), counts)
        checkpoint = Path(selected['checkpoint'])
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        data = lab.interval_tensor(dev, args.dev_offset, args.dev_tokens, args.eval_lanes)
        baseline = lab.evaluate(model, data, args)
        assert abs(baseline-selected['dev_nll']) < 2e-6
        interventions = []
        for gain in (1., .5, .25, 0.):
            stats = dict(count=0, sum=0., minimum=1., maximum=0., below_001=0, above_099=0)
            def hook(module, inputs, logits):
                probability = torch.sigmoid(logits)
                stats['count'] += probability.numel()
                stats['sum'] += float(probability.sum())
                stats['minimum'] = min(stats['minimum'], float(probability.min()))
                stats['maximum'] = max(stats['maximum'], float(probability.max()))
                stats['below_001'] += int((probability<.01).sum())
                stats['above_099'] += int((probability>.99).sum())
                if gain == 1.:
                    return logits
                if gain == 0.:
                    return torch.full_like(logits, -float('inf'))
                scaled = probability*gain
                return torch.log(scaled)-torch.log1p(-scaled)
            handle = model.core.source_gate.register_forward_hook(hook)
            try:
                nll = lab.evaluate(model, data, args)
            finally:
                handle.remove()
            assert math.isfinite(nll)
            if gain == 1.:
                assert nll == baseline
            n = stats.pop('count')
            stats['mean'] = stats.pop('sum')/n
            stats['fraction_below_001'] = stats.pop('below_001')/n
            stats['fraction_above_099'] = stats.pop('above_099')/n
            interventions.append(dict(message_gain=gain, dev_nll=nll, difference_from_intact=nll-baseline, observed_gate_coordinates=n, original_gate_stats=stats))
        rows.append(dict(tag=tag, selected=selected, checkpoint_sha256=lab.sha(checkpoint), targets=data.shape[0]*(data.shape[1]-1), interventions=interventions))
        sources[str(parent_path.relative_to(ROOT))] = lab.sha(parent_path)
    record = dict(status='completed', host=platform.node(), wall_s=time.perf_counter()-started, rows=rows, parent_sha256=sources, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Frozen selected-checkpoint DEV interventions, gain multiplies learned sigmoid message gate before transport residual mixing; gain1 identity contracts exactly. Same seed+100000 reset for each causal evaluator, same data/target/history partition. State trajectories change as intended. No fitting, threshold selection, public validation or retrained ablation. Gate statistics include calls at no-context/EOS events, not only delivered messages. Original per-event evaluation kernel used for all checkpoints; gathered inference forward is contract-equivalent.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
