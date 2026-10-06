"""Selected-checkpoint utility and promotion audit for a completed token stage.

All target scores are development-only. Run through a unique run_safe queue.
"""
import argparse
import importlib
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import split_horizon_token_language_engine as lab
from token_promotion_gate import assess

@torch.no_grad()
def features(model, data, args):
    state = None
    generator = torch.Generator().manual_seed(args.seed + 100000)
    xs, ys = [], []
    for begin in range(0, data.shape[1] - 1, args.chunk):
        n = min(args.chunk, data.shape[1] - 1 - begin)
        x, state, _ = model(data[:, begin:begin+n], state, generator, args,
                             deterministic=args.deterministic_eval)
        xs.append(x.reshape(-1, x.shape[-1]))
        ys.append(data[:, begin+1:begin+n+1].reshape(-1))
    return torch.cat(xs), torch.cat(ys)

@torch.no_grad()
def erase_score(model, data, args, mode):
    state = None
    generator = torch.Generator().manual_seed(args.seed + 100000)
    total = 0.
    targets = 0
    for begin in range(data.shape[1] - 1):
        if state is not None:
            state = dict(state)
            if mode in ('memory', 'both'):
                for key in ('mem', 'arr', 'seen'):
                    state[key] = [torch.zeros_like(v) for v in state[key]]
            if mode in ('message', 'both'):
                for key in ('ctx_vals', 'ctx_arr', 'has_ctx'):
                    state[key] = torch.zeros_like(state[key])
        x, state, _ = model(data[:, begin:begin+1], state, generator, args,
                             deterministic=args.deterministic_eval)
        loss = model.readout.nll(x, data[:, begin+1:begin+2])
        total += float(loss.sum())
        targets += loss.numel()
    return total / targets, generator.get_state(), targets

@torch.no_grad()
def streamed_mean_features(model, data, args):
    state = None
    generator = torch.Generator().manual_seed(args.seed + 100000)
    total = None
    count = 0
    for begin in range(0, data.shape[1] - 1, args.chunk):
        n = min(args.chunk, data.shape[1] - 1 - begin)
        x, state, _ = model(data[:, begin:begin+n], state, generator, args,
                             deterministic=args.deterministic_eval)
        flat = x.reshape(-1, x.shape[-1])
        block = flat.double().sum(0)
        total = block if total is None else total + block
        count += flat.shape[0]
    assert count == data.shape[0]*(data.shape[1]-1)
    return (total/count).to(x.dtype), count

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tags', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mean-contract', action='store_true')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(1)
    rows = []
    folder = ROOT / 'experiments/results/token_language'
    for tag in a.tags:
        result = json.loads((folder / (tag + '.json')).read_text())
        if result['status'] != 'completed': raise ValueError(tag)
        for file, digest in result['identity']['source_sha256'].items():
            assert lab.sha(ROOT / file) == digest, ('Source mismatch', file)
        selection = json.loads((folder / (tag + '.selection.json')).read_text())['selected']
        args = SimpleNamespace(**result['args'])
        train, dev = lab.load_tokens(args.train_file), lab.load_tokens(args.dev_file)
        counts = lab.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
        torch.manual_seed(args.seed)
        if 'future_credit_window' not in result['args']:
            raise ValueError('Separated-horizon control required')
        backend = lab
        model = backend.Model(args, torch.argsort(counts, descending=True, stable=True), counts)
        checkpoint = Path(selection['checkpoint'])
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model.eval()
        train_data = lab.interval_tensor(train, 0, args.train_tokens, args.lanes)
        dev_data = lab.interval_tensor(dev, args.dev_offset, args.dev_tokens, args.eval_lanes)
        train_mean, train_observations = streamed_mean_features(model, train_data, args)
        mean_contract = None
        if a.mean_contract:
            train_x, _ = features(model, train_data, args)
            reference_mean = train_x.mean(0)
            error = float((reference_mean-train_mean).abs().max())
            assert error < 1e-6
            assert train_x.shape[0] == train_observations
            mean_contract = dict(max_feature_mean_error=error, count=train_observations)
            del train_x
        x, y = features(model, dev_data, args)
        with torch.no_grad():
            exact = float(model.readout.nll(x, y).mean())
            constant = float(model.readout.nll(train_mean.expand_as(x), y).mean())
        if mean_contract is not None:
            reference_constant = float(model.readout.nll(reference_mean.expand_as(x), y).mean())
            error = abs(reference_constant-constant)
            assert error < 2e-6
            mean_contract['constant_nll_max_error'] = error
        assert math.isclose(exact, selection['dev_nll'], abs_tol=2e-6)
        interventions = {}
        ref_rng = None
        for mode in ('intact', 'memory', 'message', 'both'):
            loss, rng, targets = erase_score(model, dev_data, args, mode)
            assert targets == y.numel()
            if ref_rng is None: ref_rng = rng
            else: assert torch.equal(rng, ref_rng), ('RNG mismatch', tag, mode)
            interventions[mode] = loss
        assert math.isclose(interventions['intact'], exact, abs_tol=2e-6)
        row = dict(tag=tag, seed=args.seed, credit_window=args.credit_window,
                   minimum_tail_width=args.minimum_tail_width,
                   unit_gain_scale=getattr(args, "unit_gain_scale", 1.),
                   train_tokens=args.train_tokens, fit_targets=result['presentations_total'],
                   dev_targets=y.numel(), selected=selection, promotion=assess(result),
                   context_nll=exact, train_mean_nll=constant, context_gain=constant-exact,
                   erasure_nll=interventions,
                   history_gains={k:v-exact for k,v in interventions.items() if k != 'intact'},
                   partition_parity=True, matched_rng=True,
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   train_feature_observations=train_observations, mean_contract=mean_contract)
        rows.append(row)
        print(json.dumps(row), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(status='completed', rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Frozen development interventions on initial-inclusive selected checkpoints, exact causal model features and streaming train-only constant-feature mean (float64 sums, cast to feature dtype). State erased before each subsequent token; same route RNG and token partition parity. Out-of-distribution erasure, not retrained ablation. Mean-feature control not an optimally refitted unigram. No fitting, public validation or full hard-alternative credit measurement.'), indent=2)+'\n')

if __name__ == '__main__': main()
