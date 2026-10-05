"""Selected-model memory diagnostics, admitted only through run_safe.sh.

The observer calls the unchanged selected-value primitive and records its
analytic rotation/decay factors. It changes no model parameters or outputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def validate_inputs(records):
    if len(records) != 4:
        raise ValueError('Require complete two-seed/two-horizon comparison')
    cells = set()
    shared = None
    for record in records:
        if record.get('status') != 'completed':
            raise ValueError('Completed fits required')
        args = record['args']
        cell = (args['seed'], args['credit_window'])
        if cell in cells:
            raise ValueError('Duplicate seed/horizon')
        cells.add(cell)
        if args['state_mode'] != 'carry' or args['compiled']:
            raise ValueError('Observer requires eager persistent inference')
        if args['dev_offset'] < 10485760:
            raise ValueError('Public validation is reserved')
        settings = {k: v for k, v in args.items() if k not in
                    ('tag', 'seed', 'credit_window', 'resume', 'stop_after_step')}
        identity = (settings, record['identity']['train_sha256'],
                    record['identity']['dev_sha256'], record['presentations_total'])
        if shared is None:
            shared = identity
        elif shared != identity:
            raise ValueError('Unmatched data/settings/presentations')
    if cells != {(s, w) for s in (6, 7) for w in (16, 64)}:
        raise ValueError('Require seeds6/7 and horizons16/64')


def primitive_contract(torch):
    from sleeping_machines.sparse_training import _unit
    g = torch.Generator().manual_seed(1912)
    rand = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)
    n, H, U, P = 2, 2, 3, 4
    sel = torch.tensor([[0, 2], [1, 0]])
    memory = rand(n, H, U, P).requires_grad_(True)
    stamps = torch.zeros(n, H, U, dtype=torch.float64)
    seen = torch.ones(n, H, U, dtype=torch.bool)
    arrival = torch.tensor([1., 3.], dtype=torch.float64)
    incoming = rand(n, H, P)
    tail = (stamps, seen, arrival, incoming, U, P, rand(H*U, 2, P),
            rand(H*U, 2), torch.full((H*U, P//2), .05, dtype=torch.float64),
            rand(H*U, P//2), rand(H*U, P, P), rand(H*U, P, P),
            rand(H*U, P, P), rand(H*U, P), .3)
    cotangent = rand(n, H, P)
    value = _unit(sel, memory, *tail)[0]
    gradient = torch.autograd.grad((value*cotangent).sum(), memory)[0]
    direction = rand(n, H, U, P)
    epsilon = 1e-5
    plus = _unit(sel, memory.detach()+epsilon*direction, *tail)[0]
    minus = _unit(sel, memory.detach()-epsilon*direction, *tail)[0]
    finite = float(((plus-minus)*cotangent).sum()/(2*epsilon))
    analytic = float((gradient*direction).sum())
    if not math.isclose(finite, analytic, rel_tol=2e-6, abs_tol=2e-8):
        raise ValueError('Memory-value derivative failed finite difference')
    chosen = torch.nn.functional.one_hot(sel, U).bool()[..., None].expand_as(memory)
    if not torch.equal(gradient[~chosen], torch.zeros_like(gradient[~chosen])):
        raise ValueError('Unselected value path touched another memory')
    if not gradient[chosen].abs().sum() > 0:
        raise ValueError('Selected memory value path is dead')
    query = rand(n, H, P) * .1
    key_read = rand(H, U, P, P) * .1
    key = rand(H, U, P) * .1
    def key_scores(m):
        read = key + torch.einsum('hupq,nhuq->nhup', key_read, m)
        return ((query[:, :, None, :] * read).sum(-1) / math.sqrt(P)).clamp(-12, 12)
    score_gradient = torch.autograd.grad(key_scores(memory).sum(), memory)[0]
    key_analytic = float((score_gradient*direction).sum())
    key_finite = float((key_scores(memory.detach()+epsilon*direction).sum()
                        - key_scores(memory.detach()-epsilon*direction).sum())/(2*epsilon))
    if not math.isclose(key_analytic, key_finite, rel_tol=2e-6, abs_tol=2e-8):
        raise ValueError('Stored-memory score derivative failed finite difference')
    return dict(status='passed', analytic=analytic, finite_difference=finite,
                key_score_analytic=key_analytic, key_score_finite_difference=key_finite,
                unselected_gradient_zero=True, selected_gradient_nonzero=True,
                scope='Synthetic double-precision selected-value primitive; fixed receiver and clocks')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    packet = json.loads(Path(a.manifest).read_text())
    pins = packet['audit_source_sha256']
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Source mismatch: ' + name)
    output = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if output.exists():
        raise FileExistsError(output)
    records = [json.loads((ROOT / name).read_text()) for name in packet['audit_inputs']]
    validate_inputs(records)
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
    import torch
    from torch.nn import functional as F
    import horizon_token_language_engine as lab
    import sleeping_machines.sparse_counterfactual_layer as layer
    from sleeping_machines.sparse_training import _rotate
    torch.set_num_threads(1)
    started = time.monotonic()
    contract = primitive_contract(torch)
    rows = []

    def summary(values):
        if not values:
            return dict(count=0)
        t = torch.cat(values).double()
        return dict(count=t.numel(), mean=float(t.mean()), min=float(t.min()),
                    median=float(t.median()), max=float(t.max()))

    @torch.no_grad()
    def score(model, data, args, mode, chunk):
        state = None
        gen = torch.Generator().manual_seed(args.seed + 100000)
        total = 0.; targets = 0
        for start in range(0, data.shape[1] - 1, chunk):
            n = min(chunk, data.shape[1] - 1 - start)
            if state is not None:
                state = dict(state)
                keys = []
                if mode in ('no_memory', 'no_history'):
                    keys += ['mem', 'arr', 'seen']
                if mode in ('no_message', 'no_history'):
                    keys += ['ctx_vals', 'ctx_arr', 'has_ctx']
                for key in keys:
                    value = state[key]
                    state[key] = ([torch.zeros_like(v) for v in value] if isinstance(value, list)
                                  else torch.zeros_like(value))
            x, state, _ = model(data[:, start:start+n], state, gen, args,
                                deterministic=args.deterministic_eval)
            loss = model.readout.nll(x, data[:, start+1:start+n+1])
            total += float(loss.sum()); targets += loss.numel()
        return total / targets, gen.get_state(), targets

    for record in records:
        args = SimpleNamespace(**record['args'])
        for path, digest in ((args.train_file, record['identity']['train_sha256']),
                             (args.dev_file, record['identity']['dev_sha256'])):
            if sha(ROOT / path) != digest:
                raise ValueError('Data mismatch: ' + path)
        for name, digest in record['identity']['source_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Fit source mismatch: ' + name)
        selected = record['selection']['selected']
        checkpoint = Path(selected['checkpoint'])
        if not checkpoint.is_absolute(): checkpoint = ROOT / checkpoint
        train = lab.load_tokens(ROOT / args.train_file)
        counts = lab.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
        torch.manual_seed(args.seed)
        model = lab.Model(args, torch.argsort(counts, descending=True, stable=True), counts)
        model.load_state_dict(torch.load(checkpoint, weights_only=True)); model.eval()
        dev = lab.load_tokens(ROOT / args.dev_file)
        data = lab.interval_tensor(dev, args.dev_offset, args.dev_tokens, args.eval_lanes)
        original, rng, targets = score(model, data, args, 'intact', args.chunk)
        if not math.isclose(original, selected['dev_nll'], abs_tol=2e-6):
            raise ValueError('Selected quality did not reproduce')
        stats = {k: [] for k in ('selected_age', 'decay', 'retained_norm', 'write_norm',
                                 'retained_to_write', 'memory_value_vjp_norm',
                                 'memory_score_contribution', 'memory_score_vjp_norm')}
        primitive = layer._unit
        layers = model.core._stacked(0)
        unit_calls = 0
        probes = 0

        def observer(*inputs):
            nonlocal probes, unit_calls
            value, new_memory, idx = primitive(*inputs)
            sel, memory, stamps, seen, arrival, incoming, U, P, cw, cb, rate, freq, iw = inputs[:13]
            pick = sel[:, :, None, None].expand(*sel.shape, 1, P)
            old = torch.gather(memory, 2, pick).squeeze(2)
            used = torch.gather(seen, 2, sel[:, :, None]).squeeze(2)
            prev = torch.where(used, torch.gather(stamps, 2, sel[:, :, None]).squeeze(2), arrival[:, None])
            age = (arrival[:, None] - prev).clamp_min(0)
            H = sel.shape[1]
            bank = layers[unit_calls % model.core.depth]
            unit_calls += 1
            features = F.layer_norm(incoming.flatten(1), (H*P,))
            query = torch.einsum('hpd,nd->nhp', bank['query'], features)
            read_w = bank['key_read'].view(H, U, P, P)
            memory_read = torch.einsum('hupq,nhuq->nhup', read_w, memory)
            contribution = (query[:, :, None, :] * memory_read).sum(-1) / math.sqrt(P)
            raw_scores = contribution + (query[:, :, None, :] * bank['key'].view(H, U, P)).sum(-1) / math.sqrt(P) + bank['clock_bias'].view(H, U)
            live = (raw_scores > -12) & (raw_scores < 12)
            score_vjp = torch.einsum('nhp,hupq->nhuq', query, read_w) / math.sqrt(P)
            stats['memory_score_contribution'].append(contribution[seen].detach().flatten())
            stats['memory_score_vjp_norm'].append((score_vjp.norm(dim=-1)*live)[seen].detach().flatten())
            controls = torch.einsum('nhcp,nhp->nhc', cw.view(H*U, 2, P)[idx],
                                    F.layer_norm(incoming, (P,))) + cb.view(H*U, 2)[idx]
            forget = F.softplus(controls[..., 0]) / math.log(2)
            factors = torch.exp(-age.to(old.dtype)[..., None] * rate.view(H*U, P//2)[idx] * forget[..., None])
            retained = _rotate(old * factors.repeat_interleave(2, -1), age[..., None] * freq.view(H*U, P//2)[idx])
            write = new_memory - retained
            rnorm = retained.norm(dim=-1); wnorm = write.norm(dim=-1)
            for key, values in (('selected_age', age), ('decay', factors),
                                ('retained_norm', rnorm), ('write_norm', wnorm),
                                ('retained_to_write', rnorm / wnorm.clamp_min(1e-12))):
                mask = used[..., None].expand_as(values) if values.ndim == 3 else used
                stats[key].append(values[mask].detach().flatten())
            if probes < 8 and bool(used.any()):
                with torch.enable_grad():
                    differentiable = memory.detach().clone().requires_grad_(True)
                    changed = list(inputs); changed[1] = differentiable
                    v = primitive(*changed)[0]
                    cotangent = torch.arange(1, P+1, dtype=v.dtype).view(1, 1, P) / P
                    gradient = torch.autograd.grad((v * cotangent * used[..., None]).sum(), differentiable)[0]
                    selected_gradient = torch.gather(gradient, 2, pick).squeeze(2)
                    stats['memory_value_vjp_norm'].append(selected_gradient.norm(dim=-1)[used].detach().flatten())
                probes += 1
            return value, new_memory, idx

        layer._unit = observer
        try:
            observed, observed_rng, observed_targets = score(model, data, args, 'intact', 1)
        finally:
            layer._unit = primitive
        if not math.isclose(observed, original, abs_tol=2e-6) or not torch.equal(observed_rng, rng) or observed_targets != targets:
            raise ValueError('Observer/partition altered predictions or RNG')
        losses = {'intact': observed}
        for mode in ('no_memory', 'no_message', 'no_history'):
            losses[mode], intervention_rng, count = score(model, data, args, mode, 1)
            if not torch.equal(intervention_rng, rng) or count != targets:
                raise ValueError('Intervention RNG/target mismatch')
        row = dict(tag=args.tag, seed=args.seed, credit_window=args.credit_window,
                   targets=targets, checkpoint_sha256=sha(checkpoint), selected=selected,
                   losses=losses, history_gains={k: v-observed for k,v in losses.items() if k != 'intact'},
                   selected_seen_write_statistics={k: summary(v) for k,v in stats.items()},
                   vjp_probes=probes, observer_partition_rng_parity=True)
        rows.append(row); print(json.dumps(row), flush=True)
    for name, digest in pins.items():
        if sha(ROOT / name) != digest: raise ValueError('Source changed: ' + name)
    result = dict(status='completed', rows=rows, primitive_contract=contract, source_sha256=pins,
                  wall_s=time.monotonic()-started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Frozen selected 8K development checkpoints, same targets and route RNG. Selected-slot decay factors and retained/write norms are measured on intact trajectories; bounded VJP probes measure selected value sensitivity at fixed receiver. Erasure interventions are out of distribution and are not retrained ablations. No public scoring, fitting or FLOP claim.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
