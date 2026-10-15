"""R1 numerical contracts for the addressed episodic memory before any fit."""
import argparse, copy, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'experiments/credit')]
import torch
import episodic_memory_continuation as run
from sleeping_machines.episodic_pointer_memory import EpisodicPointerMemory


def module_contracts():
    torch.manual_seed(0); L, n, d, V = 3, 64, 12, 40
    emb = torch.randn(V, d)
    ids = torch.randint(0, V, (L, n + 1)); ids[0, 10] = 50256 % V  # vocabulary-local stand-in; boundary tested separately
    def feats(t): return emb[t[:, :-1]] + .5 * torch.cat([torch.zeros(L, 1, d), emb[t[:, :-2]]], 1)
    mem = EpisodicPointerMemory(d, 8, 1024, 16, gate_init=.3)
    with torch.no_grad(): mem.gate.weight.normal_(std=.3)
    native = torch.log_softmax(torch.randn(L, n, V), -1)
    x = feats(ids); targets = ids[:, 1:]
    nll = -native.gather(-1, targets[..., None]).squeeze(-1)
    full, _ = mem.full_log_prob(native, x, ids[:, :-1], targets, mem.empty(L, d))
    loss, _, _ = mem.mixed_nll(nll.reshape(-1), x, ids[:, :-1], targets, mem.empty(L, d))
    out = dict(normalization_max_error=float((full.exp().sum(-1) - 1).abs().max()),
               target_consistency_max_error=float((full.gather(-1, targets[..., None]).squeeze(-1) + loss).abs().max()))
    # Causality: changing token s changes features at s,s+1 and target s-1 only.
    s = 40; ids2 = ids.clone(); ids2[:, s] = (ids2[:, s] + 7) % V
    loss2, _, _ = mem.mixed_nll(nll.reshape(-1), feats(ids2), ids2[:, :-1], ids2[:, 1:], mem.empty(L, d))
    out['causality_prefix_max_change'] = float((loss2[:, :s - 1] - loss[:, :s - 1]).abs().max())
    out['causality_suffix_changes'] = bool((loss2[:, s - 1:] - loss[:, s - 1:]).abs().max() > 0)
    # Chunk split: two chunks with a persistent store equal one chunk.
    a, m1, _ = mem.mixed_nll(nll[:, :32].reshape(-1), x[:, :32], ids[:, :32], targets[:, :32], mem.empty(L, d))
    b, m2, _ = mem.mixed_nll(nll[:, 32:].reshape(-1), x[:, 32:], ids[:, 32:64], targets[:, 32:], m1)
    out['chunk_split_max_error'] = float((torch.cat([a, b], 1) - loss).abs().max())
    # Read-time credit: only second-chunk losses, target values stored only in detached first-chunk entries.
    mem.zero_grad(); xr = x.clone().requires_grad_(True)
    _, m1, _ = mem.mixed_nll(nll[:, :32].reshape(-1), xr[:, :32], ids[:, :32], targets[:, :32], mem.empty(L, d))
    b, _, _ = mem.mixed_nll(nll[:, 32:].reshape(-1), xr[:, 32:], ids[:, 32:64], targets[:, 32:], m1)
    b.sum().backward()
    out['old_entries_detached'] = not m1['feats'].requires_grad
    out['read_time_key_gradient_norm'] = float(mem.key.weight.grad.norm())
    out['read_time_query_gradient_norm'] = float(mem.query.weight.grad.norm())
    out['age_clock_gradient'] = float(mem.age_rate.grad.abs())
    out['first_chunk_feature_gradient_from_second_chunk_loss'] = float(xr.grad[:, :32].abs().max())
    # Document boundary: bias only affects entries before the latest EOS.
    with torch.no_grad():
        idsb = torch.randint(0, 100, (1, 33)); idsb[0, 20] = 50256
        xb = torch.randn(1, 32, d); nb = torch.full((32,), 3.)
        base, _, _ = mem.mixed_nll(nb, xb, idsb[:, :-1], idsb[:, 1:], mem.empty(1, d))
        mem.boundary.fill_(-50.); after, _, _ = mem.mixed_nll(nb, xb, idsb[:, :-1], idsb[:, 1:], mem.empty(1, d)); mem.boundary.fill_(0.)
    out['boundary_pre_eos_unchanged'] = float((after[:, :20] - base[:, :20]).abs().max())
    # Capacity bound.
    small = EpisodicPointerMemory(d, 8, 50, 16); state = small.empty(L, d)
    for i in range(0, 64, 16): _, state, _ = small.mixed_nll(nll[:, i:i + 16].reshape(-1), x[:, i:i + 16], ids[:, i:i + 16], targets[:, i:i + 16], state)
    out['capacity_entries'] = int(state['feats'].shape[1])
    return out


def equivalence_contract(compiled=False):
    """Closed gate: episodic step equals the published raw_step exactly (loss, weights, state)."""
    results = {}
    for arm in ('none', 'episodic'):
        model, args, train = run.build(compiled); parent = torch.load(run.PARENT.with_suffix('.pt'), weights_only=False); model.load_state_dict(parent['model'])
        opt = run.gate.CalibratedReceiverBankAdamW(model, parent['optimizer']['bank_caps'], lr=args.lr, weight_decay=args.weight_decay); opt.load_state_dict(copy.deepcopy(parent['optimizer']))
        for name, rng in parent['owned_rng'].items(): getattr(model, name).set_state(rng)
        gen = torch.Generator().set_state(parent['generator']); teacher = torch.Generator().set_state(parent['teacher_generator']); torch.set_rng_state(parent['torch_rng'])
        state = run.gate.pilot.clone_state(parent['state']); d = model.core.total_payload * model.core.depth
        memory = None; mem_state = None
        if arm == 'episodic':
            memory = EpisodicPointerMemory(d); mem_state = memory.empty(args.lanes, d)
            with torch.no_grad(): memory.gate.bias.fill_(-1e4)
        start = parent['binding']['stream_start'] + run.TOTAL * args.chunk * args.lanes + args.lanes
        tokens = run.driver.LaneTokens(train, start, 3 * args.chunk * args.lanes + args.lanes, args.lanes)
        losses = []
        for i in range(3):
            opt.zero_grad(); loss, state, mem_state, _, _ = run.memory_step(model, memory, mem_state, state, gen, teacher, args, tokens[:, i * args.chunk:(i + 1) * args.chunk + 1], None)
            opt.step(); losses.append(loss)
        results[arm] = dict(loss=torch.cat(losses), weights=[p.detach().clone() for p in model.parameters()], state=state)
    a, b = results['none'], results['episodic']
    return dict(closed_gate_loss_max_error=float((a['loss'] - b['loss']).abs().max()),
                closed_gate_weight_max_error=max(float((x - y).abs().max()) for x, y in zip(a['weights'], b['weights'])),
                closed_gate_state_equal=all(torch.equal(a['state'][k], b['state'][k]) for k in a['state'] if torch.is_tensor(a['state'][k])))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--compiled', action='store_true'); a = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    out = module_contracts(); out.update(equivalence_contract(a.compiled)); out['compiled'] = a.compiled
    checks = dict(normalization=out['normalization_max_error'] < 1e-5, target_consistency=out['target_consistency_max_error'] < 1e-4,
                  causality=out['causality_prefix_max_change'] == 0 and out['causality_suffix_changes'], chunk_split=out['chunk_split_max_error'] < 1e-5,
                  read_time_credit=out['old_entries_detached'] and out['read_time_key_gradient_norm'] > 0 and out['read_time_query_gradient_norm'] > 0
                  and out['age_clock_gradient'] > 0 and out['first_chunk_feature_gradient_from_second_chunk_loss'] == 0,
                  boundary=out['boundary_pre_eos_unchanged'] == 0, capacity=out['capacity_entries'] == 50,
                  closed_gate_equivalence=out['closed_gate_loss_max_error'] == 0 and out['closed_gate_weight_max_error'] == 0 and out['closed_gate_state_equal'])
    record = dict(status='completed' if all(checks.values()) else 'failed', battle='R1', checks=checks, measurements=out,
                  source_sha256={n: run.driver.sha(ROOT / n) for n in ('experiments/episodic_memory_contracts.py', 'experiments/episodic_memory_continuation.py', 'sleeping_machines/episodic_pointer_memory.py')},
                  process_wall_s=time.monotonic() - began)
    path = ROOT / f'experiments/results/diagnostics/{a.tag}.json'
    with path.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=record['status'], checks=checks)))
    if record['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__':
    main()
