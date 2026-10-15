"""R1 addressed episodic memory vs plain continuation from the strongest native learner (5.978388).

Both arms restore the identical acquired learner, calibrated-cap optimizer, numeric
state, private RNG and teacher, and continue on the identical next fresh-TRAIN
stream (eight contiguous lanes, 4096 updates, about 2M targets). Arm `none` is the
plain data-scaling continuation; arm `episodic` adds the persistent addressed
episodic memory (sleeping_machines/episodic_pointer_memory.py) whose own
parameters use AdamW. Core counterfactual route credit is unchanged and uses the
native readout terms; the training objective is the actual model's likelihood.
TRAIN losses are scored before each update. Final scoring reuses the recorded
8192 DEV targets / disjoint 1024-token windows / no EOS reset, with the store
reset at each window exactly like the numeric state.
"""
import argparse, copy, json, resource, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'experiments/credit')]
import torch
import protected_depth_local_gate as gate
from public_token_scoring import score, windows, reference_forward
from sleeping_machines.episodic_pointer_memory import EpisodicPointerMemory
from sleeping_machines.sparse_counterfactual_episodes import detach
from sleeping_machines.event_credit_sites import sample_site
from sleeping_machines.sampled_suffix_utility import sample_positions, suffix_difference
from sleeping_machines.paired_route_credit import paired_route_credit

TOTAL = 4096
PARENT = ROOT / 'experiments/results/token_language/aws_memory_policy_corpus_eos_2m_20261015_v1'
ACQUIRED = ROOT / 'experiments/results/token_language/aws_multidepth_readout_8m_20261013_s6_v1.json'
driver = gate.pilot.driver


def build():
    args = SimpleNamespace(**json.loads(ACQUIRED.read_text())['args'])
    train = driver.load_tokens(args.train_file)
    counts = driver.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
    order = torch.argsort(counts, descending=True, stable=True)
    torch.manual_seed(args.seed)
    return driver.Model(args, order, counts), args, train


def memory_step(model, memory, mem_state, state, gen, teacher_gen, args, buf, opt_extra):
    """pilot.raw_step for the ungrown learner, with the episodic likelihood as objective."""
    if memory is None:  # control arm: the exact published continuation step
        terms, state, rows = gate.pilot.raw_step(model, state, gen, teacher_gen, args, buf, 0)
        return terms, state, None, rows, {}
    n = buf.shape[1] - 1; audits = []
    if torch.rand((), generator=teacher_gen) < args.teacher_probability:
        token, d, h, inv = sample_site(n, args.depth, args.heads, model.site_generator); audits.append((token, d, h, inv))
    model.suppress_site = audits[0][:3] if audits else None
    values, state, pis = model(buf[:, :-1], state, gen, args)
    terms = model.readout.nll(values, buf[:, 1:]); future = 0.; rows = []
    for token, d, h, inv in audits:
        pi = pis[token * args.depth + d][1][:, h, :].double(); winner = state['race_winners'][token, d, :, h]
        eligible = 1 - torch.nn.functional.one_hot(winner, args.pool).double()
        proposal = .9 * pi.detach() * eligible / (pi.detach() * eligible).sum(-1, keepdim=True) + .1 * eligible / (args.pool - 1)
        alt = torch.multinomial(proposal, 1, generator=model.alternative_generator).squeeze(1); q = proposal.gather(1, alt[:, None]).squeeze(1)
        begin = (token // args.credit_window) * args.credit_window; window = model.replay_window
        model.force_site = (token - begin, d, h, alt); model.shadow_mode = True
        shadow_gen = torch.Generator().set_state(window['rng'])
        end = min(n, (token // args.credit_window + 1) * args.credit_window)
        positions = sample_positions(end - token, args.future_score_positions, model.position_generator) + token; length = int(positions[-1]) + 1
        with torch.no_grad():
            shadow, _, _ = model(buf[:, begin:length], window['state'], shadow_gen, args)
            difference = suffix_difference(model.readout, shadow, buf[:, begin + 1:], terms.reshape(args.lanes, n)[:, begin:], positions - begin)
        model.shadow_mode = False
        weight = inv * (end - token) / n; future = future + weight * paired_route_credit(pi, alt, q, difference) / args.teacher_probability
        rows.append(dict(replayed_targets=(length - begin) * args.lanes, scored_targets=len(positions) * args.lanes))
    model.force_site = model.suppress_site = None; model.replay_window = None
    actual, mem_state, info = memory.mixed_nll(terms, values, buf[:, :-1], buf[:, 1:], mem_state)
    actual = actual.reshape(-1)
    objective = actual.mean() + args.future_scale * future
    assert torch.isfinite(objective); objective.backward()
    return actual.detach(), detach(state), mem_state, rows, info


@torch.no_grad()
def score_episodic(model, memory, tokens, args, start=25165824, targets=8192, length=1024, lanes=8):
    """public_token_scoring.score with the store reset per window like the numeric state."""
    model.eval(); memory.eval(); forward = reference_forward(model, driver); gen = torch.Generator().manual_seed(args.seed + 100000)
    total = 0.; count = 0; d = model.core.total_payload * model.core.depth
    for inputs, labels, begin in windows(tokens, start, targets, length, lanes):
        state = None; mem = memory.empty(inputs.shape[0], d)
        for pos in range(0, length, args.chunk):
            x, state, _ = forward(inputs[:, pos:pos + args.chunk], state, gen, args, deterministic=args.deterministic_eval)
            native = model.readout.nll(x, labels[:, pos:pos + args.chunk])
            losses, mem, _ = memory.mixed_nll(native, x, inputs[:, pos:pos + args.chunk], labels[:, pos:pos + args.chunk], mem)
            total += float(losses.double().sum()); count += losses.numel()
    assert count == targets
    return dict(nll=total / count, loss_sum=total, targets=count, target_start=start + 1, target_end_inclusive=start + targets,
                window=length, lanes=lanes, eos_state_reset=False, store_reset='per window, like numeric state')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--memory', choices=['none', 'episodic'], required=True)
    p.add_argument('--updates', type=int, required=True); p.add_argument('--capacity', type=int, default=1024); p.add_argument('--topk', type=int, default=32)
    p.add_argument('--key-dim', type=int, default=32); p.add_argument('--memory-lr', type=float, default=.003); p.add_argument('--resume-from')
    a = p.parse_args(); assert 1 <= a.updates <= TOTAL; torch.set_num_threads(1); began = time.monotonic()
    model, args, train = build(); parent = torch.load(PARENT.with_suffix('.pt'), weights_only=False)
    assert parent['binding']['reset_policy'] == 'corpus_eos' and parent['added_updates'] == TOTAL
    model.load_state_dict(parent['model'])
    opt = gate.CalibratedReceiverBankAdamW(model, parent['optimizer']['bank_caps'], lr=args.lr, weight_decay=args.weight_decay)
    d = model.core.total_payload * model.core.depth
    memory = EpisodicPointerMemory(d, a.key_dim, a.capacity, a.topk) if a.memory == 'episodic' else None
    mopt = torch.optim.AdamW(memory.parameters(), lr=a.memory_lr, weight_decay=0.) if memory is not None else None
    stream_start = parent['binding']['stream_start'] + TOTAL * args.chunk * args.lanes + args.lanes
    binding = dict(memory=a.memory, capacity=a.capacity, topk=a.topk, key_dim=a.key_dim, memory_lr=a.memory_lr, parent_sha256=driver.sha(PARENT.with_suffix('.pt')),
                   stream_start=stream_start, stream_total_updates=TOTAL, driver_sha256=driver.sha(Path(__file__)),
                   memory_sha256=driver.sha(ROOT / 'sleeping_machines/episodic_pointer_memory.py'))
    ck = parent; offset = 0; elapsed = 0.; work = dict(calls=0, replayed_targets=0, scored_targets=0); curve = []; loss_sum = 0.; count = 0
    mem_state = memory.empty(args.lanes, d) if memory is not None else None; memory_work = dict(key_map_macs=0, query_macs=0, score_macs=0, selected_reads=0)
    if a.resume_from:
        ck = torch.load(ROOT / a.resume_from, weights_only=False); assert ck['binding'] == binding
        model.load_state_dict(ck['model']); offset = ck['added_updates']; elapsed = ck['training_s']; work = ck['teacher_work']; curve = ck['curve']
        loss_sum = ck['loss_sum']; count = ck['targets']
        if memory is not None:
            memory.load_state_dict(ck['memory']); mopt.load_state_dict(ck['memory_optimizer']); mem_state = ck['memory_state']; memory_work = ck['memory_work']
    opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    for name, rng in ck['owned_rng'].items(): getattr(model, name).set_state(rng)
    gen = torch.Generator().set_state(ck['generator']); teacher = torch.Generator().set_state(ck['teacher_generator']); torch.set_rng_state(ck['torch_rng'])
    state = gate.pilot.clone_state(ck['state'])
    tokens = driver.LaneTokens(train, stream_start, TOTAL * args.chunk * args.lanes + args.lanes, args.lanes); assert tokens.shape[1] == TOTAL * args.chunk + 1
    model.train(); prefix = ROOT / f'experiments/results/token_language/{a.tag}'; segment = time.monotonic(); gates = []
    for index in range(offset, a.updates):
        cursor = index * args.chunk; buf = tokens[:, cursor:cursor + args.chunk + 1]; opt.zero_grad()
        if mopt is not None: mopt.zero_grad()
        stored = mem_state['feats'].shape[1] if mem_state is not None else 0
        loss, state, mem_state, audits, info = memory_step(model, memory, mem_state, state, gen, teacher, args, buf, mopt)
        loss_sum += float(loss.double().sum()); count += loss.numel(); opt.step(); opt.zero_grad()
        if mopt is not None:
            mopt.step(); mopt.zero_grad(); gates.append(info)
            for key, value in memory.work(args.lanes, args.chunk, stored, d).items(): memory_work[key] += value
        for audit in audits: work['calls'] += 1; work['replayed_targets'] += audit['replayed_targets']; work['scored_targets'] += audit['scored_targets']
        if (index + 1) % 256 == 0 or index + 1 == a.updates:
            training_s = elapsed + time.monotonic() - segment
            row = dict(update=index + 1, prequential_nll=loss_sum / count, training_s=training_s)
            if gates: row.update({k: sum(g[k] for g in gates) / len(gates) for k in gates[0]}); gates = []
            if memory is not None: row.update(age_rate=float(torch.nn.functional.softplus(memory.age_rate)), boundary=float(memory.boundary), null=float(memory.null))
            curve.append(row)
            saved = dict(model=model.state_dict(), optimizer=opt.state_dict(), state=state, generator=gen.get_state(), teacher_generator=teacher.get_state(),
                         owned_rng=gate.private_state(model), torch_rng=torch.get_rng_state(), step=parent['step'] + index + 1, added_updates=index + 1, targets=count,
                         loss_sum=loss_sum, teacher_work=work, training_s=training_s, curve=curve, binding=binding)
            if memory is not None: saved.update(memory=memory.state_dict(), memory_optimizer=mopt.state_dict(), memory_state=mem_state, memory_work=memory_work)
            tmp = prefix.with_suffix('.tmp'); torch.save(saved, tmp); tmp.replace(prefix.with_suffix('.pt')); print(json.dumps(row), flush=True)
    dev = driver.load_tokens(args.dev_file)
    scored = score(model, dev, args, driver, 25165824, 8192, 1024, 8) if memory is None else score_episodic(model, memory, dev, args)
    native_scored = score(model, dev, args, driver, 25165824, 8192, 1024, 8) if memory is not None else scored
    names = ['experiments/episodic_memory_continuation.py', 'sleeping_machines/episodic_pointer_memory.py', 'experiments/credit/protected_depth_learning_pilot.py',
             'experiments/multidepth_readout_token_language.py', 'experiments/public_token_scoring.py']
    record = dict(status='completed', battle='R1', memory=a.memory, added_updates=a.updates, added_targets=count, prequential_nll=loss_sum / count, scored=scored,
                  native_readout_scored=native_scored, training_s=saved['training_s'], teacher_work=work, memory_work=memory_work if memory is not None else None,
                  memory_parameters=sum(x.numel() for x in memory.parameters()) if memory is not None else 0,
                  final_weights=str(prefix.with_suffix('.pt').relative_to(ROOT)), source_sha256={n: driver.sha(ROOT / n) for n in names},
                  input_sha256={str(PARENT.with_suffix('.pt').relative_to(ROOT)): driver.sha(PARENT.with_suffix('.pt'))}, binding=binding,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, process_wall_s=time.monotonic() - began, host=__import__('platform').node(),
                  scope='Single-factor addressed episodic memory versus plain continuation from the strongest native learner 5.978388 (shared lanes, corpus-EOS core boundary). Same restored learner/caps/moments/state/RNG/teacher and identical next fresh-TRAIN 2M stream. Episodic store persists across chunks and documents (learned age clock and boundary bias), top-k sparse read, read-time credit; core counterfactual credit unchanged. Same 8192 DEV/1024-window scoring, store reset per window. Single seed; no public validation.')
    with prefix.with_suffix('.json').open('x') as f: json.dump(record, f, indent=2)


if __name__ == '__main__':
    main()
