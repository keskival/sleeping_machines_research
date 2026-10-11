"""B1/R1 connected native route policy, paired credit acquisition and reuse."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import resource
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from task_valued_route_credit import ROOT, SOURCE, load, native
from shared_prefix_receiver_credit import SharedPrefixCredit
from paired_native_replay import replay_pairs

TAG = 'curie_sparse_paired_ac_20261011T0910Z'
SOURCES = ['experiments/credit/sparse_paired_actor_critic.py',
           'experiments/credit/shared_prefix_receiver_credit.py',
           'experiments/credit/paired_native_replay.py',
           'experiments/credit/optimizer_aware_route_credit.py',
           'experiments/credit/task_valued_route_credit.py',
           'experiments/credit/fit_optimizer_aware_credit.py', *native.SOURCES]


def probabilities(scores, tau):
    return .9*(-scores.flatten(1)/tau).softmax(1)+.1/scores[0].numel()


def policy_loss(probs, credit, action, reference, audited, contrast, p):
    if not 0 < p <= 1: raise ValueError('Positive audit probability required')
    h = credit.detach().flatten()-credit.detach().flatten()[reference]
    result = (probs.flatten()*h).sum()
    if audited:
        result = result+(float(contrast)-h[action])*probs.flatten()[action].log()/p
    return result


def scores(network, x, version):
    activation = network.encode_context(x[:, 0, :, :30], 0, version)
    a = native.base.actions()[None, None].expand(len(x), x.shape[1], -1, -1)
    return network.values(activation, x[:, :, :, 30:], a, 0, version)


def contrasts(network, x, actions, references, version):
    activation = network.encode_context(x[:, 0, :, :30], 0, version)
    index = torch.tensor(np.stack((actions, references), 1))
    routes = index//9
    selected = x[torch.arange(len(x))[:, None], routes, :, 30:]
    messages = native.base.actions()[index%9][:, :, None]
    values = network.values(activation, selected, messages, 0, version).squeeze(-1)
    return values[:, 0]-values[:, 1]


def initial():
    result, packets = load(SOURCE)
    _, prior = load(result['prior_result'])
    carrier = native.actor(4, 170); carrier.load_state_dict(prior['actor']); carrier.requires_grad_(False)
    baseline = SharedPrefixCredit(carrier.M, native.prior.bound(carrier)).double()
    baseline.load_state_dict(prior['critic'])
    tau = float(packets['train_tensors']['aware'].square().mean().sqrt().clamp_min(.001))
    scale = float(((packets['train_tensors']['aware']-packets['train_tensors']['base']).square().mean()*2).sqrt().clamp_min(.0001))
    return carrier, baseline, tau, scale, result['prior_checkpoint_sha256']


def contract():
    logits = torch.tensor([[.1, -.3, .4]], requires_grad=True)
    q = torch.tensor([.04, -.01, .02]); predicted = torch.tensor([.01, .03, -.04])
    pi = .9*logits.softmax(1)+.1/3
    exact = torch.autograd.grad((pi*q).sum(), logits, retain_graph=True)[0]
    errors = []
    for p in [.125, .25, 1.]:
        expected = torch.zeros_like(logits)
        for a in range(3):
            for audited, probability in [(False, 1-p), (True, p)]:
                loss = policy_loss(pi, predicted, a, 1, audited, float(q[a]-q[1]), p)
                gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
                expected += float(pi[0, a])*probability*gradient
        error = float((exact-expected).abs().max()); assert error < 1e-12; errors.append(error)
    carrier, baseline, tau, scale, _ = initial()
    t, m = native.base.batch(911111, size=1, pairs=2)
    with torch.no_grad(): x = native.prefix(carrier, t[:, :4], m[:, :4], 3, list(range(8)))
    full = scores(baseline, x, 0)
    a, b = [13], [65]
    sparse = contrasts(baseline, x, a, b, 0)
    parity = float((sparse-(full.flatten(1)[:, a[0]]-full.flatten(1)[:, b[0]])).abs().max())
    g1 = torch.autograd.grad(sparse.sum(), tuple(baseline.parameters()))
    g2 = torch.autograd.grad((full.flatten(1)[:, a[0]]-full.flatten(1)[:, b[0]]).sum(), tuple(baseline.parameters()))
    grad_parity = max(float((u-v).abs().max()) for u, v in zip(g1, g2))
    assert parity < 1e-12 and grad_parity < 1e-11
    activation = baseline.encode_context(x[:, 0, :, :30], 0, 3)
    rejected = 0
    try: baseline.values(activation, x[:, :, :, 30:], native.base.actions()[None, None].expand(1, 8, -1, -1), 0, 4)
    except ValueError: rejected += 1
    try: policy_loss(pi, predicted, 0, 1, False, 0., 0.)
    except ValueError: rejected += 1
    assert rejected == 2
    return dict(raw_expected_gradient_errors=errors, sparse_contrast_value_error=parity,
                sparse_contrast_gradient_error=grad_parity, stale_or_zero_support_rejections=rejected,
                controller_parameters=sum(p.numel() for p in baseline.parameters()),
                critic_parameters=sum(p.numel() for p in baseline.parameters()), tau=tau, critic_scale=scale)


def create_state(baseline, arm):
    actor = deepcopy(baseline); critic = deepcopy(baseline)
    policy_opt = torch.optim.AdamW(actor.parameters(), lr=.0003, weight_decay=.001)
    credit_opt = torch.optim.AdamW(critic.parameters(), lr=.001, weight_decay=.001)
    return dict(actor=actor, critic=critic, policy_opt=policy_opt, credit_opt=credit_opt,
                arm=arm, cursor=0, critic_version=0, buffer=[], history=[])


def pack(state):
    return dict(actor=state['actor'].state_dict(), critic=state['critic'].state_dict(),
                policy_optimizer=state['policy_opt'].state_dict(), critic_optimizer=state['credit_opt'].state_dict(),
                arm=state['arm'], cursor=state['cursor'], critic_version=state['critic_version'],
                buffer=state['buffer'], history=state['history'])


def unpack(payload, baseline):
    state = create_state(baseline, payload['arm'])
    state['actor'].load_state_dict(payload['actor']); state['critic'].load_state_dict(payload['critic'])
    state['policy_opt'].load_state_dict(payload['policy_optimizer']); state['credit_opt'].load_state_dict(payload['critic_optimizer'])
    for key in ['cursor', 'critic_version', 'buffer', 'history']: state[key] = payload[key]
    return state


def train(state, carrier, baseline, tau, scale, end, learn_credit, preview):
    start = time.monotonic(); begin = state['cursor']; p = 1. if state['arm'] == 'full_fixed' else .25
    policy_updates = credit_updates = extra_worlds = audit_draws = 0
    residuals, factual_losses = [], []
    with FlopCounterMode(display=False) as counter:
        for i in range(begin, end):
            t, m = native.base.batch(144000000+701*1000+i, size=1, pairs=2)
            with torch.no_grad():
                x = native.prefix(carrier, t[:, :4], m[:, :4], 3, list(range(6)))
                reference = int(scores(baseline, x, 0).flatten().argmin())
                issued_credit = scores(state['critic'], x, state['critic_version']).detach()
            pi = probabilities(scores(state['actor'], x, i), tau)
            draw = np.random.default_rng(73000000+i).random()
            action = min(int(np.searchsorted(pi.detach().flatten().numpy().cumsum(), draw)), 53)
            audited = bool(np.random.default_rng(74000000+i).random() < p); audit_draws += audited
            identity = action == reference or (action%9 == reference%9 == 0)
            paid = audited and not identity
            rng = np.random.default_rng(75000000+i)
            ft, fm = native.prior.continuation(t, m, 2, rng)
            ids = [action, reference] if paid else [action]
            losses, _ = replay_pairs(carrier, ft, fm, [k//9 for k in ids], native.base.actions()[[k%9 for k in ids]], .05)
            cost = float(losses[0].mean()); factual_losses.append(cost)
            delta = float((losses[0]-losses[1]).mean()) if paid else 0.
            predicted = float(issued_credit.flatten()[action]-issued_credit.flatten()[reference])
            if paid: residuals.append((delta-predicted)**2); extra_worlds += 1
            loss = policy_loss(pi, issued_credit, action, reference, audited, delta, p)
            state['policy_opt'].zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(state['actor'].parameters(), 1.); assert torch.isfinite(norm)
            state['policy_opt'].step(); policy_updates += 1
            if learn_credit and paid:
                state['buffer'].append(dict(x=x.detach(), action=action, reference=reference, delta=delta,
                                            carrier_version=0, policy_version=i, horizon=4))
                state['buffer'] = state['buffer'][-128:]
                for j in range(4):
                    rng = np.random.default_rng(76000000+i*4+j)
                    selected = rng.choice(len(state['buffer']), min(16, len(state['buffer'])), replace=False)
                    data = [state['buffer'][int(k)] for k in selected]
                    bx = torch.cat([d['x'] for d in data])
                    pred = contrasts(state['critic'], bx, [d['action'] for d in data], [d['reference'] for d in data], state['critic_version'])
                    target = torch.tensor([d['delta'] for d in data])
                    credit_loss = ((pred-target)/scale).square().mean(); assert torch.isfinite(credit_loss)
                    state['credit_opt'].zero_grad(); credit_loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(state['critic'].parameters(), 1.); assert torch.isfinite(norm)
                    state['credit_opt'].step(); credit_updates += 1; state['critic_version'] += 1
            state['cursor'] = i+1
            state['history'].append(dict(prefix=i, action=action, reference=reference, audited=audited, extra_world=paid,
                                         factual_loss=cost, issued_contrast=predicted, observed_contrast=delta if paid or identity else None,
                                         credit_version=state['critic_version']))
            if (i+1)%32 == 0 or i+1 == end:
                preview.mkdir(parents=True, exist_ok=True)
                torch.save(pack(state), preview/'resume.pt')
                progress = dict(arm=state['arm'], completed_prefixes=i+1, stage_start=begin,
                                wall_s=time.monotonic()-start, supported_flops=counter.get_total_flops(),
                                paid_extra_worlds=extra_worlds, critic_updates=credit_updates)
                (preview/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                print(json.dumps(progress), flush=True)
    work = counter.get_total_flops()
    return dict(arm=state['arm'], start_prefix=begin, end_prefix=end, unique_factual_targets=(end-begin)*4,
                prefix_presentations=end-begin, policy_updates=policy_updates, credit_updates=credit_updates,
                audit_draws=audit_draws, extra_worlds=extra_worlds, factual_worlds=end-begin,
                total_readout_updates=(end-begin+extra_worlds)*4, supported_flops=work,
                supported_flops_per_factual_target=work/((end-begin)*4), wall_s=time.monotonic()-start,
                mean_factual_prequential_loss=float(np.mean(factual_losses)),
                issued_audited_contrast_mse=float(np.mean(residuals)) if residuals else None,
                critic_training_enabled=learn_credit)


def evaluate(states, carrier, baseline, count, futures, epoch):
    names = list(states); gains = {name: [] for name in names}; choices = {name: [] for name in names}
    references = []
    start = time.monotonic(); teacher_worlds = 0
    with FlopCounterMode(display=False) as counter:
        for i in range(count):
            t, m = native.base.batch(145000000+721*1000+i, size=1, pairs=2)
            with torch.no_grad():
                x = native.prefix(carrier, t[:, :4], m[:, :4], 3, list(range(8)))
                reference = int(scores(baseline, x, 0).flatten().argmin())
                selected = {name: int(scores(states[name]['actor'], x, epoch).flatten().argmin()) for name in names}
                references.append(reference)
            # Shared verification worlds; all requested actions are paid, no min-result selection.
            ids = list(dict.fromkeys([reference, *selected.values()])); position = {a: j for j, a in enumerate(ids)}
            rng = np.random.default_rng(77000000+i)
            response = []
            for _ in range(futures):
                ft, fm = native.prior.continuation(t, m, 2, rng)
                losses, _ = replay_pairs(carrier, ft, fm, [a//9 for a in ids], native.base.actions()[[a%9 for a in ids]], .05)
                response.append(losses.mean(1)); teacher_worlds += len(ids)
            response = torch.stack(response).mean(0)
            for name in names:
                gains[name].append(float(response[position[reference]]-response[position[selected[name]]]))
                choices[name].append(selected[name])
    rng = np.random.default_rng(781); indices = rng.integers(0, count, (4000, count))
    rows = []
    for name in names:
        values = np.array(gains[name])
        rows.append(dict(arm=name, gain_vs_baseline=float(values.mean()), bootstrap95=np.quantile(values[indices].mean(1), [.025, .975]).tolist(),
                         changed_prefixes=sum(a != b for a, b in zip(choices[name], references)), choices=choices[name]))
    comparisons = {}
    for first, second in [('sparse_learned', 'sparse_fixed'), ('sparse_learned', 'full_fixed'), ('sparse_learned', 'sparse_reset')]:
        if first in gains and second in gains:
            delta = np.array(gains[first])-np.array(gains[second])
            comparisons[f'{first}_over_{second}'] = dict(gain=float(delta.mean()), bootstrap95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist())
    return dict(rows=rows, comparisons=comparisons, prefixes=count, continuations=futures,
                teacher_worlds=teacher_worlds, supported_flops=counter.get_total_flops(), wall_s=time.monotonic()-start)


def run(mode, tag):
    carrier, baseline, tau, scale, prior_sha = initial()
    prior_fits = {}
    if mode == 'reuse':
        previous, cp = load(f'experiments/results/credit/{TAG}_acquire.json')
        for path, digest in previous['source_sha256'].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
        states = {name: unpack(payload, baseline) for name, payload in cp['states'].items()}
        reset = unpack(cp['states']['sparse_learned'], baseline); reset['arm'] = 'sparse_reset'
        reset['critic'].load_state_dict(baseline.state_dict()); reset['buffer'] = []; reset['critic_version'] = 0
        states['sparse_reset'] = reset; end = 512
        prior_fits = {r['arm']: r for r in previous['metrics']['fits']}
        prior_fits['sparse_reset'] = prior_fits['sparse_learned']
    else:
        states = {name: create_state(baseline, name) for name in ['full_fixed', 'sparse_fixed', 'sparse_learned']}
        end = 4 if mode == 'smoke' else 128
    fits = []
    for name, state in states.items():
        fits.append(train(state, carrier, baseline, tau, scale, end, mode != 'reuse' and name == 'sparse_learned',
                          ROOT/'.git/sparse-paired-ac-preview'/tag/name))
    evaluation = evaluate(states, carrier, baseline, 2 if mode == 'smoke' else 32, 1 if mode == 'smoke' else 8, end)
    cumulative = []
    for fit in fits:
        total = dict(fit)
        if fit['arm'] in prior_fits:
            parent = prior_fits[fit['arm']]
            for field in ['unique_factual_targets', 'prefix_presentations', 'policy_updates', 'credit_updates',
                          'audit_draws', 'extra_worlds', 'factual_worlds', 'total_readout_updates', 'supported_flops', 'wall_s']:
                total[field] += parent[field]
            total['start_prefix'] = parent['start_prefix']
            total['supported_flops_per_factual_target'] = total['supported_flops']/total['unique_factual_targets']
        cumulative.append(total)
    return dict(fits=fits, evaluation=evaluation, tau=tau, critic_scale=scale, initial_checkpoint_sha256=prior_sha,
                cumulative_fits=cumulative,
                controller_parameters=sum(p.numel() for p in baseline.parameters()), critic_parameters=sum(p.numel() for p in baseline.parameters()),
                total_training_supported_flops=sum(r['supported_flops'] for r in fits)), dict(states={name: pack(state) for name, state in states.items()}, evaluation=evaluation)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=['contract', 'smoke', 'acquire', 'reuse'], required=True); a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    out = ROOT/'experiments/results/credit'/f'{a.tag}.json'; assert not out.exists(); start = time.monotonic()
    if a.mode in ['acquire', 'reuse']:
        for mode in ['contract', 'smoke']:
            d = json.loads((ROOT/f'experiments/results/credit/{TAG}_{mode}.json').read_text()); assert d['status'] == 'completed'
            for path, digest in d['source_sha256'].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    if a.mode == 'contract': metrics, payload = contract(), None
    else: metrics, payload = run(a.mode, a.tag)
    result = dict(status='completed', tag=a.tag, mode=a.mode, battle='B1/R1', metrics=metrics,
                  whole_wall_s=time.monotonic()-start, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                  scope='Native timed/keyed depth4 carrier fixed; equally sized connected route controllers and credit networks. Hard optional writes; every episode has factual supervised feedback/four causal readout Adam updates, initialized independently per episode. Route policy and credit co-adapt across episodes. TRAIN receivers0–5, DEV0–7. Raw unbiased gradient identity covers route-policy parameters only; clipping/Adam nonlinear. Positive-support audit0.25 versus full1; all discovery, controller/critic/optimizer/replay work charged by supported FLOPs and full wall, inherited pretraining additional. DEV teacher shared across arms, separately charged. No TEST, all-carrier learning, continuous market history, hardware or depth-scaling claim.')
    if payload is not None:
        cp = out.with_suffix('.pt'); torch.save(payload, cp)
        result.update(checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__': main()
