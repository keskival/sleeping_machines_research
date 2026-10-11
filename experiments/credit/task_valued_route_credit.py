"""B1/R1 decision-valued combined route response, TRAIN cross-fit then fresh worlds."""
import argparse
from copy import deepcopy
import hashlib
import json
import resource
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from fit_optimizer_aware_credit import ROOT, Correction, collect, native
from shared_prefix_receiver_credit import SharedPrefixCredit
from cached_learning_effect import values

PREFIX = 'curie_task_route_20261011T0340Z'
SOURCE = 'experiments/results/credit/curie_optimizer_aware_fit_20261011T0010Z_pilot.json'
SOURCES = ['experiments/credit/task_valued_route_credit.py',
           'experiments/credit/cached_learning_effect.py',
           'experiments/credit/fit_optimizer_aware_credit.py',
           'experiments/credit/optimizer_aware_route_credit.py',
           'experiments/credit/shared_prefix_receiver_credit.py', *native.SOURCES]


def load(path):
    result = json.loads((ROOT/path).read_text())
    assert result['status'] == 'completed'
    cp = ROOT/result['checkpoint']
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == result['checkpoint_sha256']
    return result, torch.load(cp, map_location='cpu', weights_only=False)


def objective(pred, base, truth, scale, tau):
    logp = (-pred.flatten(1)/tau).log_softmax(1)
    logprior = (-base.flatten(1)/tau).log_softmax(1).detach()
    p = logp.exp()
    return ((p*truth.flatten(1)).sum(1)/scale + .1*(p*(logp-logprior)).sum(1)).mean()


def calibration(data):
    scale = float((data['aware']-data['base']).square().mean().sqrt().clamp_min(1e-8))
    sorted_base = data['base'].flatten(1).sort(1).values
    tau = float((sorted_base[:, 1]-sorted_base[:, 0]).median().clamp_min(1e-4))
    return scale, tau


def fit(data, arm, steps):
    scale, tau = calibration(data)
    torch.manual_seed(170)
    head = Correction(scale).double()
    opt = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.001)
    n, r, _ = data['aware'].shape
    actions = native.base.actions()[None, None].expand(n, r, -1, -1)
    tick = time.monotonic()
    with FlopCounterMode(display=False) as counter:
        for _ in range(steps):
            pred = data['base'] + values(head, data['z'], actions)
            loss = objective(pred, data['base'], data['aware'], scale, tau) if arm == 'task' else ((pred-data['aware'])/scale).square().mean()
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head, opt, dict(scale=scale, tau=tau, supported_flops=counter.get_total_flops(), wall_s=time.monotonic()-tick)


def contract():
    torch.manual_seed(17)
    base = torch.randn(3, 2, 4)*.02
    truth = torch.randn(3, 2, 4)*.03
    pred = (base+torch.randn_like(base)*.001).requires_grad_()
    scale, tau = .02, .004
    loss = objective(pred, base, truth, scale, tau)
    grad = torch.autograd.grad(loss, pred)[0]
    shifted = (pred.detach()+.4).requires_grad_()
    newloss = objective(shifted, base+.4, truth+.4, scale, tau)
    shiftgrad = torch.autograd.grad(newloss, shifted)[0]
    shift_error = float((grad-shiftgrad).abs().max())
    offset_error = abs(float(newloss-loss)-.4/scale)
    eps = 1e-7
    plus, minus = pred.detach().clone(), pred.detach().clone()
    plus[1, 1, 2] += eps
    minus[1, 1, 2] -= eps
    fd = float((objective(plus, base, truth, scale, tau)-objective(minus, base, truth, scale, tau))/(2*eps))
    fd_error = abs(fd-float(grad[1, 1, 2]))
    p0 = (-base.flatten(1)/tau).softmax(1)
    logstar = (-base.flatten(1)/tau-truth.flatten(1)/(.1*scale)).log_softmax(1)
    star = logstar.exp()
    def risk(p):
        return ((p*truth.flatten(1)).sum(1)/scale+.1*(p*(p.clamp_min(1e-300).log()-p0.log())).sum(1)).mean()
    improvement = float(risk(p0)-risk(star))
    # Exact soft-policy optimum has equal constrained derivatives for supported actions.
    stationarity = truth.flatten(1)/scale+.1*(logstar-p0.log()+1)
    stationarity_error = float((stationarity-stationarity[:, :1]).abs().max())
    head = Correction(.02).double()
    torch.nn.init.normal_(head.net[-1].weight)
    z = torch.randn(3, 2, 38)
    actions = native.base.actions()[None, None].expand(3, 2, -1, -1)
    no_op_error = float(values(head, z, actions)[:, :, 0].abs().max())
    assert shift_error < 1e-10 and offset_error < 1e-10 and fd_error < 1e-7
    assert stationarity_error < 1e-10 and improvement >= -1e-12 and no_op_error == 0
    return dict(shift_gradient_error=shift_error, loss_offset_error=offset_error,
                finite_difference_error=fd_error, soft_optimum_stationarity_error=stationarity_error,
                soft_optimum_risk_improvement=improvement, no_op_error=no_op_error)


def selection(data, smoke):
    n = 8 if smoke else 32
    folds, steps = (2, 2) if smoke else (4, 256)
    data = {k: v[:n] for k, v in data.items()}
    order = np.random.default_rng(291).permutation(n)
    candidates = [('baseline', 0.)]+[(arm, alpha) for arm in ['mse', 'task'] for alpha in [.25, .5, 1.]]
    gains = np.zeros((n, len(candidates)))
    records, work = [], []
    for held in np.array_split(order, folds):
        train = np.setdiff1d(np.arange(n), held)
        fitdata = {k: v[train] for k, v in data.items()}
        truth = data['aware'][held].flatten(1)
        base = data['base'][held]
        before = truth.gather(1, base.flatten(1).argmin(1)[:, None]).squeeze(1)
        for arm in ['mse', 'task']:
            head, _, cost = fit(fitdata, arm, steps)
            work.append(dict(arm=arm, stage='crossfit', **cost))
            with torch.no_grad():
                effect = values(head, data['z'][held], native.base.actions()[None, None].expand(len(held), 6, -1, -1))
                for j, (name, alpha) in enumerate(candidates):
                    if name != arm: continue
                    chosen = (base+alpha*effect).flatten(1).argmin(1)
                    after = truth.gather(1, chosen[:, None]).squeeze(1)
                    gains[held, j] = (before-after).numpy()
        records.append(dict(fit_indices=train.tolist(), held_indices=held.tolist(), **dict(zip(['scale', 'tau'], calibration(fitdata)))))
    bootstrap = np.random.default_rng(292).integers(0, n, (4000, n))
    means = gains.mean(0)
    lower = np.quantile(gains[bootstrap].mean(1), .05/6, axis=0)
    accepted = [j for j in range(1, len(candidates)) if lower[j] > 0]
    chosen = max(accepted, key=lambda j: means[j]) if accepted else 0
    payload = dict(heads={}, optimizers={}, preprocessing={})
    for arm in ['mse', 'task']:
        head, opt, cost = fit(data, arm, steps)
        work.append(dict(arm=arm, stage='refit', **cost))
        payload['heads'][arm] = deepcopy(head.state_dict())
        payload['optimizers'][arm] = deepcopy(opt.state_dict())
        payload['preprocessing'][arm] = dict(scale=head.scale, tau=cost['tau'])
    return dict(chosen_arm=candidates[chosen][0], chosen_alpha=candidates[chosen][1],
                candidates=[dict(arm=a, alpha=b, gain=float(means[j]), lower=float(lower[j])) for j, (a, b) in enumerate(candidates)],
                folds=records, train_prefixes=n, steps=steps, fit_work=work), payload


def fresh(saved):
    selected, cp = load(f'experiments/results/credit/{PREFIX}_select.json')
    # Verify all selection implementation dependencies before admitting unseen episodes.
    for path, digest in selected['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    source, _ = load(SOURCE)
    _, prior = load(source['prior_result'])
    actor = native.actor(4, 170)
    actor.load_state_dict(prior['actor'])
    critic = SharedPrefixCredit(actor.M, native.prior.bound(actor)).double()
    critic.load_state_dict(prior['critic'])
    critic.requires_grad_(False)
    with FlopCounterMode(display=False) as teacher:
        z, base, static, truth = collect(actor, critic, 471, 32, 8, list(range(8)), .05, True)
    decisions = selected['metrics']
    heads = {}
    for arm in ['mse', 'task']:
        heads[arm] = Correction(cp['preprocessing'][arm]['scale']).double()
        heads[arm].load_state_dict(cp['heads'][arm])
    actions = native.base.actions()[None, None].expand(32, 8, -1, -1)
    flat = truth.flatten(1)
    fixed = int(saved['train_tensors']['aware'].mean(0).flatten().argmin())
    boot = np.random.default_rng(293).integers(0, 32, (4000, 32))
    rows, losses = [], {}
    with torch.no_grad():
        policies = [('baseline', 'baseline', 0.), ('locked', decisions['chosen_arm'], decisions['chosen_alpha']),
                    ('mse_full_diagnostic', 'mse', 1.), ('task_full_diagnostic', 'task', 1.)]
        for name, arm, alpha in policies:
            with FlopCounterMode(display=False) as query:
                pred = base if arm == 'baseline' else base+alpha*values(heads[arm], z, actions)
            chosen = pred.flatten(1).argmin(1)
            value = flat.gather(1, chosen[:, None]).squeeze(1)
            losses[name] = value
            gain = (flat[:, fixed]-value).numpy()
            over = (losses['baseline']-value).numpy()
            rows.append(dict(policy=name, arm=arm, alpha=alpha, gain_vs_fixed=float(gain.mean()),
                             gain_over_baseline=float(over.mean()), bootstrap95=np.quantile(over[boot].mean(1), [.025, .975]).tolist(),
                             query_supported_flops=query.get_total_flops(),
                             oracle_regret=float((value-flat.min(1).values).mean()),
                             choices=torch.stack((chosen//9, chosen%9), 1).tolist()))
    return dict(rows=rows, chosen_arm=decisions['chosen_arm'], chosen_alpha=decisions['chosen_alpha'],
                teacher_supported_flops=teacher.get_total_flops(), stream_seed=471, prefixes=32, continuations=8,
                selection_checkpoint_sha256=selected['checkpoint_sha256']), dict(packets=dict(z=z, base=base, static=static, aware=truth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=['contract', 'smoke', 'select', 'fresh'], required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    out = ROOT/'experiments/results/credit'/f'{args.tag}.json'
    assert not out.exists()
    tick = time.monotonic()
    source, saved = load(SOURCE)
    if args.mode not in ['contract', 'smoke']:
        c = json.loads((ROOT/f'experiments/results/credit/{PREFIX}_contract.json').read_text())
        s = json.loads((ROOT/f'experiments/results/credit/{PREFIX}_smoke.json').read_text())
        assert c['status'] == s['status'] == 'completed'
        for packet in [c, s]:
            for path, digest in packet['source_sha256'].items():
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    if args.mode == 'contract': metrics, payload = contract(), None
    elif args.mode == 'fresh': metrics, payload = fresh(saved)
    else: metrics, payload = selection(saved['train_tensors'], args.mode == 'smoke')
    result = dict(status='completed', battle='B1/R1', tag=args.tag, mode=args.mode, metrics=metrics,
                  whole_wall_s=time.monotonic()-tick, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_checkpoint_sha256=source['checkpoint_sha256'],
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                  scope='One synthetic actor seed170; native depth4 timed/keyed actor with optional hard writes, readout-only causal Adam. TRAIN selection uses32 paid prefixes/receivers0–5; fresh stream471 generated after locked selection. Fold bootstrap is exploratory, with overlapping training folds. Same2817 correction parameters/initialization/256 steps; known softmax/KL primitives. All prior teacher/baseline work remains charged; new fitting and replay recorded. No TEST, online critic co-adaptation or whole-fit/depth efficiency claim.')
    if payload is not None:
        checkpoint = out.with_suffix('.pt')
        torch.save(payload, checkpoint)
        result.update(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__': main()
