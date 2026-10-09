#!/usr/bin/env python3
"""v6: matched seeded teacher, synthetic DEV labels, compact credit networks and
proposal diagnostics. v5 queues stay immutable. Leading-linear MACs are estimates;
optimizer parameter visits and evaluation work are reported separately. This is a
hard-route credit diagnostic, without the integrated temporal memory.

v5 (9 Oct, note 160 sect. 6, founder direction "the recursion needs to close ... converge towards learning both
sides"): adds `closed`, reweighted wake-sleep with one shared verdict. The credit model proposes --proposals causes per
real example; the forward pass evaluates ONLY those (prior x expert likelihood); the self-normalized importance weights
w_s = p(y, k_s | x) / q(k_s | x, y) are the forward pass's verdict on the backward pass's proposals. The router, the
proposed experts AND the credit model are all trained on the same weights (plus the sleep loss), so both sides climb toward
the a candidate common stationary point: q = exact posterior on the data and theta = stationary marginal likelihood.
Finite-proposal updates are biased away from that posterior; global convergence is not proved. Cost: --proposals experts per
example. Also logs the credit gap KL(rho || q) and total variation on the test set for every credit-model arm (dense, for
evaluation only): the convergence monitor of the backward side.

v4 (9 Oct, note 160 sect. 6): adds hindsight_hybrid. The credit model is trained on the model's own samples (sleep) and,
on a fraction --dense-frac of REAL examples, on the exact posterior computed densely (wake supervision), so q learns credit
on the data distribution as well, at --dense-frac x the dense expert cost. Sect. 6 predicts the bias of learned credit is
bounded by q's error on real data, which pure sleep training reaches only as the model approaches the data.

v3 (9 Oct, founder direction: the forward pass should give the backward pass more than activations, and what it gives
should train): adds hindsight_msg, in which the router emits a learned --msg-dim message m(x) per example and the credit
model sees only (m, y), not x; the message head is trained solely by the credit model's sleep loss (it never affects the
forward output), so the forward pass learns what to tell the backward pass.

v2 (9 Oct, founder direction: reduced synchrony, reduced FLOPs, the forward pass training the SLOW parameters of the
backward pass): adds hindsight_async (sleep samples from a stale snapshot of the student, refreshed every --snap steps,
through a replay buffer: the credit model trains asynchronously, off the forward lockstep) and hindsight_slow (credit model
updated at 1/--q-every of the forward rate), and counts multiply-accumulates per training example for every arm
(router, experts, credit model, sleep sampling).

Theory note 160, prediction 2-3: a credit model trained by the forward race's own sampled causes (hindsight credit)
versus the standard ways of training hard, winner-only routing. CPU, small, synthetic with a known teacher.

Model (student and teacher share the structure): input x -> a race of K route clocks with log-rates s(x) = R x (exponential
race: P(route k) = softmax(s)_k exactly) -> only the winning route's expert runs: p(y | x, k) = softmax(E_k x) over C classes.
Depth 2: a second race among K2 routes conditioned on the first route (R2[k1] x) -> expert indexed by (k1, k2).
Exact identity (note 160): d log p(y|x) / d s_k = rho_k - pi_k, rho = posterior over routes given (x, y).

Arms (training signal for the router; expert update), with expert evaluations per training example counted:
  dense      exact marginal likelihood log sum_k pi_k p(y|x,k) (all experts; upper reference)
  reinforce  winner-only: route k ~ pi; expert k trained; router by score function (log p(y|x,k) - batch baseline)
  gumbel_st  straight-through Gumbel-softmax: hard forward, soft backward (needs every expert's output for the backward)
  hindsight  PROPOSED: a credit model q(route | x, y) is trained ONLY on the student's own forward samples (sleep: sample
             k ~ pi(x), y ~ p(.|x,k); the true cause k is known); on real data the router gets (q - pi) (wake) and one expert
             k ~ q(.|x, y) is trained (EM with a learned E-step): winner-only execution
  oracle     as hindsight but with the exact posterior rho (computed with all experts): diagnostic upper bound for q
Evaluation: held-out marginal log-likelihood (dense, exact) and route recovery (purity of the student's argmax route
w.r.t. the teacher's route). Run under run_safe.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


class Net(nn.Module):
    def __init__(self, d, K, C, depth, K2):
        super().__init__()
        self.depth, self.K, self.K2 = depth, K, K2
        self.R = nn.Linear(d, K)
        if depth == 2:
            self.R2 = nn.Linear(d, K * K2)
        self.n_exp = K * (K2 if depth == 2 else 1)
        self.E = nn.Parameter(torch.randn(self.n_exp, C, d) * 0.1); self.Eb = nn.Parameter(torch.zeros(self.n_exp, C))

    def route_logprior(self, x):
        """log prior over leaf experts [B, n_exp] (exact race probabilities)."""
        l1 = F.log_softmax(self.R(x), -1)
        if self.depth == 1:
            return l1
        l2 = F.log_softmax(self.R2(x).view(-1, self.K, self.K2), -1)
        return (l1[:, :, None] + l2).reshape(len(x), -1)

    def expert_logp(self, x, y, k=None):
        """log p(y | x, expert); k=None evaluates every expert [B, n_exp], else only expert k [B]."""
        if k is None:
            logits = torch.einsum('ecd,bd->bec', self.E, x) + self.Eb
            return F.log_softmax(logits, -1).gather(-1, y[:, None, None].expand(-1, self.n_exp, 1)).squeeze(-1)
        logits = torch.einsum('bcd,bd->bc', self.E[k], x) + self.Eb[k]
        return F.log_softmax(logits, -1).gather(-1, y[:, None]).squeeze(-1)

    def sample(self, x, gen):
        k = torch.multinomial(self.route_logprior(x).exp(), 1, generator=gen).squeeze(-1)
        logits = torch.einsum('bcd,bd->bc', self.E[k], x) + self.Eb[k]
        return k, torch.multinomial(F.softmax(logits, -1), 1, generator=gen).squeeze(-1)


class Credit(nn.Module):
    """q(expert | x, y): the learned backward pass (credit model). With a message head, q sees only (m(x), y)."""
    def __init__(self, d, C, n_exp, h=64, msg=0):
        super().__init__()
        self.msg = nn.Linear(d, msg) if msg else None                 # forward-side message parameters (phi)
        self.emb = nn.Embedding(C, h); self.inp = nn.Linear(msg if msg else d, h)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, n_exp))

    def forward(self, x, y):
        if self.msg is not None:
            x = self.msg(x)                                           # the backward pass receives only the message
        return F.log_softmax(self.out(self.inp(x) + self.emb(y)), -1)


def make_data(a, gen):
    # Isolate the teacher initialization from learner/global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(a.data_seed)
        T = Net(a.d, a.K, a.C, a.depth, a.K2)
    with torch.no_grad():
        T.R.weight.mul_(4.0); T.E.mul_(30.0)                      # confident teacher routes, distinct experts
        if a.depth == 2:
            T.R2.weight.mul_(4.0)
        x = torch.randn(a.n_train + a.n_test, a.d, generator=gen)
        k, y = T.sample(x, gen)
        xt, yt = x[a.n_train:], y[a.n_train:]
        teacher_ll = torch.logsumexp(T.route_logprior(xt) + T.expert_logp(xt, yt), -1).mean().item()
    return x[:a.n_train], y[:a.n_train], k[:a.n_train], xt, yt, k[a.n_train:], teacher_ll


def credit_gap(m, q, x, y):
    """KL(rho || q) and total variation between the exact posterior and the credit model on held-out data."""
    with torch.no_grad():
        lr = F.log_softmax(m.route_logprior(x) + m.expert_logp(x, y), -1); lq = q(x, y)
        return (lr.exp() * (lr - lq)).sum(-1).mean().item(), 0.5 * (lr.exp() - lq.exp()).abs().sum(-1).mean().item()


def evaluate(m, x, y, k_true, n_exp):
    with torch.no_grad():
        lp = torch.logsumexp(m.route_logprior(x) + m.expert_logp(x, y), -1).mean().item()
        kh = m.route_logprior(x).argmax(-1)                       # student's most likely route
        joint = torch.zeros(n_exp, n_exp); joint.index_put_((kh, k_true), torch.ones(len(kh)), accumulate=True)
        purity = (joint.max(1).values.sum() / len(kh)).item()     # each student route mapped to its majority teacher route
    return lp, purity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--arm', required=True,
                    choices=['dense', 'reinforce', 'gumbel_st', 'hindsight', 'oracle', 'hindsight_async', 'hindsight_slow', 'hindsight_msg', 'hindsight_hybrid', 'closed'])
    ap.add_argument('--depth', type=int, default=1); ap.add_argument('--K', type=int, default=8); ap.add_argument('--K2', type=int, default=4)
    ap.add_argument('--d', type=int, default=16); ap.add_argument('--C', type=int, default=10)
    ap.add_argument('--n-train', type=int, default=20000); ap.add_argument('--n-test', type=int, default=5000)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--sleep', type=int, default=1, help='credit-model samples per example')
    ap.add_argument('--msg-dim', type=int, default=4); ap.add_argument('--dense-frac', type=float, default=0.1)
    ap.add_argument('--proposals', type=int, default=2)
    ap.add_argument('--credit-hidden', type=int, default=64)
    ap.add_argument('--snap', type=int, default=50); ap.add_argument('--q-every', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--data-seed', type=int, default=1234)
    a = ap.parse_args(); t0 = time.time(); torch.set_num_threads(1)
    if min(a.proposals, a.credit_hidden, a.epochs, a.n_train, a.n_test, a.batch, a.sleep) < 1:
        ap.error('proposal, width, sample and epoch counts must be positive')
    if not 0 <= a.dense_frac <= 1:
        ap.error('dense-frac must be in [0, 1]')
    output = ROOT / 'experiments/results/credit' / f'{a.tag}.json'
    if output.exists():
        raise SystemExit(f'result exists: {output}; use a unique tag')
    gd = torch.Generator().manual_seed(a.data_seed)
    xtr, ytr, ktr, xte, yte, kte, teacher_ll = make_data(a, gd)
    torch.manual_seed(a.seed); g = torch.Generator().manual_seed(a.seed)
    m = Net(a.d, a.K, a.C, a.depth, a.K2); n_exp = m.n_exp
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    import copy
    hs = a.arm.startswith('hindsight') or a.arm == 'closed'
    q = Credit(a.d, a.C, n_exp, h=a.credit_hidden, msg=a.msg_dim if a.arm == 'hindsight_msg' else 0) if hs else None
    snap = copy.deepcopy(m) if a.arm == 'hindsight_async' else None
    buf = []; step = 0
    h = a.credit_hidden; mac_router = a.d * a.K + (a.d * a.K * a.K2 if a.depth == 2 else 0); mac_exp = a.C * a.d
    mac_q = (a.d * a.msg_dim + a.msg_dim * h if a.arm == 'hindsight_msg' else a.d * h) + h * h + h * n_exp; macs = 0.0
    qopt = torch.optim.Adam(q.parameters(), lr=a.lr) if q else None
    expert_evals = 0; hist = []
    optimizer_parameter_visits = 0
    proposal_ess_sum = 0.0; proposal_unique_sum = 0.0; proposal_examples = 0
    data_digest = hashlib.sha256()
    for tensor in (xtr, ytr, ktr, xte, yte, kte):
        data_digest.update(tensor.contiguous().numpy().tobytes())
    data_sha256 = data_digest.hexdigest()
    for ep in range(a.epochs):
        perm = torch.randperm(a.n_train, generator=g)
        for i in range(0, a.n_train, a.batch):
            ix = perm[i:i + a.batch]; x, y = xtr[ix], ytr[ix]; B = len(ix); step += 1
            lpri = m.route_logprior(x); macs += 3 * B * mac_router            # forward + backward of the router
            if a.arm == 'dense':
                loss = -torch.logsumexp(lpri + m.expert_logp(x, y), -1).mean(); expert_evals += B * n_exp; macs += 3 * B * n_exp * mac_exp
            elif a.arm == 'reinforce':
                k = torch.multinomial(lpri.exp().detach(), 1, generator=g).squeeze(-1)
                le = m.expert_logp(x, y, k); r = le.detach(); adv = r - r.mean()
                loss = -(le.mean() + (adv * lpri.gather(-1, k[:, None]).squeeze(-1)).mean()); expert_evals += B; macs += 3 * B * mac_exp
            elif a.arm == 'gumbel_st':
                gum = -torch.log(-torch.log(torch.rand(B, n_exp, generator=g).clamp_min(1e-20)))
                soft = F.softmax(lpri + gum, -1); hard = F.one_hot(soft.argmax(-1), n_exp).to(soft.dtype)
                w = hard + soft - soft.detach()
                loss = -(w * m.expert_logp(x, y)).sum(-1).mean(); expert_evals += B * n_exp; macs += 3 * B * n_exp * mac_exp
            elif a.arm == 'closed':                               # one verdict trains both sides
                S = a.proposals
                with torch.no_grad():
                    xs, (ks, ys) = x, m.sample(x, g)              # sleep anchor: the forward race's own labelled causes
                lq = q(x, y); ksel = torch.multinomial(lq.detach().exp(), S, replacement=True, generator=g)   # [B, S] proposals
                lp_s = lpri.gather(-1, ksel)
                le_s = torch.stack([m.expert_logp(x, y, ksel[:, j]) for j in range(S)], -1)
                w = F.softmax((lp_s + le_s).detach() - lq.detach().gather(-1, ksel), -1)   # forward's verdict on the proposals
                proposal_ess_sum += (1 / w.square().sum(-1)).sum().item()
                sorted_k = ksel.sort(-1).values
                proposal_unique_sum += B + (sorted_k[:, 1:] != sorted_k[:, :-1]).sum().item()
                proposal_examples += B
                loss = -(w * (lp_s + le_s)).sum(-1).mean()
                ql = -(w * lq.gather(-1, ksel)).sum(-1).mean() - q(xs, ys).gather(-1, ks[:, None]).mean()
                qopt.zero_grad(); ql.backward(); qopt.step()
                optimizer_parameter_visits += sum(p.numel() for p in q.parameters() if p.grad is not None)
                expert_evals += B * (S + 1); macs += 3 * B * S * mac_exp + B * (mac_router + mac_exp) + 3 * 2 * B * mac_q
            else:                                                 # hindsight / oracle: credit = posterior - prior
                if a.arm == 'oracle':
                    with torch.no_grad():
                        post = F.softmax(lpri + m.expert_logp(x, y), -1)
                    expert_evals += B * n_exp; macs += B * n_exp * mac_exp
                else:
                    gen_model = snap if snap is not None else m
                    if snap is not None and step % a.snap == 0:
                        snap.load_state_dict(m.state_dict())          # asynchronous teacher: refreshed only every --snap steps
                    with torch.no_grad():                         # sleep: the forward race supplies labelled causes
                        xs = x.repeat(a.sleep, 1); ks, ys = gen_model.sample(xs, g)
                    macs += len(xs) * (mac_router + mac_exp)
                    expert_evals += B * a.sleep                   # one expert per sleep sample (to draw y)
                    train_q = (a.arm != 'hindsight_slow') or (step % a.q_every == 0)
                    if a.arm == 'hindsight_async':
                        buf.append((xs, ys, ks)); buf = buf[-20:]
                        xs, ys, ks = buf[int(torch.randint(len(buf), (1,), generator=g))]
                    if train_q:
                        ql = -q(xs, ys).gather(-1, ks[:, None]).mean()
                        if a.arm == 'hindsight_hybrid':           # wake supervision: exact posterior on a fraction of real data
                            nd = max(1, int(round(a.dense_frac * B)))
                            with torch.no_grad():
                                rho = F.softmax(m.route_logprior(x[:nd]) + m.expert_logp(x[:nd], y[:nd]), -1)
                            ql = ql - (rho * q(x[:nd], y[:nd])).sum(-1).mean()
                            expert_evals += nd * n_exp; macs += nd * (mac_router + n_exp * mac_exp + 3 * mac_q)
                        qopt.zero_grad(); ql.backward(); qopt.step()
                        optimizer_parameter_visits += sum(p.numel() for p in q.parameters() if p.grad is not None)
                        macs += 3 * len(xs) * mac_q
                    with torch.no_grad():
                        post = q(x, y).exp()
                    macs += B * mac_q
                k = torch.multinomial(post, 1, generator=g).squeeze(-1)
                le = m.expert_logp(x, y, k); expert_evals += B; macs += 3 * B * mac_exp   # wake: the credited expert alone
                loss = -(le.mean() + (post * lpri).sum(-1).mean())  # router gradient = prior - posterior (descent)
            opt.zero_grad(); loss.backward(); opt.step()
            optimizer_parameter_visits += sum(p.numel() for p in m.parameters() if p.grad is not None)
        lp, pur = evaluate(m, xte, yte, kte, n_exp)
        hist.append(dict(epoch=ep, dev_marginal_ll=lp, route_purity=pur, expert_evals_per_example=expert_evals / ((ep + 1) * a.n_train), macs_per_example=macs / ((ep + 1) * a.n_train)))
        if q is not None:
            hist[-1]['credit_kl'], hist[-1]['credit_tv'] = credit_gap(m, q, xte, yte)
        seen = (ep + 1) * a.n_train
        hist[-1]['optimizer_parameter_visits_per_example'] = optimizer_parameter_visits / seen
        if proposal_examples:
            hist[-1]['proposal_ess'] = proposal_ess_sum / proposal_examples
            hist[-1]['unique_proposals'] = proposal_unique_sum / proposal_examples
        print(json.dumps(hist[-1]), flush=True)
    res = dict(status='completed', battle='ENABLER (route credit, theory note 160)', tag=a.tag, args=vars(a),
               teacher_dev_marginal_ll=teacher_ll, final_dev_marginal_ll=hist[-1]['dev_marginal_ll'], final_route_purity=hist[-1]['route_purity'],
               best_dev_marginal_ll=max(h['dev_marginal_ll'] for h in hist), history=hist,
               selection='fixed final epoch for cross-arm comparison; best DEV is diagnostic only',
               final_credit_kl=hist[-1].get('credit_kl'), final_credit_tv=hist[-1].get('credit_tv'),
               expert_evals_per_example=hist[-1]['expert_evals_per_example'], macs_per_example=hist[-1]['macs_per_example'],
               whole_fit_linear_macs=macs, wall_s=time.time() - t0,
               data_sha256=data_sha256, evaluation_split='synthetic development; no sealed benchmark test',
               mechanism_scope='hard-route credit diagnostic; no temporal memory or integrated benchmark claim',
               work_convention='leading-linear MAC estimate (3x for differentiated operations); nonlinear, sampling, bias and optimizer FLOPs excluded',
               optimizer_parameter_visits=optimizer_parameter_visits,
               optimizer_parameter_visits_per_example=optimizer_parameter_visits / (a.epochs * a.n_train),
               evaluation_linear_macs=a.epochs * a.n_test * (mac_router + n_exp * mac_exp) * (2 if q else 1) + (a.epochs * a.n_test * mac_q if q else 0),
               evaluation_work_convention='model marginal once, plus posterior and q once for credit arms; excludes teacher data generation and route-purity mapping',
               max_rss_kb=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
               hardware=dict(host=__import__('socket').gethostname(), cpu=__import__('platform').processor(), threads=torch.get_num_threads(), torch=torch.__version__),
               source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    od = ROOT / 'experiments/results/credit'; od.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res[k] for k in ('final_dev_marginal_ll', 'final_route_purity', 'expert_evals_per_example', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
