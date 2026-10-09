#!/usr/bin/env python3
"""v3 (9 Oct, founder direction: the forward pass should give the backward pass more than activations, and what it gives
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
                    choices=['dense', 'reinforce', 'gumbel_st', 'hindsight', 'oracle', 'hindsight_async', 'hindsight_slow', 'hindsight_msg'])
    ap.add_argument('--depth', type=int, default=1); ap.add_argument('--K', type=int, default=8); ap.add_argument('--K2', type=int, default=4)
    ap.add_argument('--d', type=int, default=16); ap.add_argument('--C', type=int, default=10)
    ap.add_argument('--n-train', type=int, default=20000); ap.add_argument('--n-test', type=int, default=5000)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--sleep', type=int, default=1, help='credit-model samples per example')
    ap.add_argument('--msg-dim', type=int, default=4)
    ap.add_argument('--snap', type=int, default=50); ap.add_argument('--q-every', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--data-seed', type=int, default=1234)
    a = ap.parse_args(); t0 = time.time(); torch.set_num_threads(1)
    gd = torch.Generator().manual_seed(a.data_seed)
    xtr, ytr, ktr, xte, yte, kte, teacher_ll = make_data(a, gd)
    torch.manual_seed(a.seed); g = torch.Generator().manual_seed(a.seed)
    m = Net(a.d, a.K, a.C, a.depth, a.K2); n_exp = m.n_exp
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    import copy
    hs = a.arm.startswith('hindsight')
    q = Credit(a.d, a.C, n_exp, msg=a.msg_dim if a.arm == 'hindsight_msg' else 0) if hs else None
    snap = copy.deepcopy(m) if a.arm == 'hindsight_async' else None
    buf = []; step = 0
    h = 64; mac_router = a.d * a.K + (a.d * a.K * a.K2 if a.depth == 2 else 0); mac_exp = a.C * a.d
    mac_q = (a.d * a.msg_dim + a.msg_dim * h if a.arm == 'hindsight_msg' else a.d * h) + h * h + h * n_exp; macs = 0.0
    qopt = torch.optim.Adam(q.parameters(), lr=a.lr) if q else None
    expert_evals = 0; hist = []; best_lp = -1e9
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
                        ql = -q(xs, ys).gather(-1, ks[:, None]).mean(); qopt.zero_grad(); ql.backward(); qopt.step()
                        macs += 3 * len(xs) * mac_q
                    with torch.no_grad():
                        post = q(x, y).exp()
                    macs += B * mac_q
                k = torch.multinomial(post, 1, generator=g).squeeze(-1)
                le = m.expert_logp(x, y, k); expert_evals += B; macs += 3 * B * mac_exp   # wake: the credited expert alone
                loss = -(le.mean() + (post * lpri).sum(-1).mean())  # router gradient = prior - posterior (descent)
            opt.zero_grad(); loss.backward(); opt.step()
        lp, pur = evaluate(m, xte, yte, kte, n_exp)
        hist.append(dict(epoch=ep, test_marginal_ll=lp, route_purity=pur, expert_evals_per_example=expert_evals / ((ep + 1) * a.n_train), macs_per_example=macs / ((ep + 1) * a.n_train)))
        print(json.dumps(hist[-1]), flush=True)
    res = dict(status='completed', battle='ENABLER (route credit, theory note 160)', tag=a.tag, args=vars(a),
               teacher_test_marginal_ll=teacher_ll, final_test_marginal_ll=hist[-1]['test_marginal_ll'], final_route_purity=hist[-1]['route_purity'],
               best_test_marginal_ll=max(h['test_marginal_ll'] for h in hist), history=hist,
               expert_evals_per_example=hist[-1]['expert_evals_per_example'], macs_per_example=hist[-1]['macs_per_example'], wall_s=time.time() - t0,
               source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    od = ROOT / 'experiments/results/credit'; od.mkdir(parents=True, exist_ok=True)
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res[k] for k in ('final_test_marginal_ll', 'final_route_purity', 'expert_evals_per_example', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
