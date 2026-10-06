"""Toy check of THEORY §434: posterior-routed race readout = 1-particle SIS with the optimal proposal.
Model: S slots; slot state = (last type, last write time); laws from the state via fixed maps; event_terms gives
log p(x_k, a_k = s | path) = base + fire_s. Exact p(x) by enumerating all S^T assignment paths."""
import itertools, math, sys
import numpy as np, torch
sys.path.insert(0, '.')
from sleeping_machines.race_readout import event_terms
torch.manual_seed(0); rng = np.random.default_rng(0)
S, V, T = 3, 4, 7
A = torch.log_softmax(torch.randn(V + 1, V, dtype=torch.float64) * 1.5, -1)    # type law given last type (V = fresh)
MU = torch.randn(V + 1, dtype=torch.float64) * .7; LS = torch.full((V + 1,), -.3, dtype=torch.float64)
Q = torch.tensor([.9, .8, .7], dtype=torch.float64)
def laws(last):                      # last (S,) long
    return dict(logp=A[last][None], mu=MU[last][None], log_sigma=LS[last][None], logq=Q.log()[None], log1mq=(1 - Q).log()[None])
types = torch.tensor(rng.integers(0, V, T)); times = torch.tensor(np.cumsum(rng.exponential(.8, T)), dtype=torch.float64)
def step(state, k):
    last, tref, tprev = state
    lt, _, logr = event_terms(laws(last), tref[None], tprev[None], times[k:k+1], types[k:k+1], 1e-3)
    return float(lt), logr[0]
def write(state, k, s):
    last, tref, _ = state
    last = last.clone(); tref = tref.clone(); last[s] = types[k]; tref[s] = times[k]
    return (last, tref, times[k])
init = (torch.full((S,), V), torch.zeros(S, dtype=torch.float64), torch.tensor(0., dtype=torch.float64))
# exact marginal: sum over all paths of prod_k p(x_k, a_k | path)
logs = []
for path in itertools.product(range(S), repeat=T):
    st, tot = init, 0.
    for k, s in enumerate(path):
        lt, logr = step(st, k); tot += lt + float(logr[s]); st = write(st, k, s)
    logs.append(tot)
exact = float(torch.logsumexp(torch.tensor(logs), 0))
# 1-particle SIS with posterior proposal
def sis():
    st, tot = init, 0.
    for k in range(T):
        lt, logr = step(st, k); tot += lt
        s = int(torch.multinomial(logr.exp(), 1)); st = write(st, k, s)
    return tot
M = 20000
est = np.array([sis() for _ in range(M)])
z = np.exp(est - exact)
print(f'log p(x) exact {exact:.4f}; mean Zhat/p {z.mean():.4f} +- {z.std() / math.sqrt(M):.4f}; '
      f'E log Zhat {est.mean():.4f} (gap {exact - est.mean():.4f} nats); argmax-path log-lik', end=' ')
st, tot = init, 0.
for k in range(T):
    lt, logr = step(st, k); tot += lt; st = write(st, k, int(logr.argmax()))
print(f'{tot:.4f}')
