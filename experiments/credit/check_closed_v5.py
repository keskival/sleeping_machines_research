import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/experiments/credit'); import hindsight_race_v5 as H
torch.manual_seed(0); g = torch.Generator().manual_seed(0)
for depth in (1, 2):
    m = H.Net(16, 8, 10, depth, 4); x = torch.randn(4, 16); y = torch.randint(0, 10, (4,))
    # exact router gradient of the marginal likelihood
    m.zero_grad(); (-torch.logsumexp(m.route_logprior(x) + m.expert_logp(x, y), -1).sum()).backward(); ex = m.R.weight.grad.clone()
    # closed-arm estimator with a deliberately poor proposal (uniform q) and many proposals
    S = None; lq = torch.full((4, m.n_exp), -torch.log(torch.tensor(float(m.n_exp))))
    ks = torch.arange(m.n_exp).expand(4, -1); S = m.n_exp; m.zero_grad()   # enumerate every cause once: weights must equal the exact posterior
    lpri = m.route_logprior(x); lp_s = lpri.gather(-1, ks)
    with torch.no_grad(): le_all = m.expert_logp(x, y)
    le_s = le_all.gather(-1, ks)
    w = F.softmax((lp_s + le_s).detach() - lq.gather(-1, ks), -1)
    (-(w * lp_s).sum()).backward(); est = m.R.weight.grad
    err = (est - ex).abs().max().item() / ex.abs().max().item()
    print(f'depth {depth}: relative router-gradient error with enumerated proposals, S={S}: {err:.2e}', 'PASS' if err < 1e-5 else 'FAIL')
