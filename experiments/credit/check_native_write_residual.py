"""R1 non-fitting bridge: paired credit over full native addressed-write replays.

The existing R1 model has deterministic writes. This diagnostic adds one
declared three-way write choice (original slot, another slot, silence), without
changing the event message, clock time, likelihood or any production driver.
It certifies an expected-loss route component, not marginal-posterior learning.
Run only through the battle's guarded queue.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from recall_tpp_v4 import KeyedRaceTPP
import torch
import torch.nn.functional as F


class ForcedWriteProbe(KeyedRaceTPP):
    """Replace one addressed key AND value commit; leave incoming content fixed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site = None
        self.destination = None

    def encode(self, t, m, mask):
        h, slots = super().encode(t, m, mask)
        if self.site is None:
            return h, slots
        j = self.site
        if not 0 <= j < m.shape[1] or m.shape[0] != 1:
            raise ValueError('This finite diagnostic requires one sequence and a valid site')
        source = F.one_hot(m[:, j], self.K).to(h.dtype)
        destination = torch.zeros_like(source) if self.destination is None else F.one_hot(
            torch.full_like(m[:, j], self.destination), self.K).to(h.dtype)
        address_delta = destination - source
        xk = h.detach() if self.local else h
        predecessor = self.embed(m[:, j-1]) if j else torch.zeros_like(h[:, j])
        kin = torch.cat([xk[:, j], predecessor.detach() if self.local else predecessor], -1)
        key = self.key_w(kin)
        if self.qk_norm:
            key = F.normalize(key, dim=-1)
        value = F.softplus(self.marks.value(h[:, j]))
        elapsed = ((t - t[:, j:j+1]) / self.scale).clamp_min(0)
        suffix = (torch.arange(m.shape[1], device=m.device) >= j).to(h.dtype)
        key_retention = torch.exp(-elapsed[..., None]*self.key_log_rate.exp())*suffix[None, :, None]
        value_retention = torch.exp(-elapsed[..., None]*self.marks.log_rate.exp())*suffix[None, :, None]
        self._keyed = self._keyed + (address_delta[:, None, :, None] *
                                    key[:, None, None, :] * key_retention[:, :, None, :])
        slots = slots + address_delta[:, None, :, None]*value[:, None, None, :]*value_retention[:, :, None, :]
        return h, slots


def losses(model, t, m):
    mask = torch.ones_like(m, dtype=torch.bool)
    _, _, _, (_, _, params, gap, valid) = model.loglik(t, m, mask)
    log_h, log_s = model.clock_terms(gap, *params[:4])
    observed = params[4].gather(-1, m[:, 1:, None, None].expand(-1, -1, model.M, 1)).squeeze(-1)
    # Exact time + mark likelihood, including silence through survival.
    return -(torch.logsumexp(log_h + observed, -1) + log_s.sum(-1))*valid


def flat_gradient(value, parameters):
    gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)])


def checks():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(162)
    model = ForcedWriteProbe(6, 8, 3, 2, 2, 2, 2, 0., 1., [-1., 0.],
                             dk=4, local=False, prev_msg=True, qk_norm=False)
    model.eval()
    t = torch.tensor([[0., .2, 1.1, 1.3, 4., 4.2, 8., 8.2]])
    m = torch.tensor([[0, 3, 1, 4, 0, 3, 1, 4]])
    site = 1
    original = losses(model, t, m)
    model.site, model.destination = site, int(m[0, site])
    parity = float((losses(model, t, m)-original).abs().max())
    assert parity < 1e-12
    destinations = [int(m[0, site]), 5, None]
    branches = []
    prefix_error = 0.
    for destination in destinations:
        model.destination = destination
        per_event = losses(model, t, m)
        prefix_error = max(prefix_error, float((per_event[:, :site]-original[:, :site]).abs().max()))
        branches.append(per_event[:, site:].sum())
    assert prefix_error < 1e-12
    utility = torch.stack(branches)
    assert float(utility.max()-utility.min()) > 1e-6, 'Witness must distinguish the complete writes'

    # Router reads only the same causal prefix, before the optional commit.
    model.site = None
    prefix_h, _ = model.encode(t[:, :site+1], m[:, :site+1], torch.ones_like(m[:, :site+1], dtype=torch.bool))
    router = torch.nn.Linear(8, 3)
    logits = router(prefix_h[:, -1]).squeeze(0)
    pi = logits.softmax(-1)
    log_pi = logits.log_softmax(-1)
    parameters = list(model.parameters())+list(router.parameters())
    scores = torch.stack([flat_gradient(lp, parameters) for lp in log_pi])
    p, f = pi.detach(), utility.detach()
    target = flat_gradient((pi*utility.detach()).sum(), parameters)
    expected_full = flat_gradient((pi*utility).sum(), parameters)
    path = sum(p[i]*flat_gradient(branches[i], parameters) for i in range(3))
    decomposition_error = float((expected_full-target-path).abs().max())
    assert decomposition_error < 1e-11

    def enumerate_predictor(b):
        base = ((p*b)[:, None]*scores).sum(0)
        mean = torch.zeros_like(target)
        variance = 0.
        conditional_error = 0.
        for w in range(3):
            conditional = torch.zeros_like(target)
            for i in range(3):
                if i == w:
                    continue
                q = .8*p[i]/(1-p[w])+.1
                residual = (f[i]-f[w])-(b[i]-b[w])
                estimate = base + p[i]/q*residual*scores[i]
                conditional += q*estimate
                mean += p[w]*q*estimate
                variance += float(p[w]*q*((estimate-target)**2).sum())
            conditional_error = max(conditional_error, float((conditional-target).abs().max()))
        return dict(mean_error=float((mean-target).abs().max()),
                    conditional_mean_error=conditional_error, variance_trace=variance)

    arbitrary = torch.tensor([3., -2., 1.])
    poor = enumerate_predictor(arbitrary)
    better = enumerate_predictor(.75*f+.25*arbitrary+7.)
    perfect = enumerate_predictor(f+17.)
    for record in (poor, better, perfect):
        assert max(record['mean_error'], record['conditional_mean_error']) < 1e-11
    assert abs(better['variance_trace']-poor['variance_trace']/16) < 1e-10
    assert perfect['variance_trace'] < 1e-22
    assert float(target.abs().max()) > 1e-6
    assert float(target[:-sum(p.numel() for p in router.parameters())].abs().max()) > 1e-6
    # A changed future mark cannot alter the route scores at this write.
    changed = m.clone(); changed[:, site+1:] = (changed[:, site+1:]+1)%model.K
    changed_h, _ = model.encode(t, changed, torch.ones_like(changed, dtype=torch.bool))
    causal_error = float((router(changed_h[:, site])-logits).abs().max())
    assert causal_error < 1e-12
    return dict(original_write_parity_error=parity, prefix_likelihood_error=prefix_error,
                future_route_causality_error=causal_error,
                full_gradient_decomposition_error=decomposition_error,
                branch_suffix_losses=f.tolist(), probabilities=p.tolist(),
                arbitrary_predictor=poor, improved_predictor=better, perfect_predictor=perfect,
                gradient_parameters=sum(p.numel() for p in parameters), scored_suffix_events=t.shape[1]-site-1,
                scope='Initialized two-layer native R1 model; one declared optional key/value write, '
                      'all forced suffixes, exact conditional expected-loss route component. No fit, '
                      'no benchmark quality, no posterior or optimizer claim.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True); args = parser.parse_args()
    destination = ROOT/'experiments/results/credit'/f'{args.tag}.json'
    if destination.exists():
        raise FileExistsError(destination)
    start = time.monotonic()
    evidence = checks()
    sources = [Path(__file__).resolve(), ROOT/'experiments/tpp/recall_tpp_v4.py', ROOT/'experiments/tpp/race_tpp_v5.py']
    record = dict(status='completed', battle='R1/B1', tag=args.tag, training=False,
                  decision='Admit or reject full native write residual credit before an integrated DEV fit',
                  metrics=evidence, wall_s=time.monotonic()-start,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp'); temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(destination)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
