"""Independent dense-vocabulary frozen-snapshot oracle for carried credit."""
import argparse, hashlib, json, resource, time
import torch
from torch.nn import functional as F
from check_token_forward_credit import ROOT, KeyedTokenLM
from token_forward_credit import packet, bonuses
from token_static_carried_credit import static_carried_scores


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    args = ap.parse_args(); torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64); start = time.monotonic()
    out = ROOT / f'experiments/results/credit/{args.tag}.json'
    assert not out.exists(); checks = []
    for depth in [2, 4, 8]:
        for reset in [False, True]:
            torch.manual_seed(7910 + depth)
            model = KeyedTokenLM(8, 3, depth, 2, 4, 1, 0.)
            model.embed = torch.nn.Embedding(23, 8)
            model.bias = torch.nn.Parameter(torch.randn(23) * .02)
            model.requires_grad_(False); model.eval()
            tokens = torch.tensor([2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8])
            if reset: tokens[7] = 7
            p = packet(model, tokens, index=3, eos_token=7)
            z = torch.randn(8) * .15; before = z.clone()
            actions = torch.zeros(3, 2, 4); actions[1, 0, 1] = .25
            actions[2, 1, 2] = -.25; receivers = [2, 4, 22]
            actual = static_carried_scores(p, receivers, actions, z)
            beta = bonuses(p, actions); truth = []
            frozen = z.clone()
            for j, target in enumerate(p.targets.tolist()):
                if bool(p.reset[j]): frozen.zero_()
                logits = p.log_probabilities[j] + p.embedding @ frozen
                truth.append(torch.stack([
                    F.cross_entropy((logits + F.one_hot(torch.tensor(r), 23)
                                     * beta[i, j])[None], torch.tensor([target]))
                    for i, r in enumerate(receivers)]))
            error = float((actual - torch.stack(truth, 1)).abs().max())
            assert error < 1e-12 and torch.equal(z, before)
            checks.append(dict(depth=depth, reset=reset, error=error))
    sources = ['experiments/credit/check_token_static_carried_credit.py',
               'experiments/credit/token_static_carried_credit.py',
               'experiments/credit/token_forward_credit.py']
    result = dict(status='completed', battle='R1', tag=args.tag, gate=True,
                  checks=checks, whole_wall_s=time.monotonic()-start,
                  peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                  scope='Nonzero issued fast-state snapshot, independent full-vocabulary CE oracle, no-op and two hard writes, depths2/4/8 with causal EOS reset. Incoming state unmodified; no actor fitting or quality claim.')
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
