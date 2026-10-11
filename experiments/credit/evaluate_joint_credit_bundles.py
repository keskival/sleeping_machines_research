"""B1 evaluate saved joint fields against unlearned identity credit, no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch

from fit_joint_credit_transfer import evaluate_observed_credit, SharedProducerActor, native

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    start = time.monotonic()
    train = native.load_split('taxi', 'train')
    contexts = [(torch.tensor(t[:4])[None], torch.tensor(m[:4])[None])
                for t,m in train if len(m) >= 4][:3]
    sources = {'experiments/credit/evaluate_joint_credit_bundles.py'}
    inputs = {}; rows = []
    for seed in [0, 1, 2]:
        parent = ROOT/'experiments/results/credit'/f'aws_joint_credit_transfer_20261011_s{seed}.json'
        record = json.loads(parent.read_text()); assert record['status'] == 'completed'
        for source,digest in record['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest
            sources.add(source)
        inputs[str(parent.relative_to(ROOT))] = hashlib.sha256(parent.read_bytes()).hexdigest()
        weights = ROOT/record['final_weights']
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == record['checkpoint_sha256']
        inputs[str(weights.relative_to(ROOT))] = record['checkpoint_sha256']
        bundle = torch.load(weights, weights_only=True)
        for depth in [2, 4, 8]:
            saved = bundle[str(depth)]
            actor = SharedProducerActor(*saved['actor_args']).eval()
            actor.load_state_dict(saved['actor'])
            W = saved['estimator']['weight']
            field = torch.nn.Linear(W.shape[1], W.shape[0], bias=False)
            field.load_state_dict(saved['estimator'])
            identity = torch.nn.Linear(W.shape[1], W.shape[0], bias=False)
            with torch.no_grad():
                identity.weight.zero_();identity.weight[:W.shape[1]].copy_(torch.eye(W.shape[1]))

            def evaluate(times,marks,name):
                learned = evaluate_observed_credit(actor,field,times,marks,name)
                control = evaluate_observed_credit(actor,identity,times,marks,name)
                assert abs(learned['observed_loss']-control['observed_loss']) < 1e-10
                zero = learned['producer_zero_error']
                rows.append(dict(seed=seed,depth=depth,condition=name,
                                 learned_producer_risk=learned['producer_credit_error'],
                                 identity_producer_risk=control['producer_credit_error'],
                                 zero_producer_risk=zero,
                                 learned_gain_vs_identity=control['producer_credit_error']-learned['producer_credit_error'],
                                 residual_reduction_vs_identity=1-learned['producer_credit_error']/control['producer_credit_error'],
                                 learned_gain_vs_zero=zero-learned['producer_credit_error'],
                                 learned_joint_risk=learned['joint_credit_error'],identity_joint_risk=control['joint_credit_error']))
                return learned

            original = evaluate(*contexts[0],'same-context')
            prior = next(x for x in record['rows'] if x['depth']==depth)['same_context_observed_credit']
            assert abs(original['producer_credit_error']-prior['producer_credit_error']) < 1e-10
            for index,context in enumerate(contexts[1:]):
                evaluate(*context,f'new-context-{index+1}')
            actor.zero_grad();tl,ml,n,_=actor.loglik(*contexts[0],torch.ones_like(contexts[0][1],dtype=torch.bool))
            (-(tl+ml)/n).backward()
            optimizer=torch.optim.Adam(actor.parameters(),lr=.001);optimizer.step()
            evaluate(*contexts[0],'one-Adam-model-update')
    result = dict(status='completed',tag=args.tag,battle='B1 actual-loss local-credit transfer',rows=rows,
                  source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
                  input_sha256=inputs,wall_s=time.monotonic()-start,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Saved fields/actors restored without refitting;3seeds depths2/4/8, same/two new four-event TRAIN contexts/one Adam update. Identity content credit and zero time credit is an unlearned local control. Actual loss identical between arms; only credit residual energy changes. Expired learned maps tested diagnostically, no sparse-audit operation, future task-learning, full-fit efficiency, route or Transformer claim.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:
        json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
