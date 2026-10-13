"""R1 nested stream contract, with fixed external losses and no model training."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import resource
import time
from finite_horizon_receipt import HorizonReceipt
from nested_horizon_audit import NestedHorizonAudit, PairedOutcome

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True)
    args = parser.parse_args(); start = time.monotonic()
    path = ROOT / f'experiments/results/credit/{args.tag}.json'
    if path.exists(): raise FileExistsError(path)
    ids = dict(objective_id='fixed-weighted-sum', world_id='paired-stream', learner_version='cut0')
    receipt = HorizonReceipt((2,4,8),(.3,-.2,.8),(.2,.3,.5),**ids)
    weights = [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]
    differences = [-2.,-1.,4.,8.,16.,32.,64.,128.]
    outcomes = [PairedOutcome(j+1,f'token{j}',f'token{j}',150.+delta,150.,**ids)
                for j,delta in enumerate(differences)]
    values = [math.fsum(w*d for w,d in zip(weights[:h],differences[:h])) for h in receipt.horizons]
    targets = []
    for level in range(3):
        audit = NestedHorizonAudit(receipt,level,weights)
        for outcome in outcomes[:receipt.horizons[level]]: audit.append(outcome)
        expected = receipt.corrected_target(level, values[level-1] if level else 0., values[level],**ids)
        assert abs(audit.target()-expected)<1e-12
        targets.append(audit.target())
    error = abs(math.fsum(q*t for q,t in zip(receipt.probabilities,targets))-values[-1])
    assert error<1e-12 and values[0]<0<values[-1]
    audit = NestedHorizonAudit(receipt,1,weights)
    weights[0]=1000.; assert audit.weights[0]==1.
    rejected=[]
    for name,bad in [('gap',replace(outcomes[0],step=2)),
                     ('pairing',replace(outcomes[0],intervention_observation='other-token')),
                     ('version',replace(outcomes[0],learner_version='new-cut')),
                     ('nonfinite',replace(outcomes[0],reference_loss=float('inf')))]:
        try: audit.append(bad)
        except ValueError: rejected.append(name)
        else: raise AssertionError(name)
        assert audit.count==0 and audit.total==0.
    try: audit.target()
    except ValueError: rejected.append('incomplete')
    else: raise AssertionError('incomplete')
    for outcome in outcomes[:4]: audit.append(outcome)
    try: audit.append(outcomes[4])
    except ValueError: rejected.append('after-maturity')
    else: raise AssertionError('after-maturity')
    sources=['experiments/credit/finite_horizon_receipt.py','experiments/credit/nested_horizon_audit.py',
             'experiments/credit/check_nested_horizon_audit.py']
    result=dict(status='completed',battle='R1',tag=args.tag,gate='PASS',
                scope='Fixed paired-loss streaming ledger; no native rollout or fit',
                terminal_mean_error=error,nested_values=values,rejection_checks=rejected,
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__': main()
