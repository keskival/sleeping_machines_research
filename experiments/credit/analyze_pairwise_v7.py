"""Automatic matched DEV comparison; outputs a development decision, not a win."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def analyze(paths):
    records=[json.loads(p.read_text()) for p in paths]
    assert all(r['status']=='completed' for r in records)
    assert len({r['data_sha256'] for r in records})==1,'unmatched teacher/data'
    keys=('depth','K','K2','d','C','n_train','n_test','epochs','batch','seed','data_seed','lr')
    assert len({tuple(r['args'][k] for k in keys) for r in records})==1,'unmatched fitting protocol'
    dense=next(r for r in records if r['args']['arm']=='dense')
    rows=[]
    for r in records:
        rows.append(dict(tag=r['tag'],arm=r['args']['arm'],final_dev_ll=r['final_dev_marginal_ll'],
                         dev_gap_to_dense=r['final_dev_marginal_ll']-dense['final_dev_marginal_ll'],
                         credit_kl=r['final_credit_kl'],credit_tv=r['final_credit_tv'],
                         whole_fit_linear_macs=r['whole_fit_linear_macs'],linear_macs_per_target=r['macs_per_example'],
                         linear_ratio_to_dense=r['macs_per_example']/dense['macs_per_example'],
                         optimizer_parameter_visits=r['optimizer_parameter_visits'],
                         optimizer_parameter_visits_per_target=r['optimizer_parameter_visits_per_example'],
                         wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb'],last_epoch=r['history'][-1]))
    persistent=next(r for r in rows if r['arm']=='pairwise_persistent')
    decision=('Confirm on three seeds, then R1 integration with temporal/write credit retained' if persistent['dev_gap_to_dense']>=-.02 and persistent['credit_kl']<=.02 else
              'Continue development; diagnose compact-credit approximation, pair coverage and posterior-chain tracking separately')
    return dict(status='completed',battle='B1/R1 shared route-credit enabler',
                evidence_level='single-seed synthetic DEV diagnostic; no benchmark verdict',rows=rows,decision=decision,
                gates=dict(final_dev_gap_nats=-.02,credit_kl=.02),
                work_scope='modeled linear work, optimizer parameter visits separate; wall time on shared host, no energy measurement',
                inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--inputs',nargs='+',required=True);a=ap.parse_args()
    result=analyze([ROOT/p for p in a.inputs]);result['tag']=a.tag
    me=Path(__file__).resolve();result['source_sha256']={str(me.relative_to(ROOT)):hashlib.sha256(me.read_bytes()).hexdigest()}
    with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
