"""Matched-teacher DEV analysis for the B1/R1 route-credit enabler."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def analyze(paths):
    records = [json.loads(p.read_text()) for p in paths]
    assert all(r['status'] == 'completed' for r in records)
    assert len({r['data_sha256'] for r in records}) == 1, 'unmatched data'
    keys = ('depth','K','K2','d','C','n_train','n_test','epochs','batch','data_seed','seed','lr')
    assert all(tuple(r['args'][k] for k in keys) == tuple(records[0]['args'][k] for k in keys) for r in records)
    dense = next(r for r in records if r['args']['arm']=='dense')
    rows = []
    for r in records:
        final = r['history'][-1]
        rows.append(dict(tag=r['tag'],arm=r['args']['arm'],credit_hidden=r['args']['credit_hidden'],
                         proposals=r['args']['proposals'], final_dev_ll=r['final_dev_marginal_ll'],
                         dev_ll_gap_to_dense=r['final_dev_marginal_ll']-dense['final_dev_marginal_ll'],
                         credit_kl=r['final_credit_kl'],credit_tv=r['final_credit_tv'],
                         linear_mac_ratio_to_dense=r['macs_per_example']/dense['macs_per_example'],
                         whole_fit_linear_macs=r['macs_per_example']*r['args']['epochs']*r['args']['n_train'],
                         linear_macs_per_target=r['macs_per_example'],
                         optimizer_parameter_visits=r['optimizer_parameter_visits'],
                         optimizer_parameter_visits_per_target=r['optimizer_parameter_visits_per_example'],
                         evaluation_linear_macs=r['evaluation_linear_macs'],wall_s=r['wall_s'],
                         max_rss_kb=r['max_rss_kb'],proposal_ess=final.get('proposal_ess'),
                         unique_proposals=final.get('unique_proposals')))
    small=next(r for r in rows if r['arm']=='closed' and r['credit_hidden']==8 and r['proposals']==2)
    if small['dev_ll_gap_to_dense'] >= -0.02 and small['linear_mac_ratio_to_dense'] < 1:
        decision='Confirm compact closed credit on three seeds; then integrate in R1. Linear-work gate only; full work remains to audit.'
    else:
        decision='Continue development: inspect posterior gap, proposal support and gradient bias before confirmation or integration.'
    return dict(status='completed',evidence_level='single-seed synthetic development diagnostic',
                battle='B1/R1 shared route-credit enabler',data_sha256=records[0]['data_sha256'],
                rows=rows,decision=decision, quality_gate_nats=0.02,
                work_scope=records[0]['work_convention'],
                inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); ap.add_argument('--inputs',nargs='+',required=True)
    a=ap.parse_args(); res=analyze([ROOT/p for p in a.inputs]); res['tag']=a.tag
    me=Path(__file__).resolve(); res['source_sha256']={str(me.relative_to(ROOT)):hashlib.sha256(me.read_bytes()).hexdigest()}
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json'
    with out.open('x') as f: json.dump(res,f,indent=2); f.write('\n')
    print(json.dumps(res,indent=2))
