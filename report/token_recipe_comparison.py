"""Completed bounded learning-recipe comparison, separate from scaling cells."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def pages():
    folder=ROOT/'experiments/results/token_language'
    tags=['curie_data_growth_tokens_256k_b64_c16_p24_s6_20261005_v1','curie_data_growth_tokens_256k_b64_c16_p24_lr001_s6_20261005_v1']
    records=[];rows=[]
    for tag in tags:
        path=folder/(tag+'.json');selection=path.with_suffix('.selection.json')
        if not path.exists() or not selection.exists():return []
        r=json.loads(path.read_text());s=json.loads(selection.read_text())
        if r['status']!='completed' or s['status']!='completed':return []
        assert r['presentations_total']==524272 and r['args']['steps']==1024
        assert s['selected']['dev_nll']==min(row['dev_nll'] for row in r['curve'])
        chosen=s['selected'];final=r['curve'][-1]['dev_nll']
        rows.append([str(r['args']['lr']),f"{chosen['dev_nll']:.6f}",str(chosen['step']),f'{final:.6f}',f"{final-chosen['dev_nll']:.6f}",str(r['presentations_total'])])
        records.append(r)
    a,b=records
    assert a['identity']==b['identity']
    assert {k:v for k,v in a['args'].items() if k not in ('tag','lr')}=={k:v for k,v in b['args'].items() if k not in ('tag','lr')}
    return [[('h1','Appendix. Tokenized256K: bounded learning-rate comparison'),
        ('p','P24seed6, same data/source/initialization, 524272fitting targets over two passes,1024updates,four-checkpoint cadence and2040development targets. Only constant learning rate changes; all core mechanisms retained. Public validation untouched. This tuning comparison is separate from the fixed-recipe scaling packet.'),
        ('table',(['LR','Selected NLL','Selected step','Final NLL','Final − selected','Fit targets'],rows,[45,105,95,100,120,90])),
        ('p','Lower rate improves selected NLL0.000900 and final NLL0.066162. Both select step512, the first-pass boundary; late deterioration falls0.126195→0.060933. The bounded comparison reduces late deterioration without materially changing selected quality in this seed. Retain the fixed0.003recipe for the crossed scaling comparisons; no expanded rate grid or second-pass cure is inferred.'),
        ('p','Both fits charge all524272fitting targets. Actual complete256Kwork is pending for each trajectory; no cost is borrowed from64Kor from the other rate. Ordinary measured throughput384.66/495.85targets/s reflects separate host executions and is not a causal speed effect of learning rate.')]]
