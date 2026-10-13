"""Source-bound current B10 prospective interface acquisition evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
names={'unscaled':'future_interface','scaled':'future_interface_scaled','geometry':'future_interface_geometry'}
packets={name:ROOT/f'experiments/results/market/curie_b10_{stem}_20261013_v1_fit.json' for name,stem in names.items()}
results={name:json.loads(path.read_text()) for name,path in packets.items()}
reference=np.array([x['target'] for x in results['unscaled']['dev']['records']])
train=np.array([x['target'] for x in results['unscaled']['train']['records']])
rows=[]
for name,r in results.items():
    target=np.array([x['target'] for x in r['dev']['records']])
    assert np.array_equal(target,reference)
    for phase in ('train','dev'):
        assert r[phase]['blocks']==results['unscaled'][phase]['blocks']
        for path,digest in r['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    rows.append(dict(variant=name,dev_mse=r['dev']['mse'],zero_relative_reduction=1-r['dev']['mse']/r['dev']['zero_credit_mse'],
                     parameters=r['critic_parameters'],train_teacher_targets=r['train']['forward_teacher_targets'],
                     dev_teacher_targets=r['dev']['forward_teacher_targets'],wall_s=r['wall_s']))
summary=dict(status='completed',battle='B10',rows=rows,
             zero_mse=float(np.mean(reference**2)),train_mean_mse=float(np.mean((reference-train.mean(0))**2)),
             causal_previous_mse=float(np.mean((reference-np.vstack((np.zeros(4),reference[:-1])))**2)),
             retained_variant='scaled',input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in packets.values()},
             source_sha256={'experiments/credit/analyze_market_future_interface.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
             scope='One seed,32 paired DEV decision frames/four coordinates, fixed restored actor. Exact same labels across variants. Native prospective-credit prediction diagnostic; no actor-update quality, total-work or hardware advantage. Source128 TRAIN frames,8192 forward teacher targets; DEV2048 teacher targets used only for scoring. Contract derivative/finite-difference audits additionally paid.')
out=ROOT/'experiments/results/market/curie_b10_future_interface_20261013_v1_analysis.json'
with out.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(summary),flush=True)
