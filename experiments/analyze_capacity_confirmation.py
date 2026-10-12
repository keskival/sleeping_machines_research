"""R1 three-seed acquired-capacity quality and complete training-cost record."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    inputs,sources,rows={},{},[]
    for seed in (6,7,8):
        records=[]
        for variant in ('window_replay','capacity8'):
            path=ROOT/f'experiments/results/token_language/aws_integrated_{variant}_512k_20261012_s{seed}_v1.json'
            blob=path.read_bytes();d=json.loads(blob);assert d['status']=='completed'
            inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
            for source,digest in d['source_sha256'].items():
                assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
                assert source not in sources or sources[source]==digest
                sources[source]=digest
            records.append(d)
        r,c=records
        assert {k:v for k,v in r['args'].items() if k not in ('tag','pool')}=={k:v for k,v in c['args'].items() if k not in ('tag','pool')}
        assert r['identity']==c['identity'] and r['presentations_total']==c['presentations_total']==524272
        rows.append(dict(seed=seed,reference_nll=r['best_dev_nll'],candidate_nll=c['best_dev_nll'],
            paired_quality_gain=r['best_dev_nll']-c['best_dev_nll'],reference_training_s=r['train_wall_s_total'],
            candidate_training_s=c['train_wall_s_total'],training_wall_ratio=c['train_wall_s_total']/r['train_wall_s_total']))
    gains=[r['paired_quality_gain'] for r in rows];ratios=[r['training_wall_ratio'] for r in rows]
    source=str(Path(__file__).relative_to(ROOT));sources[source]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',rows=rows,input_sha256=inputs,source_sha256=sources,
        paired_quality_gain_mean=statistics.mean(gains),paired_quality_gain_sd=statistics.stdev(gains),
        training_wall_ratio_mean=statistics.mean(ratios),all_seeds_quality_improved=all(x>0 for x in gains),
        bounded_cost_gate_passed=all(x<=1.05 for x in ratios),
        scope='Three-seed own-model capacity development at524272fresh TRAIN targets/1024updates,2040DEV targets per fit. Same source/tokenizer/data/prior/update/evaluation protocol, architecture-dependent initialization differs. Four selected writes/target,16→32scored keys and receivers. Complete training wall versus saved cached pool4 fits; 5% premium gate is a development budget, not iso-compute or a Pareto claim. No external-model, publicTEST, FLOP or hardware-energy claim.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
