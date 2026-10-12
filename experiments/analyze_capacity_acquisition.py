"""R1 pool-size development comparison at two fresh-acquisition sizes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    inputs,sources,levels={},{},[]
    for size in ('128k','512k'):
        rows=[];records=[]
        for variant in ('window_replay','capacity8'):
            path=ROOT/f'experiments/results/token_language/aws_integrated_{variant}_{size}_20261012_s6_v1.json'
            blob=path.read_bytes();d=json.loads(blob);assert d['status']=='completed'
            inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
            for source,digest in d['source_sha256'].items():
                assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
                assert source not in sources or sources[source]==digest
                sources[source]=digest
            records.append(d)
            rows.append(dict(pool=d['args']['pool'],targets=d['presentations_total'],parameters=d['parameters'],
                selected_nll=d['best_dev_nll'],final_nll=d['curve'][-1]['dev_nll'],
                complete_training_s=d['train_wall_s_total'],complete_process_s=d['process_wall_s'],
                training_ms_per_target=1000*d['train_wall_s_total']/d['presentations_total'],
                scored_keys_per_target=d['args']['pool']*d['args']['heads']*d['args']['depth'],
                selected_writes_per_target=d['args']['heads']*d['args']['depth'],
                final_routing=d['curve'][-1]['routing']))
        r,c=records
        assert {k:v for k,v in r['args'].items() if k not in ('tag','pool')}=={k:v for k,v in c['args'].items() if k not in ('tag','pool')}
        assert r['identity']==c['identity'] and r['presentations_total']==c['presentations_total']
        gain=r['best_dev_nll']-c['best_dev_nll'];ratio=c['train_wall_s_total']/r['train_wall_s_total']
        levels.append(dict(exposure=size,rows=rows,pool8_selected_gain=gain,training_wall_ratio=ratio,
            quality_time_pareto_improvement=gain>0 and ratio<=1))
    source=str(Path(__file__).relative_to(ROOT));sources[source]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',levels=levels,input_sha256=inputs,source_sha256=sources,
        scope='First-pass own-model pool8 development at two acquired-data sizes, seed6. Same proper tokens, TRAIN prior, updates and DEV opportunities; architecture-dependent initialization differs. Selected writes unchanged, scored keys and core receivers doubled. Full factual BP and sampled actual alternative replay retained. Complete CPU training/process wall, no FLOP/energy/external-model or family verdict. Small quality deficit did not veto the reserved larger point.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(levels),flush=True)


if __name__=='__main__':main()
