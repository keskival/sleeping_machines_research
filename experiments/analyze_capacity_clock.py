"""R1 completed cardinality-clock development against two saved members."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    inputs,sources,levels={},{},[]
    for size in ('128k','512k'):
        records=[];rows=[]
        for variant in ('window_replay','capacity8','clock_capacity8'):
            path=ROOT/f'experiments/results/token_language/aws_integrated_{variant}_{size}_20261012_s6_v1.json'
            blob=path.read_bytes();d=json.loads(blob);assert d['status']=='completed'
            inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
            for source,digest in d['source_sha256'].items():
                assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
                assert source not in sources or sources[source]==digest
                sources[source]=digest
            records.append(d)
            rows.append(dict(variant=variant,pool=d['args']['pool'],targets=d['presentations_total'],
                selected_nll=d['best_dev_nll'],final_nll=d['curve'][-1]['dev_nll'],
                training_s=d['train_wall_s_total'],training_ms_per_target=1000*d['train_wall_s_total']/d['presentations_total'],
                parameters=d['parameters'],scored_keys_per_target=d['args']['pool']*d['args']['heads']*d['args']['depth'],
                selected_writes_per_target=d['args']['heads']*d['args']['depth']))
        incumbent,raw,calibrated=records
        for d in (incumbent,raw):
            assert {k:v for k,v in d['args'].items() if k not in ('tag','pool','clock_reference_pool')}=={k:v for k,v in calibrated['args'].items() if k not in ('tag','pool','clock_reference_pool')}
            for key in ('train_sha256','dev_sha256'):assert d['identity'][key]==calibrated['identity'][key]
            assert d['presentations_total']==calibrated['presentations_total']
        assert calibrated['args']['clock_reference_pool']==4
        levels.append(dict(exposure=size,rows=rows,
            calibrated_gain_over_pool4=incumbent['best_dev_nll']-calibrated['best_dev_nll'],
            calibrated_gain_over_raw8=raw['best_dev_nll']-calibrated['best_dev_nll'],
            training_ratio_to_pool4=calibrated['train_wall_s_total']/incumbent['train_wall_s_total']))
    source=str(Path(__file__).relative_to(ROOT));sources[source]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',levels=levels,input_sha256=inputs,source_sha256=sources,
        scope='First-pass seed6 own-model development at two acquired-data sizes. Same proper-token data, TRAIN priors, updates, credit16 and DEV opportunities. Raw8/calibrated8 initialize the same model parameters; calibrated8 changes computational clocks using reference pool4. Pool4 initialization differs with topology. Selected writes4/target, scored keys16or32. Complete measured CPU training wall; no external-model, iso-compute, FLOP, energy or family verdict. Small quality did not veto the registered larger point.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(levels),flush=True)


if __name__=='__main__':main()
