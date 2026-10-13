"""Instrumented operation ledger must preserve the ordinary learning trajectory."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    inputs={};records=[];sources={}
    for name in ('aws_native_work_reference_20261013_v1','aws_native_work_probe_20261013_v1'):
        path=ROOT/f'experiments/results/token_language/{name}.json';blob=path.read_bytes();r=json.loads(blob);assert r['status']=='completed'
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest();records.append(r);sources.update(r['source_sha256'])
    for name,digest in sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    def canon(row):return {k:v for k,v in row.items() if k not in ('wall_s','memory_effective_rank')}
    rank_errors=[]
    def flat(v):
        if isinstance(v,list):
            for x in v:yield from flat(x)
        else:yield v
    for before,after in zip(records[0]['curve'],records[1]['curve']):
        if 'memory_effective_rank' in before:
            rank_errors.extend(abs(x-y) for x,y in zip(flat(before['memory_effective_rank']),flat(after['memory_effective_rank'])))
    assert max(rank_errors,default=0.)<1e-12
    assert [canon(r) for r in records[0]['curve']]==[canon(r) for r in records[1]['curve']]
    assert records[0]['presentations_total']==records[1]['presentations_total']
    source=str(Path(__file__).relative_to(ROOT));sources[source]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',exact_instrumented_learning_curve=True,diagnostic_rank_max_error=max(rank_errors,default=0.),input_sha256=inputs,source_sha256=sources,
        accounting=records[1]['native_work_accounting'],scope='Same16-update cached integrated fit, entire recorded quality/teacher/gradient/fingerprint curve unchanged except wall and non-learning SVD memory-rank rounding(<1e-12); exact learner fingerprint remains required. Operation/payload/FLOP coverage bounds in accounting remain binding; no ASIC energy or full-fit FLOP advantage claim.')
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Exact accounting-trajectory gate PASS',flush=True)


if __name__=='__main__':main()
