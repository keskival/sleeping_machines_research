"""Admit B10 engineering gates, then gate-bound family/seed fits, without preemption."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
ORDER=ROOT/'experiments/queue/aws_b10_matched_controls_20261014_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['synthetic','data','fits'],default='synthetic');a=ap.parse_args()
    tag='aws_b10_matched_'+a.stage+'_20261014_v1'
    paths=list((ROOT/'experiments/market').glob('b10_matched*.py'))+[ROOT/'experiments/market/b10_operator_ledger.py',ROOT/'experiments/market/b10_depth_tpp_v1.py',ROOT/'experiments/market/prepare.py',ROOT/'experiments/tpp/race_tpp_v5.py',ROOT/'experiments/market/B10_AWS_MATCHED_CONTROLS.md']
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    jobs=[]
    if a.stage=='fits':
        admission='experiments/results/market/aws_b10_matched_admission_20261014_v1.json'
        receipt=json.loads((ROOT/admission).read_text());assert receipt['status']=='completed' and receipt['training_admitted']
        for family in ('native','lstm','transformer'):
            for seed in (0,1,2):
                name=f'aws_b10_matched_{family}_s{seed}_20261014_v1'
                jobs.append((name,'experiments/market/b10_matched_fit.py',f'--family {family} --seed {seed} --admission {admission}',receipt['timeouts_s'][family]))
    else:
        jobs=[(tag,'experiments/market/b10_matched_pipeline.py','--mode '+('synthetic' if a.stage=='synthetic' else 'gate'),14400 if a.stage=='data' else 1800)]
    slots={}
    for index,(name,driver,args,timeout) in enumerate(jobs):
        folder=ORDER/name;folder.mkdir(exist_ok=False);q=folder/(name+'.txt')
        q.write_text('# B10: common-input/head matched-arithmetic comparison; decision: admit fair DEV comparison after numerical gates\n'+f'{name} {driver} --tag {name} {args}\n')
        job=dict(name=name,kind='b10_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/market/{name}.json',requires=[],timeout_s=timeout,rss_kb=3000000,vms_kb=12000000)
        slot=(4 if a.stage=='synthetic' else 5) if len(jobs)==1 else index%6+1;slots.setdefault(str(slot),[]).append(job)
    packet=dict(host='ip-172-31-47-132',battle='B10',purpose='Founder-authorized native/LSTM/Transformer matched complete-work market comparison',parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots)
    addendum=PLAN/'addenda'/f'zzzzzzzzzzzzzzzzzzzzzzzzzzzzb10_{time.time_ns()}.json';addendum.write_text(json.dumps(packet,indent=2)+'\n')
    order=json.loads((ORDER/'work_order.json').read_text());order.update(status='engineering_gates_runtime_queued',implementation=[str(p.relative_to(ROOT)) for p in paths],runtime_addendum=str(addendum.relative_to(ROOT)),training_admitted=a.stage=='fits');order.pop('reason_not_runtime_addendum',None)
    (ORDER/'work_order.json').write_text(json.dumps(order,indent=2)+'\n');print(str(addendum.relative_to(ROOT)))
if __name__=='__main__':main()
