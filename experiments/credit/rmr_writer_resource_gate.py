"""Real integrated RMR resource gate with live first-order writer-input credit.

Uses the already product-ordered 16-update integration workload. No long fit
is admitted by this gate, and stopped/source-bound trainers stay unchanged.
"""
import argparse,hashlib,io,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import torch
import experiments.r1_rmr as driver
from experiments.r1_rmr_writer_input_credit import CheckpointedWriterInputCreditRMR

class Tee:
    def __init__(self,output):self.output=output;self.record=io.StringIO()
    def write(self,text):self.output.write(text);self.record.write(text);return len(text)
    def flush(self):self.output.flush()

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args();torch.set_num_threads(1);started=time.monotonic()
    names=['experiments/credit/rmr_writer_resource_gate.py','experiments/r1_rmr_writer_input_credit.py','experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    original_clip=torch.nn.utils.clip_grad_norm_
    def finite_clip(*a,**kw):kw['error_if_nonfinite']=True;return original_clip(*a,**kw)
    torch.nn.utils.clip_grad_norm_=finite_clip;driver.RMR=CheckpointedWriterInputCreditRMR
    sys.argv=[driver.__file__,'--tag',args.tag,'--levels','4','--top-m','4','--read-topk','0','--old-decay','0.01','--train-tokens','16777216','--seed','0','--wd-groups','1','--unit-init','0.02','--balance','0.01','--old-per-target','1','--write-race','1','--write-explore','2','--carry','256','--max-steps','16','--checkpoint-every','16','--log-every','8']
    tee=Tee(sys.stdout);old_stdout=sys.stdout;sys.stdout=tee
    try:driver.main()
    finally:sys.stdout=old_stdout;torch.nn.utils.clip_grad_norm_=original_clip
    rows=[]
    for line in tee.record.getvalue().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    def finite(value):
        if isinstance(value,float):assert math.isfinite(value),'nonfinite gate value'
        elif isinstance(value,dict):
            for item in value.values():finite(item)
        elif isinstance(value,list):
            for item in value:finite(item)
    finite(rows);smoke=next(row for row in rows if row.get('smoke_steps')==16)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert peak<2900000,'writer-input VJP requires a reduced-concurrency reservation'
    result=dict(status='completed',battle='R1',stage='live writer-input integrated resource gate',gate='PASS',source_sha256=hashes,measurements=rows,updates=16,actual_train_targets=16*4*255,elapsed_s=time.monotonic()-started,max_rss_kb=peak,checkpoint=f'experiments/results/token_language/{args.tag}.resume.pt',scope='Integrated per-target writer, write-address probes, carried memory and full first-order input VJP. Bounded 16-update resource/correctness gate only; no quality or long-history scalability claim. Extra autograd work is paid.')
    (ROOT/f'experiments/results/diagnostics/{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
