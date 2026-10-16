"""B4 Github TRAIN-only real longest-batch memory admission gate."""
import argparse,json,resource,sys,time,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import race_tpp_b4x2 as X


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.manual_seed(0);start=time.monotonic();model,train,K,rules=X.build('github_filtered',1,0,True);model.train();X.rep.CAPS=None
    longest=sorted(train,key=lambda row:len(row[1]))[-32:];batch=next(X.base.batches(longest,32,False,random.Random(0)))
    t,m,mask=batch;optimizer=torch.optim.AdamW(model.parameters(),lr=X.CFG['lr']);repairs=[]
    X.micro_update(model,optimizer,batch,rules['recording_cell'],repairs,0,0,2048)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    files=X.SOURCES+['experiments/tpp/b4_microbatch_gate_20261016.py','tests/test_b4_microbatch_update.py']
    import hashlib
    result=dict(status='completed',battle='B4',stage='b4_microbatch_gate',dataset='github_filtered',split=1,K=K,event_budget=2048,batch_rows=len(t),max_sequence_length=t.shape[1],padded_events=t.numel(),scored_events=int(mask[:,1:].sum()),parameters=sum(p.numel() for p in model.parameters()),repairs=repairs,wall_s=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},scope='One actual longest32-sequence TRAIN batch,native pointer actor,full within-sequence history,dropout0.3,global clipping;microbatch rows only. No DEV/TEST,quality or iso-compute claim. Deterministic no-dropout/no-target-jitter full-update parity tested separately;dropout/dequantization coupling changes while their per-example laws are retained.')
    with (ROOT/f'experiments/results/tpp_b4/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
