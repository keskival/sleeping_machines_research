"""B1 matched acquired-core controls for new event credit; exact replay parity."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
import run_native_batched_event_optimizer as batched
import run_native_stable_event_optimizer as stable
ROOT=batched.ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();train=[s for s in batched.native.load_split('taxi','train') if len(s[1])>=32];dev=[s for s in batched.native.load_split('taxi','dev') if len(s[1])>=32][:32];gaps=np.concatenate([np.diff(t[:32]) for t,_ in train[:32]]);gaps=gaps[gaps>0];scale=float(np.median(gaps));qs=np.log(np.quantile(gaps,[.1,.5,.9])).tolist();K=1+max(int(m.max()) for _,m in train);parents={};rows=[]
 for variant,pipeline in [('batched',batched),('stable',stable)]:
  for width in [32,16]:
   path=ROOT/f'experiments/results/credit/curie_native_{variant}_event_optimizer_20261012_v1_d{width}_fit.json';parent=json.loads(path.read_text());assert parent['status']=='completed';parents[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
   for name,digest in parent['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
   pipeline.RUN_WIDTH=width
   def evaluate(actor):
    actor.cut=False;actor.collect_events=False
    with torch.no_grad():values=[float(actor(*pipeline.one_block(s))) for s in dev]
    return dict(dev_nll=float(np.mean(values)),per_sequence=values)
   for depth in [4,8]:
    previous=next(r for r in parent['rows'] if r['depth']==depth);actor=pipeline.initial(99910,depth,K,scale,qs);field=pipeline.new_credit(actor,depth);acquired,_=pipeline.learn(actor,field,train[256:384],'online',99911);begin=evaluate(actor);assert abs(acquired['mean_nll']-previous['arms']['online']['acquisition']['mean_nll'])<1e-10;assert abs(begin['dev_nll']-previous['arms']['online']['acquisition_dev']['dev_nll'])<1e-10
    core=torch.cat([p.detach().flatten().clone() for n,p in actor.named_parameters() if pipeline.core_name(n)]);heads=[p for n,p in actor.named_parameters() if not pipeline.core_name(n)];losses=[];evaluations=[];tick=time.monotonic()
    for index,sequence in enumerate(train[384:896]):
     actor.cut=True;actor.collect_events=False;loss=actor(*pipeline.one_block(sequence));gs=torch.autograd.grad(loss,heads);norm=torch.stack([g.square().sum() for g in gs]).sum().clamp_min(1e-24).sqrt();factor=(5/norm).clamp_max(1);actor.head_optimizer.zero_grad()
     for p,g in zip(heads,gs):p.grad=(factor*g).detach()
     actor.head_optimizer.step();losses.append(float(loss.detach()));assert np.isfinite(losses[-1])
     if (index+1)%32==0:
      ev=evaluate(actor);ev.update(block=index+1,elapsed_s=time.monotonic()-tick);evaluations.append(ev)
    wall=time.monotonic()-tick;assert torch.equal(core,torch.cat([p.detach().flatten() for n,p in actor.named_parameters() if pipeline.core_name(n)]));heads_nll=evaluations[-1]['dev_nll'];row=dict(variant=variant,width=width,depth=depth,head_only_final_dev_nll=heads_nll,head_only_prequential_nll=float(np.mean(losses)),acquisition_wall_s=acquired['wall_s'],reuse_wall_s=wall,final_per_sequence=evaluations[-1]['per_sequence'],evaluations=evaluations,core_unchanged=True,online_gain_vs_matched_heads=heads_nll-previous['arms']['online']['final_dev_nll'],online_final_dev_nll=previous['arms']['online']['final_dev_nll'],strong_adam_final_dev_nll=previous['arms']['bp_adam']['final_dev_nll']);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['evaluations','final_per_sequence']}),flush=True)
 sources=['experiments/credit/diagnose_event_head_only.py','experiments/credit/EVENT_HEAD_ONLY_DIAGNOSIS.md'];result=dict(status='completed',battle='B1',tag=a.tag,rows=rows,parent_sha256=parents,source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Exact new-variant acquisition replay/meanNLL+DEV parity; same acquired core/head optimizer, then512reuse heads only/corebitwisefrozen. Reused32DEV512targets, one initialization. Earlier structured acquired-core control is a cross-configuration comparison, not a matched deep-benefit claim for these variants; no startup-matched timing claim.');out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
