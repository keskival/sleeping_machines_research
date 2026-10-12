"""B1 essential frozen-core control for future-taught native learning."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
import run_native_structured_credit as pipeline
ROOT=pipeline.ROOT

def evaluate(actor,sequences):
    tick=time.monotonic();actor.cut=False;actor.collect_bases=False
    with torch.no_grad():losses=[float(actor(*pipeline.one_block(sequence))) for sequence in sequences]
    return dict(dev_nll=float(np.mean(losses)),per_sequence=losses,wall_s=time.monotonic()-tick)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();train=[s for s in pipeline.native.load_split('taxi','train') if len(s[1])>=32];dev=[s for s in pipeline.native.load_split('taxi','dev') if len(s[1])>=32][:32]
    gaps=np.concatenate([np.diff(t[:32]) for t,_ in train[:32]]);gaps=gaps[gaps>0];scale=float(np.median(gaps));qs=np.log(np.quantile(gaps,[.1,.5,.9])).tolist();K=1+max(int(m.max()) for _,m in train);rows=[];parents={}
    for width in [16,32]:
        path=ROOT/f'experiments/results/credit/curie_native_structured_credit_20261011_v1_d{width}_structured_fit.json';parent=json.loads(path.read_text());assert parent['status']=='completed';parents[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for name,digest in parent['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        pipeline.RUN_WIDTH=width;pipeline.RUN_FIELD='structured'
        for depth in [4,8]:
            previous=next(r for r in parent['rows'] if r['depth']==depth);actor=pipeline.initial(99910,depth,K,scale,qs);field=pipeline.new_credit(actor,depth)
            acquired,_=pipeline.learn(actor,field,train[256:384],'online',99911)
            assert abs(acquired['mean_nll']-previous['arms']['online']['acquisition']['mean_nll'])<1e-10
            beginning=evaluate(actor,dev);assert abs(beginning['dev_nll']-previous['arms']['online']['acquisition_dev']['dev_nll'])<1e-10
            core=torch.cat([p.detach().flatten().clone() for n,p in actor.named_parameters() if pipeline.core_name(n)]);heads=[p for n,p in actor.named_parameters() if not pipeline.core_name(n)];losses=[];evaluations=[];tick=time.monotonic()
            for index,sequence in enumerate(train[384:896]):
                actor.cut=True;actor.collect_bases=False;loss=actor(*pipeline.one_block(sequence));gradients=torch.autograd.grad(loss,heads);norm=torch.stack([g.square().sum() for g in gradients]).sum().clamp_min(1e-24).sqrt();factor=(5/norm).clamp_max(1);actor.head_optimizer.zero_grad()
                for parameter,gradient in zip(heads,gradients):parameter.grad=(factor*gradient).detach()
                actor.head_optimizer.step();losses.append(float(loss.detach()));assert np.isfinite(losses[-1])
                if (index+1)%32==0:
                    e=evaluate(actor,dev);e.update(block=index+1,elapsed_s=time.monotonic()-tick);evaluations.append(e)
            wall=time.monotonic()-tick;final=evaluations[-1];assert torch.equal(core,torch.cat([p.detach().flatten() for n,p in actor.named_parameters() if pipeline.core_name(n)]))
            row=dict(width=width,depth=depth,head_only_final_dev_nll=final['dev_nll'],head_only_prequential_nll=float(np.mean(losses)),head_only_reuse_wall_s=wall,acquisition_wall_s=acquired['wall_s'],whole_head_only_wall_s=acquired['wall_s']+wall,final_per_sequence=final['per_sequence'],evaluations=evaluations,comparison={arm:dict(final_dev_nll=v['final_dev_nll'],whole_learning_wall_s=v['whole_learning_wall_s'],head_only_quality_gain=v['final_dev_nll']-final['dev_nll']) for arm,v in previous['arms'].items()},core_unchanged=True);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['final_per_sequence','evaluations']}),flush=True)
    paths=['experiments/credit/diagnose_native_head_only.py','experiments/credit/run_native_structured_credit.py','experiments/credit/structured_parameter_credit.py','experiments/credit/NATIVE_HEAD_ONLY_DIAGNOSIS.md']
    result=dict(status='completed',battle='B1',tag=a.tag,rows=rows,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=parents,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Replay exact common128-block acquired actor/head optimizer for each completed width/depth, verified acquisition likelihood/DEV parity. Freeze every core parameter; train heads only on same512reuse blocks, same32DEV sequences. All acquisition/reuse/evaluation work measured; parent allocated compiled startup not included here, so no timing-supremacy claim. Essential capability control, reused development data, no new native/public claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
