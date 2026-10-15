"""G-HIER: source-bound task audit and acquired-weight frontier resource gate.

No optimizer, label fitting or DEV/TEST model evaluation. Gradient objective is
output square, a numerical contract, not a task-quality measure.
"""
import argparse,gc,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.hierarchy.part_combination_task import corpus,audit
from experiments.hierarchy.native_composition_data import execute,source_hashes,sha
from sleeping_machines.frontier_native_composition import FrontierNativeComposition
from sleeping_machines.variance_reduced_native_composition import VarianceReducedNativeComposition


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.set_num_threads(1);started=time.monotonic();rows=corpus();summary=audit(rows)
    first=next(r for r in rows if r.split=='train')
    # All test metadata is used only for split checks; no model sees test data.
    manifest=[dict(split=r.split,left=r.left,right=r.right,root_swap=r.root_swap,
              nuisance=r.nuisance,label=r.label,sha256=r.digest) for r in rows]
    checkpoint=ROOT/'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_learn.final.pt'
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    measurements=[]
    for constructor in (VarianceReducedNativeComposition,FrontierNativeComposition):
        model=constructor();model.load_state_dict(weights);model.train();torch.manual_seed(113)
        before=time.monotonic();state=model.new_state()
        for micros,y,x,pol in first.events:
            model.observe(state,(int(y),int(x)),int(micros)/1e6,[float(pol),1.])
        logits,ready=model.compose(state);objective=logits.square().mean();objective.backward()
        norms=[float(obj.core.content.weight.grad.norm()) for obj in model.objects]
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(n>0 for n in norms) and torch.isfinite(ready) and not state.queue
        measurements.append(dict(model=constructor.__name__,seconds=time.monotonic()-before,
           work=model.work(state),content_gradient_norms=norms,
           max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        del model,state,logits,objective;gc.collect()
    sources=source_hashes()
    for name in ['experiments/hierarchy/frontier_preflight.py','experiments/hierarchy/part_combination_task.py',
                 'sleeping_machines/frontier_native_composition.py','sleeping_machines/variance_reduced_native_composition.py',
                 'sleeping_machines/pure_race_timer.py']:
        sources[name]=sha(ROOT/name)
    result=dict(status='completed',battle='G-HIER',stage='frontier_preflight',task_audit=summary,
        split_manifest=manifest,measurements=measurements,source_sha256=sources,
        input_sha256={str(checkpoint.relative_to(ROOT)):sha(checkpoint)},process_wall_s=time.monotonic()-started,
        scope='Synthetic split audit; one TRAIN scene, acquired 11-class gesture head retained only for numerical gradients. No optimizer, adaptation, model selection or semantic quality result.',
        decision='Admit task-specific learning only after measured frontier cost and gradient support; choose publication/coalescing repair if the stronger composition path is too costly.')
    out=ROOT/f'experiments/results/hierarchy/{args.tag}.json'
    with out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps({k:result[k] for k in ['status','task_audit','measurements','process_wall_s']}))
if __name__=='__main__':main()
