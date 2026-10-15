"""Measure one true paired native update before the conditional comparison."""
import argparse,gc,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.paired_composition_learning import make_model,paired_backward,PARENT
from experiments.hierarchy.native_composition_data import sha,source_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);started=time.monotonic();train=[r for r in corpus() if r.split=='train']
    pairs=training_pairs([r.model_input() for r in train]);i,j=pairs[0];records=[]
    for arm in ['frame','relational']:
        model=make_model(arm);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        optimizer.zero_grad(set_to_none=True);before=time.monotonic()
        loss,work=paired_backward(model,[(train[i],train[i].label),(train[j],train[j].label)])
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        norms=[float(obj.core.content.weight.grad.norm()) for obj in model.objects];assert all(n>0 for n in norms)
        interaction_gradient=None
        if arm=='relational':
            interaction_gradient=float(model.objects[-1].core.composer.pair_weight.grad.norm())
            assert interaction_gradient>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        records.append(dict(arm=arm,paired_update_s=time.monotonic()-before,work=work,loss=loss,
          content_gradient_norms=norms,top_pair_gradient_norm=interaction_gradient,parameters=sum(p.numel() for p in model.parameters()),
          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        del model,optimizer;gc.collect()
    sources=source_hashes()
    for f in ['experiments/hierarchy/relational_preflight.py','experiments/hierarchy/paired_composition_learning.py','experiments/hierarchy/observed_training_pairs.py','experiments/hierarchy/part_combination_task.py','experiments/hierarchy/frame_learning.py']:sources[f]=sha(ROOT/f)
    result=dict(status='completed',battle='G-HIER',stage='relational_paired_preflight',measurements=records,
      source_sha256=sources,input_sha256={str(PARENT.relative_to(ROOT)):sha(PARENT)},process_wall_s=time.monotonic()-started,
      scope='One paired TRAIN update per native arm,real32-event scenes,original acquired core/fresh classifier. Numerical/resource gate only. Paired loss is not a quality result; cumulative process peak is not isolated per-arm RSS.',
      decision='Admit one64paired-update comparison only with finite deep/interaction gradients and host RSS margin. Compare native role-memory composition to the same paired native frame learner; no external baseline fit.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(measurements=records)),flush=True)
if __name__=='__main__':main()
