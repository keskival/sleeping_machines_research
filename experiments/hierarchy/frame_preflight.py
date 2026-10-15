"""G-HIER: acquired-weight frame repair, exact translation and resource checks."""
import argparse,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.hierarchy.coalesced_nuisance_factorial import factorial
from experiments.hierarchy.coalesced_frozen_readout import features
from experiments.hierarchy.coalesced_learning import forward
from experiments.hierarchy.native_composition_data import source_hashes,sha
from sleeping_machines.frame_native_composition import FrameNativeComposition


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    started=time.monotonic();torch.set_num_threads(1)
    rows=[]
    for left,difference in enumerate((0,1,3,7)):
        for root in (0,1):rows+=factorial(left,left^difference,root)
    weights=ROOT/'experiments/results/hierarchy/aws_g_hier_coalesced_learning_20261015_v1.final.pt'
    model=FrameNativeComposition(classes=2);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    x,y,work=features(model,rows);array=x.numpy().reshape(4,2,4,-1)
    delta={label:float(np.linalg.norm(array[:,:,i]-array[:,:,0],axis=-1).mean())
           for i,label in [(1,'translation_only'),(2,'timing_only'),(3,'both')]}
    assert np.array_equal(array[:,:,0],array[:,:,1])
    model.train();torch.manual_seed(113);before=time.monotonic()
    logits,train_work=forward(model,rows[0]);logits.square().mean().backward()
    norms=[float(obj.core.content.weight.grad.norm()) for obj in model.objects]
    assert all(n>0 for n in norms)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    reference=ROOT/'experiments/results/hierarchy/aws_g_hier_nuisance_factorial_20261015_v1.json'
    ref=json.loads(reference.read_text());old=next(r for r in ref['rows'] if r['variant']=='final')
    sources=source_hashes()
    for f in ['experiments/hierarchy/frame_preflight.py','experiments/hierarchy/coalesced_nuisance_factorial.py','experiments/hierarchy/coalesced_frozen_readout.py','experiments/hierarchy/coalesced_learning.py','experiments/hierarchy/part_combination_task.py','sleeping_machines/frame_native_composition.py']:sources[f]=sha(ROOT/f)
    result=dict(status='completed',battle='G-HIER',stage='frame_preflight',feature_displacement=delta,
        root_relation_displacement=float(np.linalg.norm(array[:,0,0]-array[:,1,0],axis=-1).mean()),
        reference_final=old,exact_translation_feature_equality=True,work_evaluation=work,work_backward=train_work,
        content_gradient_norms=norms,backward_wall_s=time.monotonic()-before,
        parameters=sum(p.numel() for p in model.parameters()),source_sha256=sources,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [weights,reference]},
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_wall_s=time.monotonic()-started,
        scope='Same32 factorial TRAIN scenes and acquired native weights; one32-event numerical backward, no optimizer or DEV/TEST scoring. Exact translation is a mechanism contract, not a learned semantic quality result.',
        decision='Does the observed-frame repair remove the measured coordinate nuisance without merely moving it into timing sensitivity? Use this and charged work before another task fit.')
    out=ROOT/f'experiments/results/hierarchy/{a.tag}.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['feature_displacement','root_relation_displacement','work_backward','parameters','process_wall_s']}),flush=True)
if __name__=='__main__':main()
