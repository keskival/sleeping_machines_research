"""TRAIN-only geometry normalization and closed-form saved-root decoder audit."""
import argparse,json,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sleeping_machines.grounded_native_composition import ObservedGeometrySketch
from sleeping_machines.sparse_geometry_teacher import sparse_geometry_targets
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.native_composition_data import sha


def fit_affine_geometry(x,targets,ridge=.01):
    x=x.double();targets=targets.double();center=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-6)
    design=torch.cat(((x-center)/scale,torch.ones_like(x[:,:1])),1)
    penalty=x.new_full((design.shape[1],),ridge);penalty[-1]=0
    beta=torch.linalg.solve(design.T@design/len(x)+torch.diag(penalty),design.T@targets/len(x))
    residual=design.T@(design@beta-targets)/len(x)+penalty[:,None]*beta
    return dict(beta=beta,center=center,scale=scale,max_gradient=float(residual.abs().max()))


def predict(x,fit):
    x=x.double();design=torch.cat(((x-fit['center'])/fit['scale'],torch.ones_like(x[:,:1])),1)
    return design@fit['beta']


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);started=time.monotonic()
    teacher=ObservedGeometrySketch().double();training=[[] for _ in range(8)];cal=[];dev=[]
    for scene in corpus():
        if scene.split not in ['train','dev']:continue
        events,_=scene.model_input();targets,_=sparse_geometry_targets(events,8,tuple(events[:,1:3].min(0)),teacher.directions)
        if scene.split=='train':
            for l in range(8):training[l].append(torch.stack([v for (level,address),(v,n) in sorted(targets.items()) if level==l]))
            if scene.nuisance==0:cal.append(targets[(7,(0,0))][0])
        else:dev.append(targets[(7,(0,0))][0])
    center=torch.stack([torch.stack([x.mean(0) for x in batch]).mean(0) for batch in training])
    variance=torch.stack([torch.stack([(x-mu).square().mean() for x in batch]).mean() for batch,mu in zip(training,center)])
    frozen_path='experiments/results/hierarchy/aws_g_hier_grounded_decoder_only_20261016_v1.json';old=json.loads((ROOT/frozen_path).read_text());feature=ROOT/old['feature_packet']['path'];assert sha(feature)==old['feature_packet']['sha256']
    with np.load(feature) as packet:x=torch.from_numpy(packet['final_train']);xd=torch.from_numpy(packet['final_dev'])
    y=torch.stack(cal);yd=torch.stack(dev);fit=fit_affine_geometry(x,y)
    loss=lambda z,t:float((z-t).square().mean())
    mse=loss(predict(xd,fit),yd);constant=loss(center[7].expand_as(yd),yd)
    artifact=ROOT/f'experiments/results/hierarchy/{a.tag}.npz'
    np.savez_compressed(artifact,center=center.numpy(),variance=variance.numpy(),directions=teacher.directions.numpy())
    result=dict(status='completed',battle='G-HIER',stage='grounding_decoder_calibration',normalizer=dict(path=str(artifact.relative_to(ROOT)),sha256=sha(artifact),training_scenes=128,center=center.tolist(),variance=variance.tolist()),root_decoder=dict(ridge=.01,train_scenes=len(x),train_mse=loss(predict(x,fit),y),dev_mse=mse,constant_dev_mse=constant,dev_skill=1-mse/constant,adam_decoder_dev_mse=old['final']['grounding_dev'][7],max_gradient=fit['max_gradient']),input_sha256={frozen_path:sha(ROOT/frozen_path),str(feature.relative_to(ROOT)):sha(feature)},source_sha256={f:sha(ROOT/f) for f in ['experiments/hierarchy/grounding_decoder_calibration.py','sleeping_machines/centered_geometry_grounding.py','sleeping_machines/sparse_geometry_teacher.py']},process_wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='TRAIN-only per-level means/variances,fixed ridge0.01 closed-form fit of the existing32-output linear geometry decoder on saved TRAIN root features. DEV evaluates only;TEST excluded. Frozen actor not changed. No hyperparameter tuning,semantic or native representation-learning claim.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['root_decoder']),flush=True)
if __name__=='__main__':main()
