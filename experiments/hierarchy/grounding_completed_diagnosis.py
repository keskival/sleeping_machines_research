"""Compare completed native representations with calibrated equal-capacity heads."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.hierarchy.grounding_decoder_calibration import fit_affine_geometry,predict
from experiments.hierarchy.part_combination_task import corpus
from sleeping_machines.grounded_native_composition import ObservedGeometrySketch
from sleeping_machines.sparse_geometry_teacher import sparse_geometry_targets
from experiments.hierarchy.native_composition_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
    variance_path='experiments/results/hierarchy/aws_g_hier_grounding_variance_20261016_v1.json';variance=json.loads((ROOT/variance_path).read_text())
    teacher=ObservedGeometrySketch().double();tr=[];dv=[]
    for row in corpus():
        if row.split=='test' or (row.split=='train' and row.nuisance!=0):continue
        events,_=row.model_input();targets,_=sparse_geometry_targets(events,8,tuple(events[:,1:3].min(0)),teacher.directions)
        (tr if row.split=='train' else dv).append(targets[(7,(0,0))][0])
    y=torch.stack(tr);yd=torch.stack(dv);rows={};inputs={variance_path:sha(ROOT/variance_path)}
    for arm in ['joint','continued','decoder_only']:
        name=f'experiments/results/hierarchy/aws_g_hier_grounded_{arm}_20261016_v1.json';fit=json.loads((ROOT/name).read_text());feature=ROOT/fit['feature_packet']['path'];assert sha(feature)==fit['feature_packet']['sha256']
        with np.load(feature) as packet:x=torch.from_numpy(packet['final_train']);xd=torch.from_numpy(packet['final_dev'])
        decoder=fit_affine_geometry(x,y);mse=float((predict(xd,decoder)-yd).square().mean());constant=variance['constant_by_level'][7]['dev_loss']
        rows[arm]=dict(nll=fit['final']['materialized_dev_nll'],calibrated_root_geometry_mse=mse,calibrated_root_geometry_skill=1-mse/constant,
            adam_geometry_skill_by_level=[1-loss/ref['dev_loss'] for loss,ref in zip(fit['final']['grounding_dev'],variance['constant_by_level'])],decoder_max_gradient=decoder['max_gradient'])
        inputs[name]=sha(ROOT/name);inputs[str(feature.relative_to(ROOT))]=sha(feature)
    gain=rows['decoder_only']['calibrated_root_geometry_mse']-rows['joint']['calibrated_root_geometry_mse']
    if rows['joint']['nll']>=rows['decoder_only']['nll']:
        decision=('Joint learning improves calibrated root geometry but not class transfer;repair relation-specific learning before scaling.' if gain>0 else 'Joint learning fails both acquired-actor quality and calibrated root geometry;repair local target conditioning/credit before another acquisition.')
    else:decision='Joint learning improves acquired-actor point quality;use frozen decision uncertainty/channel tests before confirmation.'
    result=dict(status='completed',battle='G-HIER',stage='grounding_completed_diagnosis',rows=rows,joint_calibrated_geometry_gain_over_frozen=gain,decision=decision,input_sha256=inputs,source_sha256={f:sha(ROOT/f) for f in ['experiments/hierarchy/grounding_completed_diagnosis.py','experiments/hierarchy/grounding_decoder_calibration.py']},process_wall_s=time.monotonic()-start,scope='Fixed ridge0.01 existing linear geometry heads on saved TRAIN features;DEV scoring,one native seed,no new actor acquisition or TEST access. Geometry skill is not semantic transfer.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
