"""Freeze native checkpoints; isolate readout calibration and core/head learning."""
import argparse,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from sleeping_machines.native_memory_composition import NativeMemoryComposition
from sleeping_machines.prior_centered_readout import fit_prior_centered,linear_parameters
from experiments.hierarchy.native_query_attribution import ZeroQueryContent
from experiments.hierarchy.native_composition_data import execute,source_hashes,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
 fit_path=ROOT/'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_learn.json';fit=json.loads(fit_path.read_text())
 for n,digest in fit['source_sha256'].items():
  if sha(ROOT/n)!=digest:raise ValueError('changed fitted source: '+n)
 pre_path=ROOT/'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_data.json';pre=json.loads(pre_path.read_text());data=ROOT/pre['data']['path']
 if sha(data)!=pre['data']['sha256']:raise ValueError('changed prefix data')
 with np.load(data) as packed:events=packed['events'].copy();labels=torch.tensor(packed['labels'],dtype=torch.long)
 torch.manual_seed(101);initial_model=NativeMemoryComposition();initial={k:v.detach().clone() for k,v in initial_model.state_dict().items()}
 weights=ROOT/fit['final_weights'];final=torch.load(weights,map_location='cpu',weights_only=True)
 rows=[];arrays={'labels':labels.numpy(),'noise_seeds':np.array([113,127])};calibrated_weights=None
 for variant in ['initial','final','zero_query_final','initial_core_final_head','final_core_initial_head']:
  model=ZeroQueryContent() if variant=='zero_query_final' else NativeMemoryComposition()
  selected={k:(final[k] if variant in ['final','zero_query_final'] or (variant=='initial_core_final_head' and k in ['head.weight','head.bias']) or (variant=='final_core_initial_head' and k not in ['head.weight','head.bias']) else initial[k]) for k in initial}
  model.load_state_dict(selected);model.eval();predictions=[];work=[]
  with torch.no_grad():
   for seed in [113,127]:
    tape=[]
    for event in events:
     torch.manual_seed(seed);logits,_,counts=execute(model,event);tape.append(logits);work.append(counts)
    predictions.append(torch.stack(tape))
  z=torch.stack(predictions);flat=z.flatten(0,1);targets=labels.repeat(2)
  raw=float(F.cross_entropy(flat,targets));calibration=fit_prior_centered(flat,targets)
  if variant in ['initial','final']:
   if abs(raw-fit[variant]['nll'])>1e-5:raise ValueError('frozen endpoint mismatch: '+variant)
  if variant=='zero_query_final':
   attr_path=ROOT/'experiments/results/hierarchy/aws_g_hier_native_query_attribution_20261015_v1.json'
   attr=json.loads(attr_path.read_text())
   if abs(raw-attr['zero_query_content']['nll'])>1e-5:raise ValueError('query intervention mismatch')
  rows.append(dict(variant=variant,raw_nll=raw,**{k:v for k,v in calibration.items() if k not in ['center','log_prior']},work=work))
  arrays[variant+'_logits']=z.numpy();arrays[variant+'_center']=calibration['center'].numpy();arrays[variant+'_log_prior']=calibration['log_prior'].numpy()
  if variant=='final':
   calibrated_weights={k:v.clone() for k,v in final.items()}
   w,b=linear_parameters(final['head.weight'],final['head.bias'],calibration)
   calibrated_weights['head.weight']=w;calibrated_weights['head.bias']=b
   assert all(torch.equal(final[k],calibrated_weights[k]) for k in final if k not in ['head.weight','head.bias'])
  print(json.dumps({k:v for k,v in rows[-1].items() if k!='work'}),flush=True)
 outdir=ROOT/'experiments/results/hierarchy';packet=outdir/(a.tag+'_logits.npz')
 if packet.exists():raise ValueError('existing logits evidence')
 np.savez_compressed(packet,**arrays);calibrated=outdir/(a.tag+'.calibrated.pt');torch.save(calibrated_weights,calibrated)
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in ['experiments/hierarchy/native_frozen_readout.py','experiments/hierarchy/native_query_attribution.py']})
 result=dict(status='completed',battle='G-HIER',stage='native_frozen_readout',rows=rows,
  source_sha256=sources,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [fit_path,pre_path,data,weights]},
  logits_packet=dict(path=str(packet.relative_to(ROOT)),sha256=sha(packet)),final_weights=str(calibrated.relative_to(ROOT)),
  unchanged_native_parameters=True,process_wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  scope='Frozen native checkpoints,one FIT recording,two shared noise tapes. Prior/centres/one nonnegative scalar fitted on the same FIT logits. Optimistic learnability diagnostic, no held-out/semantic-generalization claim.',
  accounting='Backbone not refitted; all native rescoring work is recorded. One-dimensional FIT calibration is readout fitting, not free test-time postprocessing.',
  decision='Measure fixed-decoder changes and core/head coadaptation; swaps are not a rotation-invariant representation-quality comparison. Separate query-content association from prior/scale calibration. A FIT gain admits held-out composition design, not benchmark claims.')
 with (outdir/(a.tag+'.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
