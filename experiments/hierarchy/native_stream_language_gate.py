"""R1 real-token gate: native8depth,indexed history,cause messages,local credit."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM,word_classes
from experiments.hierarchy.native_composition_data import source_hashes


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(161016);began=time.monotonic()
 data=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')
 counts=np.bincount(np.asarray(data[:4194304]).astype(np.int64),minlength=50257);classes,words=word_classes(counts,256)
 model=LocalNativeDeepLM(50257,classes,words,width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);state=model.new_state()
 g=torch.Generator().manual_seed(13);alt=torch.Generator().manual_seed(14)
 tokens=torch.from_numpy(np.asarray(data[:33]).astype(np.int64));reports=[];start=time.monotonic()
 for b,e in [(0,16),(16,32)]:
  opt.zero_grad(set_to_none=True)
  nll,obj,state,packets,report=model.stream_nll(tokens[b:e],tokens[b+1:e+1],state,g,alt,stats=True)
  obj.backward();model.deliver_credit(packets,state)
  assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
  norms=[sum(float(v.grad[d].square().sum()) for v in model.core.banks.values() if v.grad is not None)**.5 for d in range(8)]
  assert min(norms)>0;torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();report['layer_gradient_norms']=norms;reports.append(report)
 assert state['events']==state['cause_bank'].count==32 and state['cause_inbox'].grad_fn is None
 assert all(v.grad_fn is None for layer in state['layers'] for v in layer.values())
 sources=source_hashes()
 for name in ['experiments/r1_local_native_deep.py','experiments/r1_cerl.py','experiments/hierarchy/native_stream_language_gate.py']:sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
 elapsed=time.monotonic()-start
 result=dict(status='completed',battle='R1',stage='native_stream_language_gate',gate='PASS',targets=32,updates=2,depth=8,parameters=sum(p.numel() for p in model.parameters()),reports=reports,
  learning_counts={k:state[k] for k in ['events','candidate_scores','selected_updates','alternative_values','credit_projections']},
  bank=state['cause_bank'].storage(),numeric_state=model.core.storage(state),whole_optimizer_s=elapsed,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_wall_s=time.monotonic()-began,source_sha256=sources,
  training_prefix_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),scope='Two16targetcausalTRAINoptimizerupdates,persistentnumericalreceivers/indexedepisodichistory/causeinboxesacrosscreditboundary.No recurrentBPTT;receiverown-writeeligibilityconditional/fixed-parameterexactness,changing-parameterhistoryapproximate.ZeroSSM/attention/MLPcarrier.Two-update numerical gate,not a quality or complete asynchronous-optimizer result.No DEV/TEST or ablation.')
 target=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['gate','whole_optimizer_s','max_rss_kb','parameters']}),flush=True)
if __name__=='__main__':main()
