"""Conditional frozen attribution of learned query content; no new optimizer."""
import argparse,json,sys,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sleeping_machines.native_memory_composition import NativeMemoryComposition
from experiments.hierarchy.native_composition_data import source_hashes,sha
from experiments.hierarchy.native_composition_learning import score


class ZeroQueryContent(NativeMemoryComposition):
    def _send(self,state,source_level,source_address,value,version,time,kind,target=None):
        if kind=='query':value=torch.zeros_like(value)
        return super()._send(state,source_level,source_address,value,version,time,kind,target)


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
 fit_path=ROOT/'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_learn.json'
 fit=json.loads(fit_path.read_text())
 for name,digest in fit['source_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('changed fitted source: '+name)
 result=dict(status='completed',battle='G-HIER',stage='native_query_attribution',
    scope='Frozen FIT-only query-content intervention. Same response count/metadata/order; input processing intact. Not generalization or exact whole-family attribution.',
    input_sha256={str(fit_path.relative_to(ROOT)):sha(fit_path)})
 promising=fit['final']['nll']<fit['initial']['nll'] and fit['final']['label_association_gain']>0
 if not promising:
  result.update(skipped_model_evaluation=True,decision='No useful combined loss/association signal; repair native learning before source attribution or scaling.')
 else:
  pre_path=ROOT/'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_data.json';pre=json.loads(pre_path.read_text());data=ROOT/pre['data']['path']
  if sha(data)!=pre['data']['sha256']:raise ValueError('changed prefixes')
  with np.load(data) as packed:events=packed['events'].copy();labels=torch.tensor(packed['labels'],dtype=torch.long)
  model=ZeroQueryContent();weights=ROOT/fit['final_weights'];model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
  altered,work=score(model,events,labels)
  result.update(skipped_model_evaluation=False,factual=fit['final'],zero_query_content=altered,
      zero_query_nll_delta=altered['nll']-fit['final']['nll'],work=work,
      decision='Positive delta supports useful queried content beyond positions/prior messages; otherwise diagnose query budget/content integration. Then design held-out composition, not a depth sweep.')
  result['input_sha256'].update({str(weights.relative_to(ROOT)):sha(weights),str(data.relative_to(ROOT)):sha(data)})
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in ['experiments/hierarchy/native_query_attribution.py','experiments/hierarchy/native_composition_learning.py']})
 result.update(source_sha256=sources,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 out=ROOT/f'experiments/results/hierarchy/{a.tag}.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','input_sha256','work']}))
if __name__=='__main__':main()
