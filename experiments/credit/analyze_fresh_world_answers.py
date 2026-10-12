"""B5/R1: causal answer-change audit of a completed fresh-system decision."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/causal_induction'))
from public_stream import records

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--decision',required=True);a=ap.parse_args()
 parent=ROOT/a.decision;d=json.loads(parent.read_text());assert d['status']=='completed'
 for group in ['source_sha256','parent_sha256']:
  for name,h in d[group].items():assert sha(ROOT/name)==h,name
 inputs={a.decision:sha(parent)};output=[]
 for arm in d['rows']:
  cases=[]
  for row in arm['rows']:
   path=ROOT/row['path'];inputs[row['path']]=sha(path)
   # Input bytes are already authenticated by each source-bound case result.
   case_inputs=[json.loads((ROOT/p).read_text()) for p in d['parent_sha256'] if p.endswith('.json') and '_case' in p]
   matching=[r for r in case_inputs if r.get('arm')==arm['arm'] and any(x['path']==row['path'] for x in r.get('data_records',[]))]
   assert len(matching)==1
   assert any(x['path']==row['path'] and x['sha256']==inputs[row['path']] for x in matching[0]['data_records'])
   bits={};issued={};feedback={}
   for event in records(path):
    if event['kind']=='event':bits[event['node']]=event['value']
    elif event['kind']=='query':issued[event['query_id']]=(bits.get(event['node'],0),event['time'])
    else:feedback[event['query_id']]=(event['target'],event['time'])
   assert len(row['issued_trace'])==len(issued)==row['targets']
   assert len({r['query_id'] for r in row['issued_trace']})==row['targets']
   changes=good=bad=baseline_correct=actor_correct=0
   for trace in row['issued_trace']:
    bit,t=issued[trace['query_id']];y,arrived=feedback[trace['query_id']]
    assert (t,arrived,y)==(trace['issued_time'],trace['feedback_time'],trace['target'])
    prediction=int(trace['probability']>=.5);changed=prediction!=bit
    changes+=int(changed);good+=int(changed and prediction==y);bad+=int(changed and prediction!=y)
    baseline_correct+=int(bit==y);actor_correct+=int(prediction==y)
   cases.append(dict(path=row['path'],depth=row['depth'],targets=row['targets'],answer_changes=changes,beneficial_changes=good,harmful_changes=bad,persistence_correct=baseline_correct,actor_correct=actor_correct))
  output.append(dict(arm=arm['arm'],cases=cases,**{key:sum(r[key] for r in cases) for key in ['targets','answer_changes','beneficial_changes','harmful_changes','persistence_correct','actor_correct']}))
 source=str(Path(__file__).relative_to(ROOT));reader='experiments/causal_induction/public_stream.py'
 result=dict(status='completed',battle='B5/R1',seed=d['seed'],rows=output,input_sha256=inputs,source_sha256={source:sha(Path(__file__)),reader:sha(ROOT/reader)},scope='Completed fresh-system forecasts replayed against issue-time public last-observed bits; exact source/data/trace identity required. Binary answer changes identify confidence-only outcomes but not unexecuted route effects. No fit, hidden graph or public test scoring.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps([{k:v for k,v in r.items() if k!='cases'} for r in output]),flush=True)
if __name__=='__main__':main()
