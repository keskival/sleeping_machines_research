"""Completed source-bound native credit-memory development; no projected cells."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(tag):
 path=ROOT/f'experiments/results/causal_induction/{tag}.json'
 if not path.exists():return None
 r=json.loads(path.read_text());assert r['status']=='completed'
 for name,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 return r

def pages():
 result=[]
 for tag,title in [('curie_causal_native_20261012_v1_fit','Native future-sensor interface'),('curie_causal_structural_20261012_v2_fit','Observation-acquired relationships and sensor repair'),('curie_causal_race_credit_20261012_v1_fit','Native temporal race memory for credit itself')]:
  r=read(tag)
  if r is None:continue
  rows=[[x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}",str(sum(t['targets'] for t in x['train'])+r.get('prior_acquisition_targets',r.get('prior_acquisition',{}).get('targets',0))))] for x in r['rows']]
  if 'mean_persistence_dev_nll' in r:rows.append(['Acquired persistence',f"{r['mean_persistence_dev_nll']:.6f}",f"{r['prior_acquisition']['wall_s']:.3f}",str(r['prior_acquisition']['targets'])])
  block=[('h2',title),('table',(['Learner','DEV NLL','Full fitting seconds','Target presentations'],rows,[61,34,40,32])),('small',r['scope'])]
  if tag=='curie_causal_native_20261012_v1_fit':block.append(('p','First-pass routed actor loses to self-only0.666639 and acquired persistence0.618503. Its selected value-read counter excludes discovery traffic: every request also reads/transports candidate_keys+1 contexts. This result diagnoses the sensor interface, not the full model family.'))
  elif tag=='curie_causal_structural_20261012_v2_fit':block.append(('p','Sparse observed relationship evidence improves routed NLL0.620672→0.617000. Self-only still leads0.615994; acquired persistence is0.618997 on the same four development graphs. No native routing/credit advantage declared.'))
  else:
   exact=next(x for x in r['rows'] if x['arm']=='exact');comparisons=[]
   for x in r['rows']:
    if x['arm']=='exact':continue
    comparisons.append(f"{x['arm']}: quality gain versus exact {exact['mean_dev_nll']-x['mean_dev_nll']:+.6f}, fitting-time ratio {x['whole_fit_wall_s']/exact['whole_fit_wall_s']:.3f}x")
   block.append(('p','; '.join(comparisons)+'. Single-seed private diagnostic; strongest saved self-only and persistence results remain the practical quality references.'))
   costs=[]
   for x in r['rows']:
    summed=lambda key:sum(t.get(key,0) for t in x['train'])
    costs.append([x['arm'],str(summed('discovery_context_reads')),str(summed('relationship_candidate_updates')),str(summed('credit_selected_updates')),str(summed('credit_counterfactual_values')),str(summed('exact_alternative_audits'))])
   block.append(('table',(['Learner','Discovery reads','Pair writes','Credit writes','Credit alternatives','Exact audits'],costs,[27,32,27,27,30,24])))
  result.append(block)
 return result
