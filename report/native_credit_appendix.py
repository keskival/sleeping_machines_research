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
  rows=[[x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}",str(sum(t['targets'] for t in x['train'])+r.get('prior_acquisition_targets',r.get('prior_acquisition',{}).get('targets',0)))] for x in r['rows']]
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
 r=read('curie_causal_race_credit_20261012_v1_controls')
 if r is not None:
  reference=read('curie_causal_race_credit_20261012_v1_fit');learned=next(x for x in reference['rows'] if x['arm']=='credit');rows=[[x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}"] for x in r['rows']];rows.append(['Native learned credit',f"{learned['mean_dev_nll']:.6f}",f"{learned['whole_fit_wall_s']:.3f}"])
  result.append([('h2','Matched audit rate and native credit-memory attribution'),('table',(['Learner','DEV NLL','Full fitting seconds'],rows,[77,45,45])),('small',r['scope']),('p','; '.join(f"Native learned credit versus {name}: quality gain {x['learned_quality_gain']:+.6f}, time ratio {x['learned_time_ratio']:.3f}x" for name,x in r['comparisons'].items())+'. These controls determine the interpretation of the preceding first quality gain.')])
 confirmed=[]
 for seed in [194,195,196]:
  r=read(f'curie_causal_race_credit_confirm_20261012_v1_s{seed}')
  if r is not None:confirmed.append(r)
 if confirmed:
  rows=[]
  for r in confirmed:
   for x in r['rows']:rows.append([str(r['seed']),x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}"])
  verdict='Fixed cohort still running; completed cells only.'
  decision=read('curie_causal_race_credit_confirm_20261012_v1_decision')
  if decision is not None:verdict='Confirmation gate '+('passes' if decision['confirmation_gate'] else 'fails')+'; mean quality gains '+', '.join(f"{k} {v:+.6f}" for k,v in decision['mean_quality_gains'].items())+'. Exact comparison wins two of three; learning/memory controls improve in all three.'
  result.append([('h2','Fixed native-credit confirmation on new causal graphs'),('table',(['Actor seed','Learner','Fresh DEV NLL','Full fitting seconds'],rows,[23,54,42,48])),('p',verdict),('small','Same unchanged architecture/data/learning rule; new graphs2004–2007, fixedactors194/195/196. Seed194 reuses saved training;195/196 fit unchanged source recipe. Four private graphs256targets per actor/learner; no target-domain weight adaptation. Full fitting includes512-target confidence acquisition and1024neural presentations; no public benchmark or efficiency claim.')])
 r=read('curie_causal_centered_credit_20261012_v3_pilot')
 if r is not None:
  rows=[[x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}"] for x in r['rows']]
  original=read('curie_causal_race_credit_20261012_v1_fit');control=read('curie_causal_race_credit_20261012_v1_controls')
  for label,parent in [('exact',original),('credit',original),('zero25',control)]:
   x=next(x for x in parent['rows'] if x['arm']==label);rows.append([label+' (saved)',f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}"])
  result.append([('h2','Native credit teacher: common-outcome control variate'),('table',(['Learner','Selection DEV NLL','Full fitting seconds'],rows,[77,45,45])),('small',r['scope']),('p','Both route targets subtract the same issued baseline BCE, preserving their actual contrast. For binary outcome probabilityp, conditional target variance becomes p(1-p)*(z-z0)^2. The predictor remains native temporal/race memory with a linear readout. This is a diagnosed teacher repair, not a replacement dense controller; fresh-graph confirmation scores are separate.')])
 r=read('curie_shared_bank_20261012_v2_fit')
 if r is not None:
  rows=[[x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}"] for x in r['rows']]
  v={x['arm']:x for x in r['rows']};reciprocal=v['reciprocal']
  verdict='; '.join(f"Reciprocal quality gain versus {name}: {v[name]['mean_dev_nll']-reciprocal['mean_dev_nll']:+.6f}" for name in ['separate','read_only','untaught'])
  costs=[]
  for x in r['rows']:
   total=lambda key:sum(t.get(key,0) for t in x['train'])
   costs.append([x['arm'],str(total('bank_discovery_records')),str(total('bank_context_reads')),str(total('bank_writes')),str(total('forward_credit_reads')),str(total('critic_forward_reads'))])
  result.append([('h2','Independent native shared-memory reciprocal teaching'),('table',(['Learner','Selection DEV NLL','Full fitting seconds'],rows,[77,45,45])),('p',verdict+'. Single-seed development; compare the saved unshared native credit0.613521/159.354s for practical quality/work. Suppressed sharing controls execute diagnostic bank overhead; their timing is not the economical separate-bank implementation.'),('table',(['Learner','Discovery scans','Payload reads','Bank writes','F reads C','C reads F'],costs,[28,32,31,26,25,25])),('small',r['scope']),('p','Independent native read races and selected receiver writes share versioned records. Later forward task loss trains critic memory content and its read policy within retained gradient windows; the untaught control stops only critic record gradients and keeps the actor reader trainable. Physical completion gates visibility. Bank discovery uses bounded recency, and global write destinations are not learned.')])
 return result
