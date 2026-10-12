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
 r=read('curie_shared_bank_20261012_v1_future_diagnosis')
 if r is not None:
  rows=[]
  for episode,x in enumerate(r['rows']):
   for v in x['variants']:rows.append([str(episode),v['update'],f"{v['future_nll']:.6f}",f"{v['gain_vs_momentum_coast']:+.6f}",f"{v['wall_s']:.3f}"])
  result.append([('h2','Native shared-memory critic: actual future-update diagnosis'),('table',(['TRAIN graph','Critic update','Next NLL','Gain vs coast','Branch seconds'],rows,[23,42,32,39,31])),('p','Component task/regression gradient cosines: '+', '.join(f"{x['gradient_cosine']:+.6f}" for x in r['rows'])+f". Source acquisition {r['source_acquisition_wall_s']:.3f}s; complete diagnosis {r['whole_pipeline_wall_s']:.3f}s including prefixes and every branch."),('small',r['scope'])])
 r=read('curie_task_taught_bank_20261012_v1_fit')
 if r is not None:
  rows=[]
  for x in r['rows']:
   targets=r['prior_acquisition_targets']+sum(t['targets'] for t in x['train'])
   dev_targets=sum(t['targets'] for t in x['dev'])
   rows.append([x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}",f"{1000*x['whole_fit_wall_s']/targets:.3f}",f"{1000*sum(t['wall_s'] for t in x['dev'])/dev_targets:.3f}"])
  v={x['arm']:x for x in r['rows']};gain=v['untaught']['mean_dev_nll']-v['reciprocal']['mean_dev_nll']
  result.append([('h2','Native shared-bank memory taught by forward task loss'),('table',(['Learner','DEV NLL','Full fit seconds','Fit ms/target','DEV ms/target'],rows,[30,32,35,35,35])),('p',f"Reciprocal versus matched untaught quality gain {gain:+.6f}. Auxiliary regression now trains only the identical linear readout; future task losses train native critic memory/access parameters, with separate actor/memory/readout clipping. Saved raw native credit0.613521/159.354s remains the practical quality/work reference. Fitting denominator includes512confidence plus1024neural presentations; DEV256targets, fixed weights and causal numerical state."),('small',r['scope'])])
 r=read('curie_addressed_bank_20261012_v1_fit')
 if r is not None:
  rows=[]
  for x in r['rows']:
   n=r['prior_acquisition_targets']+sum(t['targets'] for t in x['train'])
   rows.append([x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}",f"{1000*x['whole_fit_wall_s']/n:.3f}"])
  v={x['arm']:x for x in r['rows']};gain=v['untaught']['mean_dev_nll']-v['reciprocal']['mean_dev_nll']
  result.append([('h2','Public relative-address semantics in native shared memory'),('table',(['Learner','DEV NLL','Full fit seconds','Fit ms/target'],rows,[42,40,43,42])),('p',f"Reciprocal quality gain versus matched untaught {gain:+.6f}. Discovery uses public reader endpoints plus global recency fallback; independent learned relative-role/equality scores disambiguate identical-content queries. Active relabeling parity and causal completion visibility pass. Equality is prescribed public operation semantics; useful relationship weights are learned. No hidden parent/gate input or identity parameter table."),('small',r['scope'])])
 r=read('curie_outcome_bank_20261012_v1_fit')
 if r is not None:
  rows=[]
  for x in r['rows']:
   n=r['prior_acquisition_targets']+sum(t['targets'] for t in x['train'])
   rows.append([x['arm'],f"{x['mean_dev_nll']:.6f}",f"{x['whole_fit_wall_s']:.3f}",f"{1000*x['whole_fit_wall_s']/n:.3f}"])
  v={x['arm']:x for x in r['rows']};gain=v['untaught']['mean_dev_nll']-v['reciprocal']['mean_dev_nll']
  result.append([('h2','Outcome-preserving native credit feedback'),('table',(['Learner','DEV NLL','Full fit seconds','Fit ms/target'],rows,[42,40,43,42])),('p',f"Reciprocal quality gain versus matched untaught {gain:+.6f}. Delivered target replaces a redundant audit flag, with unchanged native payload dimension/parameter names. Scalar BCE at forecast0.5 erases target information; this token preserves it after feedback. Independent public-address races, native receiver writes and future task memory teaching remain. Raw-credit0.613521/159.354s is the retained practical reference until a completed comparable improvement."),('small',r['scope'])])
 acquired=read('curie_world_credit_20261012_v1_acquisition')
 if acquired is not None:
  checks=acquired['checks'];tail=checks['last_eight_world_nll']
  result.append([('h2','Native outcome-law credit: common nonlinear acquisition'),('table',(['Targets','Presentations','Source fit seconds','Later world NLL'],[[str(acquired['targets']),'4096',f"{acquired['whole_learning_wall_s']:.3f}",f"{tail:.6f}"]],[30,35,51,51])),('p','One target-addressed native outcome query and feedback update replaces unrelated route-wise loss predictions. Revealed outcome supervision trains the native body/read policy; forward losses also teach shared critic memory. Known proper-loss algebra projects one common outcome belief into coherent route contrasts. Source acquisition gate '+('passes' if checks['acquisition_gate'] else 'fails')+'. Acquisition uses16nonlinear TRAIN graphs with2048unique outcomes, one neural pass plus2048confidence presentations. The scalar output does not imply a tiny or disconnected world model.'),('small',acquired['scope'])])
  reuse=[]
  for arm in ['exact','learned','prior']:
   r=read(f'curie_world_credit_20261012_v2_{arm}_reuse')
   if r is not None:reuse.append(r)
  if reuse:
   rows=[];costs=[]
   for r in reuse:
    fullwall=acquired['whole_pipeline_wall_s']+r['whole_pipeline_wall_s']
    rows.append([r['arm'],f"{r['pooled_nll']:.6f}",f"{r['pooled_world_nll']:.6f}",f"{fullwall:.3f}",f"{1000*fullwall/5120:.3f}"])
    summed=lambda k:sum(x.get(k,0) for x in r['rows'])
    costs.append([r['arm'],str(summed('world_queries')),str(summed('bank_context_reads')),str(summed('sensor_decodes')),str(summed('credit_selected_updates')),f"{1000*r['fitting_wall_s']/1024:.3f}"])
   result.append([('h2','Native world credit reused on fresh nonlinear causal systems'),('table',(['Credit','Actor NLL','World NLL','Full seconds','Full ms/presentation'],rows,[25,31,31,37,43])),('table',(['Credit','World queries','Bank payload reads','Actor decodes','World writes','Reuse ms/target'],costs,[21,29,38,25,27,27])),('small','Completed arms only.16source systems/4096charged presentations plus8independent target deployments/1024prequential targets per arm: full denominator5120presentations. Full seconds include source/reuse driver preprocessing and model/optimizer copies; reuse ms/target uses actual training loops. Source acquisition charged once per deployment cohort. Fresh8/16factor4/8generating-depth systems, base/hidden-rewire. Actor/world learn causally in8feedback windows; common incoming actor/world/Adam fingerprints and identical shared-bank/outcome training. Learned and prior use deliberate half blends; exact uses observed-label contrast. All sampled alternative logits are decoded and charged. Generating depth is not neural-depth scaling; no publicTEST or efficiency claim.')])
  decision=read('curie_world_credit_20261012_v2_decision')
  if decision is not None:
   rows=[]
   for x in decision['cases']:rows.append([str(x['depth']),str(x['width']),x['variant'],f"{x['gains']['exact']:+.6f}",f"{x['gains']['prior']:+.6f}"])
   result.append([('h2','Native outcome-law credit: causal reuse decision'),('table',(['Generating depth','Factors','Variant','Gain vs exact','Gain vs prior'],rows,[31,22,32,41,41])),('p','Primary development gate '+('passes' if decision['development_gate'] else 'fails')+'; pooled quality gains '+', '.join(f"{k} {v:+.6f}" for k,v in decision['quality_gains'].items())+f". Learned world gain versus causal global frequency {decision['learned_world_gain_vs_frequency']:+.6f}; learned actor gain versus source-calibrated causal persistence {decision['learned_actor_gain_vs_persistence']:+.6f}. These diagnostics distinguish dependencies from marginal imbalance. Next: "+('unchanged fresh-actor/fresh-system confirmation before scaling.' if decision['development_gate'] else 'diagnose outcome sufficiency, credit sensitivity and causal information; no patch grid or scale-up.')),('small',decision['scope'])])
 strong=read('curie_persistence_credit_20261012_v1_decision')
 if strong is not None:
  rows=[]
  for arm in ['exact','learned','prior']:
   ref=read(f'curie_world_credit_20261012_v2_{arm}_reuse')
   full=acquired['whole_pipeline_wall_s']+ref['whole_pipeline_wall_s']
   rows.append([arm,f"{ref['pooled_nll']:.6f}",f"{full:.3f}",f"{1000*full/5120:.3f}"])
  full=strong['complete_acquisition_and_reuse_stage_wall_s']
  rows.append(['persistence credit',f"{strong['persistence_credit_nll']:.6f}",f"{full:.3f}",f"{1000*full/5120:.3f}"])
  result.append([('h2','Native credit against a strong causal persistence posterior'),('table',(['Credit','Actor NLL','Full seconds','Full ms/presentation'],rows,[52,35,40,40])),('p','Additional strong-control gate '+('passes' if strong['strong_control_gate'] else 'fails')+f"; learned gain versus persistence credit {strong['quality_gains_vs_persistence']['learned']:+.6f}. By generating depth: "+', '.join(f"{d}: {g:+.6f}" for d,g in strong['learned_depth_gains_vs_persistence'].items())+'. The original primary gate stays failed. Persistence beliefs match the independent causal public reference exactly; all native world learning, memory and alternative work are retained. Next: '+('unchanged fresh-actor/new-system confirmation against the strong control.' if strong['strong_control_gate'] else 'condition native outcome credit on issued forward/counterfactual responses and compare with forward-belief credit.')),('small',strong['scope']+' Complete denominator5120presentations:4096source plus1024target. This is an information-matched intervention, not the cheapest execution of persistence.')])
 for seed in [194,195,196]:
  confirmation=read(f'curie_world_credit_confirm_20261012_v1_s{seed}_decision')
  if confirmation is None:continue
  rows=[[r['arm'],f"{r['pooled_nll']:.6f}",f"{r['world_nll']:.6f}",f"{r['complete_acquisition_and_reuse_stage_wall_s']:.3f}",f"{1000*r['complete_acquisition_and_reuse_stage_wall_s']/5120:.3f}"] for r in confirmation['rows']]
  result.append([('h2',f'Native world-credit fresh-system confirmation: seed{seed}'),('table',(['Credit','Actor NLL','World NLL','Full seconds','Full ms/presentation'],rows,[25,31,31,37,43])),('p','Registered four-control confirmation gate '+('passes' if confirmation['confirmation_gate'] else 'fails')+'; learned gains '+', '.join(f"{k}: {g:+.6f}" for k,g in confirmation['quality_gains'].items())+'. Generating-depth gains '+', '.join(f"{d}/{k}: {g:+.6f}" for d,r in confirmation['depth_quality_gains'].items() for k,g in r.items())+'. Next: '+confirmation['next_decision']+'.'),('small',confirmation['scope'])])
 proof=read('curie_direct_route_depth_20261012_v1_contract')
 if proof is not None:
  c=proof['checks']
  metrics=[('Route-score identity',c['score_bias_identity_error']),('Producer covariance identity',c['producer_covariance_identity_error']),('Pair-conditioned mean identity',c['pair_conditioned_bias_identity_error']),('Actual clipped Adam update',c['actual_clipped_Adam_proposal_error']),('Delayed utility identity',c['delayed_utility_identity_error']),('Utility finite difference',c['utility_finite_difference_error'])]
  result.append([('h2','Direct route-score credit: depth bounds and delayed utility contracts'),('table',(['Numerical contract','Maximum absolute error'],[[name,f"{error:.3e}"] for name,error in metrics],[102,65])),('p','For a common issued binary-outcome belief independent of the route draw, score-gradient error has L1 bound |belief error| times logit range/2, with no candidate-count or neural-depth factor. Parameter-update error additionally depends on measured producer response geometry. Pair-conditioned response beliefs have a different bound and need their own consistency accounting. For a held finite optimizer transition, its future cotangent projects into one scalar teaching signal per issued credit control; both clipping and optimizer moments are retained. These conditional identities specify mechanisms to test, rather than a learned depth-efficiency result.'),('small',proof['scope']+f" Completed32score cases and both clipping regimes; {proof['whole_pipeline_wall_s']:.3f}s, peakRSS{proof['peak_rss_kb']}KB. Theory: experiments/theory/DIRECT_ROUTE_CREDIT_DEPTH_AND_DELAYED_UTILITY.md.")])
 return result
