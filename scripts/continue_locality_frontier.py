"""R1 one useful delayed-credit acquisition, gated by completed locality evidence."""
import json,math,time
from pathlib import Path
import continue_window_replay as queue
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+30*86400
PREFIX='aws_locality_frontier_20261014_s6_v1'

def complete(relative):
 r=queue.wait_result(ROOT/relative)
 for name,digest in {**r.get('source_sha256',{}),**r.get('input_sha256',{})}.items():
  assert queue.sha(ROOT/name)==digest,name
 return r

def admit(tag,script,args,result,sources,timeout,requires=()):
 p=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';text='# R1: useful depth through local delayed credit; decision quality/fullwork versus strongest completed native references\n'+tag+' '+script+' '+args+'\n'
 if p.exists():
  assert p.read_text()==text
  return complete(result)
 p.parent.mkdir(exist_ok=False);p.write_text(text)
 job=dict(name=tag,queue=str(p.relative_to(ROOT)),queue_sha256=queue.sha(p),kind='r1_job',source_sha256=sources,result=result,requires=list(requires),timeout_s=timeout,rss_kb=3000000,vms_kb=12000000)
 packet=dict(host='ip-172-31-47-132',battle='R1',purpose='One conditional full delayed-credit locality repair after completed comparisons; not a component grid',parent_manifest_sha256=queue.sha(queue.PLAN/'manifest.json'),slots={'1':[job]})
 path=queue.PLAN/'addenda'/('zzzzzzzzzzzzzzzzzzzzzzzzzzzr1_'+str(time.time_ns())+'_s1.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
 print('admitted '+tag,flush=True);return complete(result)

def main():
 local=complete('experiments/results/diagnostics/aws_local_clip_receiver_d8_8m_quality_20261014_s6_v1.json')
 global_=complete('experiments/results/diagnostics/aws_integrated_receiver_d8_8m_quality_20261014_s6_v1.json')
 l=local['rows'][-1];g=global_['rows'][-1]
 for k in ('targets','target_start','target_end_inclusive','window','lanes','eos_state_reset'):assert l['scored'][k]==g['scored'][k]
 assert l['acquisition']==g['acquisition'] and l['depth']==g['depth']==8
 delta=l['scored']['nll']-g['scored']['nll']
 eligible=delta<=.01
 decision=ROOT/f'experiments/results/diagnostics/{PREFIX}_admission.json'
 if not decision.exists():
  decision.write_text(json.dumps(dict(status='completed',battle='R1',local_minus_global_nll=delta,eligible=eligible,decision='one combined delayedcredit acquisition' if eligible else 'stop; repair local update quality before new acquisition',input_sha256={p:queue.sha(ROOT/p) for p in ['experiments/results/diagnostics/aws_local_clip_receiver_d8_8m_quality_20261014_s6_v1.json','experiments/results/diagnostics/aws_integrated_receiver_d8_8m_quality_20261014_s6_v1.json']},source_sha256={'scripts/continue_locality_frontier.py':queue.sha(Path(__file__))},scope='Completed8M comparison, <=0.01nats localquality degradation eligibility only, not a win. Full trainingtime recorded in paired results; acquisition/layout changes of subsequent candidate disclosed.'),indent=2)+'\n')
 if not eligible:print('Stopped: local quality needs repair '+str(delta),flush=True);return
 ref='aws_local_clip_receiver_d8_8m_20261014_s6_v1';r=complete(f'experiments/results/token_language/{ref}.json');sources=dict(r['source_sha256']);sources.pop('experiments/local_clip_receiver_token_language.py')
 driver='experiments/local_horizon_receiver_token_language.py'
 for name in [driver,'sleeping_machines/route_credit_horizon.py','sleeping_machines/learning/clone_local_receiver_learner.py','experiments/credit/check_local_learner_clone.py','experiments/local_clip_receiver_token_language.py']:sources[name]=queue.sha(ROOT/name)
 gate=PREFIX+'_law';result=f'experiments/results/credit/{gate}.json'
 checked=admit(gate,'experiments/credit/check_local_learner_clone.py','--tag '+gate,result,sources,1200);assert checked['gate']=='PASS'
 template=(ROOT/f'experiments/queue/r1/{ref}/{ref}.txt').read_text().splitlines()[1].split('.py ',1)[1]
 smoke=PREFIX+'_smoke';args=template.replace(ref,smoke).replace('--steps 16384','--steps 64').replace('--eval-every 1024','--eval-every 64')+' --consequence-windows 4'
 s=admit(smoke,driver,args,f'experiments/results/token_language/{smoke}.json',sources,5400,[dict(result=result)])
 assert s['max_rss_kb']<2500000
 for layer in s['curve'][-1]['receiver_gradient_norms']['banks.input']:assert sum(v for head in layer for v in head)>0
 timeout=math.ceil(2*s['train_wall_s_total']/s['presentations_total']*8388608+3600)
 fit=PREFIX+'_8m';args=template.replace(ref,fit)+' --consequence-windows 4'
 fitted=admit(fit,driver,args,f'experiments/results/token_language/{fit}.json',sources,max(timeout,5400),[dict(result=f'experiments/results/token_language/{smoke}.json')])
 sources=dict(fitted['source_sha256'])
 for name in ['experiments/check_deep_language_quality.py','experiments/public_token_scoring.py']:sources[name]=queue.sha(ROOT/name)
 tag=PREFIX+'_quality';quality=admit(tag,'experiments/check_deep_language_quality.py',f'--tag {tag} --run {fit}',f'experiments/results/diagnostics/{tag}.json',sources,3600)
 # Stronger completedD2 is retained as an external input; never edit its scorer.
 stronger=complete('experiments/results/diagnostics/aws_multidepth_8m_error_buckets_20261013_v1.json')
 q=quality['rows'][-1];best=stronger['rows'][-1]
 for k in ('targets','target_start','target_end_inclusive','window','lanes','eos_state_reset'):assert q['scored'][k]==best['scored'][k]
 summary=dict(status='completed',battle='R1',candidate_nll=q['scored']['nll'],gain_vs_local=l['scored']['nll']-q['scored']['nll'],gain_vs_global=g['scored']['nll']-q['scored']['nll'],gain_vs_richer_d2=best['scored']['nll']-q['scored']['nll'],full_training_s=q['full_training_s'],acquisition=q['acquisition'],scope='One combined continuous/noEOSreset/crossdetach routecredit candidate, final weights/seed6. Acquisition8388608vs8388352 and layout differ; not isolatedhorizon effect. No publicvalidation/energy/scaledTransformer win.',source_sha256={'scripts/continue_locality_frontier.py':queue.sha(Path(__file__))},input_sha256={f'experiments/results/diagnostics/{tag}.json':queue.sha(ROOT/f'experiments/results/diagnostics/{tag}.json'),'experiments/results/diagnostics/aws_multidepth_8m_error_buckets_20261013_v1.json':queue.sha(ROOT/'experiments/results/diagnostics/aws_multidepth_8m_error_buckets_20261013_v1.json')})
 with (ROOT/f'experiments/results/diagnostics/{PREFIX}_decision.json').open('x') as f:json.dump(summary,f,indent=2)
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
