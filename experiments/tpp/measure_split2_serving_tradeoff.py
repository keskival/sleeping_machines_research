"""B4 completed compact/pair serving comparison, entire normalized mixture charged."""
import argparse,json,time,hashlib,resource,random,statistics,math
from pathlib import Path
import torch,numpy as np
from check_single_readout_serving import ROOT,adapter,restore
from race_tpp_logspread import RaceTPP as Stable
from race_tpp_logspread_marks import RaceTPP as MarkModel
from single_mark_readout import distributions

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def score(fn,dev):
 tick=time.monotonic();rows=[];error=0.
 for seq in dev:
  t,m,mask=next(adapter.base.batches([seq],1,False,random.Random(0)));lt,lp=fn(t,m,mask);valid=mask[:,1:];error=max(error,float((lp[valid].exp().sum(-1)-1).abs().max()));selected=lp.gather(-1,m[:,1:,None]).squeeze(-1);rows.append([-float(lt[valid].sum()),-float(selected[valid].sum())])
 lt,lm=np.mean(rows,0);return dict(L_T=float(lt),L_M=float(lm),total=float(lt+lm),normalization_error=error,wall_s=time.monotonic()-tick)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();base=ROOT/'experiments/results/tpp_b4';proof_path=base/'aws_b4_split2_observation_serving_20261011.json';proof=json.loads(proof_path.read_text());teacher_path=base/'aws_b4_split2_mark_confidence_20261011.json';teacher=json.loads(teacher_path.read_text());assert proof['status']==teacher['status']=='completed' and teacher['diagnostic_gate'];bundle_path=ROOT/proof['rows'][0]['bundle'];assert digest(bundle_path)==proof['rows'][0]['bundle_sha256'];bundle=torch.load(bundle_path,weights_only=True);student=restore(bundle);models=[Stable(*bundle['core_args']).eval(),MarkModel(*bundle['core_args']).eval()]
 inputs={str(p.relative_to(ROOT)):digest(p) for p in [proof_path,teacher_path,bundle_path]}
 for model,tag in zip(models,['aws_b4_split2_logspread_develop_20261011','aws_b4_split2_marks_develop_20261011']):
  record_path=base/f'{tag}.json';record=json.loads(record_path.read_text());path=ROOT/record['best_checkpoint'];assert digest(path)==teacher['checkpoint_sha256'][str(path.relative_to(ROOT))];model.load_state_dict(torch.load(path,weights_only=True));inputs[str(path.relative_to(ROOT))]=digest(path);inputs[str(record_path.relative_to(ROOT))]=digest(record_path)
 alpha=teacher['alpha_train_locked'];dev,K=adapter.load('github_filtered',2,'val')
 def paired(t,m,mask):
  time_lp,b,_=distributions(models[0],t,m,mask);_,j,_=distributions(models[1],t,m,mask)
  return time_lp,torch.logaddexp(b+math.log1p(-alpha),j+math.log(alpha))
 measured={'student':[],'pair':[]};scores={}
 for repeat in range(6):
  for name in (['student','pair'] if repeat%2==0 else ['pair','student']):
   result=score(student.event_distributions if name=='student' else paired,dev);scores[name]=result
   if repeat:measured[name].append(result['wall_s'])
 for name,reference in [('student',proof['rows'][0]['score']),('pair',teacher['dev'])]:
  assert max(abs(scores[name][k]-reference[k]) for k in ['L_T','L_M','total'])<1e-9 and scores[name]['normalization_error']<1e-12
 sources=['experiments/tpp/measure_split2_serving_tradeoff.py','experiments/tpp/check_single_readout_serving.py','experiments/tpp/single_mark_readout.py','experiments/tpp/nonlinear_single_mark_readout.py','experiments/tpp/race_tpp_logspread.py','experiments/tpp/race_tpp_logspread_marks.py','experiments/tpp/race_tpp_v19.py','experiments/tpp/race_tpp_b4g.py'];s=statistics.median(measured['student']);p=statistics.median(measured['pair'])
 result=dict(status='completed',tag=a.tag,battle='B4 split2 full serving boundary',scores=scores,student_median_s=s,pair_median_s=p,speedup=p/s,timing_samples=measured,student_parameters=sum(x.numel() for x in student.parameters()),pair_parameters=sum(x.numel() for model in models for x in model.parameters())+1,targets=sum(len(m)-1 for t,m in dev),sequences=len(dev),source_sha256={x:digest(ROOT/x) for x in sources},input_sha256=inputs,data_sha256=dict(val=digest(adapter.DATA/'github_filtered'/'split_2/val.json')),wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One Github split2 DEV seed0, CPU float64/one thread; five warmed timings/alternating order, complete normalized full-mark conditional-density and scoring boundary including scalar mixture. Student and teacher scores exactly reproduce completed results. Restoration separate from timed serving, all training/pilot/preparation costs inherited. No TEST/production-streaming/whole-training or broad Transformer advantage claim.')
 with (base/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
