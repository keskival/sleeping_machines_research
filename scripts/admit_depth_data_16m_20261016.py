"""Admit only AWS L4/L16 ordered depth-data tests, through the six-slot gym."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 state=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text());assert not state['active'],'inspect active source-bound work before admission'
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 sources=['experiments/r1_deep_keyed_stream.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py','experiments/hierarchy/depth_data_resource_gate.py']
 declaration='''Battle R1, ordered in PRODUCT_ORDERS and SCALING_SUPREMACY_PROTOCOL. Decision: does useful native depth at 16M fresh tokens extend beyond L8, compared with Curie L2/L8 at the identical one-pass recipe? Strongest composition with demonstrated one-pass depth gain: plan-B keyed temporal-memory native member. This is a depth-data development comparison, not a family benchmark verdict. Stationarity: fixed-corpus fixed-weight evaluation; train loss is scored before each update on fresh lanes. Online weights adapt during acquisition only; evaluation weights frozen by this separately labelled protocol. State: decaying/rotating temporal state and keyed predecessor history persist within 256-token lanes, exact chunk carry and full-lane credit; reset across independently shuffled lanes; evaluation uses declared 1024-token windows. These boundaries match the existing depth-data experiment and do not establish continuous-stream competence. Retains separate keys/values, per-layer keyed race-attention, sparse token-slot writes, temporal integration, additive residual depth. Soft reads/full backprop; no hard route counterfactual credit, autonomous emission timers, silence likelihood, or cross-lane persistent state. Native continuous event hierarchy remains a separate integrated construction. Resources: six-slot one-thread runner,3M KiB RSS/12M KiB VMS per slot,8GiB host reserve; smoke <2.55M KiB RSS,32 updates before admission. All smokes/compilation/backward/optimizer work charged; no iso-compute or hardware claim. Exact resumable checkpoints every256updates. No new external-model training,public test selection or further automatic size ladder.'''
 slots={}
 for slot,depth in [(1,4),(2,16)]:
  jobs=[];gate=f'aws_r1_depthdata_L{depth}_16m_gate_20261016_v1';fit=f'aws_r1_depthdata_L{depth}_16m_s0_20261016_v1'
  for tag,script,args,result,requires in [(gate,'experiments/hierarchy/depth_data_resource_gate.py',f'--tag {gate} --layers {depth}',f'experiments/results/diagnostics/{gate}.json',[]),(fit,'experiments/r1_deep_keyed_stream.py',f'--tag {fit} --layers {depth} --train-tokens 16777216 --seed 0 --compile-blocks 1',f'experiments/results/token_language/{fit}.json',[dict(result=f'experiments/results/diagnostics/{gate}.json',stage='depth_data_resource_gate')])]:
   q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True)
   text='# '+declaration+'\n'+tag+' '+script+' '+args+'\n'
   if q.exists():assert q.read_text()==text
   else:q.write_text(text)
   jobs.append(dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sources},result=result,requires=requires,timeout_s=1800 if tag==gate else 100000,rss_kb=3000000,vms_kb=12000000))
  slots[str(slot)]=jobs
 packet=dict(host=os.uname().nodename,battle='R1',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots)
 path=PLAN/'addenda/000_aws_depthdata_16m_20261016_v1.json'
 if path.exists():assert json.loads(path.read_text())==packet
 else:
  tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
 print('Admitted L4/L16 resource gates,then conditional 16M fits in slots1/2.')
if __name__=='__main__':main()
