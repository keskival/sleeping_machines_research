"""Unchanged four-arm common-source learning on fixed fresh systems."""
import argparse
import copy
import json
import resource
import time
import torch
from public_stream import persistence
from persistence_world_credit import PersistenceWorldCreditSensor
from world_credit_sensor import WorldCreditSensor
from world_credit_run import run
from world_credit_confirmation_verify import verify_result
from world_credit_pipeline import ROOT,RESULTS,ACQUISITION,sha,fingerprint
PREPARE='curie_world_credit_confirm_20261012_v1_prepare'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,choices=[194,195,196],required=True)
 ap.add_argument('--arm',choices=['exact','learned','prior','persistence'],required=True);ap.add_argument('--case',type=int,required=True);a=ap.parse_args()
 assert 0<=a.case<8;torch.set_num_threads(1);begin=time.monotonic();out=RESULTS/(a.tag+'.json');assert not out.exists()
 prepare=verify_result(PREPARE);source_tag=ACQUISITION if a.seed==194 else f'curie_world_credit_confirm_20261012_v1_s{a.seed}_acquisition'
 source=verify_result(source_tag);assert source['checks']['acquisition_gate']
 cp=ROOT/source['checks']['checkpoint'];assert sha(cp)==source['checks']['checkpoint_sha256']
 saved=torch.load(cp,map_location='cpu',weights_only=False)
 cls=PersistenceWorldCreditSensor if a.arm=='persistence' else WorldCreditSensor
 torch.manual_seed(a.seed);model=cls(saved['confidence']);model.load_state_dict(saved['model'])
 optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
 incoming=fingerprint(dict(model=model.state_dict(),optimizer=optimizer.state_dict()));assert incoming==source['checks']['learner_fingerprint']
 row=prepare['records'][a.case];path=ROOT/row['path'];assert sha(path)==row['sha256']
 measured=run(model,path,'learned' if a.arm=='persistence' else a.arm,optimizer,seed=777+a.case)
 assert measured['targets']==128
 measured.update(path=row['path'],depth=row['depth'],width=row['width'],variant=row['variant'],incoming_learner_fingerprint=incoming)
 if a.arm=='persistence':
  independent,_=persistence(path,(model.persistence_correct,model.persistence_total),True)
  expected={qid:p if bit else 1-p for _,qid,bit,p in independent['issued_predictions']}
  error=max(abs(r['probability']-expected[r['query_id']]) for r in model.last_persistence_receipts);assert error==0.
  measured.update(persistence_credit_receipts=model.last_persistence_receipts,persistence_prediction_parity_error=error)
 sources=dict(source['source_sha256']);sources.update(prepare['source_sha256'])
 for n in ['experiments/causal_induction/persistence_world_credit.py','experiments/causal_induction/world_credit_confirmation_case.py','experiments/causal_induction/world_credit_confirmation_verify.py']:sources[n]=sha(ROOT/n)
 result=dict(status='completed',battle='B5/R1',tag=a.tag,seed=a.seed,arm=a.arm,case=a.case,rows=[measured],targets=128,
  source_sha256=sources,source_checkpoint_sha256=sha(cp),data_records=[row],
  parent_sha256={str((RESULTS/(source_tag+'.json')).relative_to(ROOT)):sha(RESULTS/(source_tag+'.json')),str((RESULTS/(PREPARE+'.json')).relative_to(ROOT)):sha(RESULTS/(PREPARE+'.json'))},
  whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  scope='Unchanged native world-credit learning on registered fresh public systems; all four controls retain native outcome learning/shared memory. Same within-seed incoming actor/world/Adam source. Fresh numeric episode state and causal8-feedback policy. All target work charged; generating depth, not neural depth.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=a.tag,nll=measured['nll'],world_nll=measured['world_nll'],wall_s=measured['wall_s'])),flush=True)
if __name__=='__main__':main()
