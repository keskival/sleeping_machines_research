"""R1 exact finite-sum and nonlinear task-credit counterexample contracts; no fitting."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,pytest
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
assert pytest.main(['-q','tests/test_composition_task_credit.py','tests/test_joint_race_credit.py','tests/test_sampled_compositional_credit.py'])==0
sources=['experiments/hierarchy/composition_credit_analytic_gate.py','tests/test_composition_task_credit.py','tests/test_joint_race_credit.py','tests/test_sampled_compositional_credit.py','sleeping_machines/composition_task_credit.py','sleeping_machines/joint_race_credit.py','sleeping_machines/coupled_native_query_credit.py','sleeping_machines/sampled_compositional_credit.py','sleeping_machines/compositional_producer_credit.py','tests/test_sampled_compositional_credit.py','tests/test_compositional_producer_credit.py','tests/test_native_query_migration.py']
r=dict(status='completed',battle='R1',stage='composition credit analytic gate',gate='PASS',elapsed_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Finite conditional joint-choice task-loss estimator and nonlinear tangent-bias witnesses;existing joint winner/time contracts retained. No native suffix-replay integration,learning improvement or full-history gradient claim.')
with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
