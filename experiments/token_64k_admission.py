"""Read-only source/evidence admission for the reserved integrated64Kcell."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
diagnostics=ROOT/'experiments/results/diagnostics'
work=json.loads((diagnostics/'curie_fixed_batch_tokens_8k_c16_work_20261005_v1.json').read_text())
assert work['status']=='completed' and work['curve_parity_max_error']<2e-6
assert work['fitting']['formula_coverage_complete'] and work['inference']['formula_coverage_complete']
utility=json.loads((diagnostics/'curie_fixed_batch_tokens_8k_utility_20261005_v1.json').read_text())
assert utility['status']=='completed'
parents=[r for r in utility['rows'] if r['credit_window']==16]
assert {r['seed'] for r in parents}=={6,7}
assert all(r['promotion']['small_fit_promotable'] and r['context_gain']>0 for r in parents)
for row in parents:
    p=ROOT/'experiments/results/token_language'/(row['tag']+'.json')
    d=json.loads(p.read_text())
    for file,digest in d['identity']['source_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest, ('source changed',file)
assert json.loads((diagnostics/'curie_horizon_token_resume_contract_20261005_v1.json').read_text())['status']=='completed'
print(json.dumps(dict(status='admitted',next_queue='curie_data_growth_tokens_64k_b64_c16_p16_s6_20261005_v1',
    parent_seeds=[6,7],gate='Replicated8Klearning/positivecontext,actual-work parity/coverage,retained resume contract/source pins',
    scope='Admission,not completed64Kquality or work. Host guards enforced by run_safe.')))
