"""Read completed evidence and apply the initial-inclusive promotion rule."""
import json
from pathlib import Path
from token_promotion_gate import assess
ROOT=Path(__file__).resolve().parents[1]
tags=['curie_integrated_2k_local_20261005_v1','curie_integrated_2k_full_20261005_v1',
      'curie_integrated_2k_k4_20261005_v1','curie_event_credit_2k_k4_20261005_v1']
rows={tag:assess(json.loads((ROOT/'experiments/results/token_language'/(tag+'.json')).read_text())) for tag in tags}
out=ROOT/'experiments/results/diagnostics/token_promotion_audit_20261005.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(dict(status='completed',rows=rows,
    scope='Completed seed6 development fits; practical gain threshold, not statistical significance. Initial model is eligible; no scaling prediction.'),indent=2)+'\n')
print(json.dumps(rows),flush=True)
