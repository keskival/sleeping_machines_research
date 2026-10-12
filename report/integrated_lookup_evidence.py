"""Completed full-fit token lookup engineering evidence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def blocks():
    path=ROOT/'experiments/results/token_language/aws_integrated_p24_64k_lookup_decision_20261012_v1.json'
    if not path.exists():return []
    d=json.loads(path.read_text());assert d['status']=='completed'
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=[[r['arm'],f"{r['selected_dev_nll']:.6f}",f"{r['final_dev_nll']:.6f}",
           f"{r['training_wall_s']:.3f}",f"{r['engine_wall_s']:.3f}",f"{r['process_wall_s']:.3f}",
           f"{r['peak_rss_kib']/1024:.1f}"] for r in d['rows']]
    return [('p','Integrated R1 execution: batching vocabulary gathers inside the existing credit windows removes repeated dense embedding-gradient accumulation. All temporal events, hard races, separate keys/values, persistent numerical state, state-gradient boundaries, counterfactual suffix teaching and optimizer updates are retained.'),
        ('table',(['Lookup','Selected NLL','Final NLL','Train seconds','Engine seconds','Process seconds','RSS MiB'],rows,[24,28,28,29,29,29,24])),
        ('p',f"Selected quality gain {d['quality_gain']:+.6f} nats; ordinary/batched training wall {d['ordinary_over_batched_training_wall']:.3f}x. Registered development gate (quality retained and at least5% training-wall saving): {'passed' if d['development_gate_passed'] else 'failed'}."),
        ('p',d['scope']),
        ('p','Numerical integration gate: three AdamW updates with carried state, repeated token IDs, EOS resets and two credit windows; maximum forward/state error9.54e-7, gradient error1.20e-7, parameter error1.06e-7, optimizer error7.46e-9; RNG states equal. Evidence aws_integrated_lookup_parity_20261012_v1 and '+path.stem+'.')]
