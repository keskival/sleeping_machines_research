"""Current source-bound successor progress; exploratory Taxi plus saved development."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def pages():
    path=ROOT/'experiments/results/tpp/aws_taxi_successor_curiosity_20261014_s2_v1.json'
    if not path.exists():return []
    d=json.loads(path.read_text());assert d['status']=='completed'
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    parent=d['original_test']['ll'];control,candidate=d['rows']
    gain=candidate['test']['ll']-parent;over_control=candidate['test']['ll']-control['test']['ll']
    rows=[['Saved unified parent','0.488586',f'{parent:.6f}','—','141.823'],
          *[[r['arm'].replace('_',' '),f"{r['dev_ll']:.6f}",f"{r['test']['ll']:.6f}",f"{r['incremental_training_s']:.3f}",f"{r['whole_training_s']:.3f}"] for r in d['rows']]]
    points=[('h1','Appendix B. Successor progress: direct quality and work comparisons'),
        ('p',f'Taxi seed2 exploratory successor: TEST gain over saved parent{gain:+.6f}nats/event; over ordinary continuation{over_control:+.6f}. Published reference0.522; original parent0.521049. Both20epoch arms were fixed before fitting, selected on DEV including unchanged parent, and frozen before TEST scoring. This revisits a known losing seed and does not revise the original24/25 record.'),
        ('table',(['Model','Selected DEV LL','TEST LL','Extra fit seconds','Whole fit seconds'],rows,[60,36,32,32,32])),
        ('table',(['Model','Time LL','Mark LL','RMSE','Accuracy'],[[r['arm'].replace('_',' '),f"{r['test']['time_ll']:.6f}",f"{r['test']['mark_ll']:.6f}",f"{r['test']['rmse']:.6f}",f"{r['test']['acc']:.6f}"] for r in d['rows']],[60,32,32,32,32])),
        ('p',f"Parent prediction error{d['zero_prediction_error']:.3g}, original-gradient error{d['original_gradient_error']:.3g}. Retained delayed clocks, state clock, temporal decay/rotation, addressed mark memory, recording-cell rule and silence likelihood. Added one zero-output temporal residual stage with exact cached primal packet. Whole work includes inherited parent fitting; repeated benchmark scoring and contracts are additional."),
        ('p','Integration scope: compatible prediction-preserving temporal depth and exact native packet execution. Shared horizon critic is not fitted yet; token-specific kernels and market action heads are not transferred. Loss here is about this variant and data scale; larger-acquisition advantage remains a separate test, and numerical failures must be audited before interpreting quality.')]
    language=json.loads((ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_8m_20261013_s6_v2_gate.json').read_text())
    market=json.loads((ROOT/'experiments/results/market/curie_b10_future_credit_confirmation_20261013_v1.json').read_text())
    points.extend([('p',f"Saved language development:8M probability0.25 teacher improves final DEV by{-language['final_nll_delta']:.6f}nats/token while reducing full training time{100*(1-language['training_wall_ratio']):.2f}% ({language['reference_training_s']:.3f} to{language['candidate_training_s']:.3f}s). One seed at8M;3seeds at128k/512k. Larger-point time savings combine ordered aggregation and teacher admission. Public validation sealed; no external Transformer win."),
        ('p',f"Saved market development: three critic initializations mean16384DEV NLL{market['mean_nll']:.9f}; gain over factual eight-event BP{market['mean_gain']['bptt_carry']:.6f}, over matched own-refresh{market['mean_gain']['own_refresh']:.6f}nats/event. All three improve those controls; shared actor/data, extra teacher/critic/refresh work. Later addressed-action cohort is mixed and not promoted. No public benchmark or profit claim.")])
    return [points]
