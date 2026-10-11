"""Completed source-bound local reciprocal teaching/update contracts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. B1 reciprocal forward teaching and native actor-update closure'
RESULTS=['aws_reciprocal_tracking_20261010T2157Z','aws_reciprocal_cut_contract_20261010T2200Z',
         'aws_reciprocal_projection_20261010T2204Z']


def direct_policy_blocks():
    tags=[f'aws_direct_audit_policy_20261011_s{s}' for s in range(3)]
    if not all((ROOT/'experiments/results/credit'/f'{t}.json').exists() for t in tags):return []
    records=[json.loads((ROOT/'experiments/results/credit'/f'{t}.json').read_text()) for t in tags]
    rows=[]
    for record in records:
        assert record['status']=='completed'
        for source,sha in record['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        for r in record['rows']:
            rows.append([str(record['seed']),str(r['depth']),str(r['train_contexts']),f"{100*r['reduction_vs_uniform']:.1f}%",f"{100*r['reduction_vs_cotangent_norm']:.1f}%",f"{100*r['reduction_vs_mse']:.1f}%"])
    return [('p','Direct learned audit allocation wins the conditional correction-variance component against uniform and credit-magnitude allocation in all27 seed/depth/data-size cases. The policy trains the allocation fitness from saved forward-response labels; the scalar budget threshold supplies its analytic implicit gradient, closing policy credit without an additional learned trainer.'),
            ('table',(['Seed','Depth','Fit contexts','vs uniform','vs magnitude','vs risk-MSE'],rows,[15,18,28,30,30,32])),
            ('p','Protocol: three seeds, depths2/4/8,64/256/1024 acquisition contexts; fresh128 private TRAIN contexts per case,16 expected audits, positive support.13 policy parameters,300 Adam steps. Native actor and acquired credit fields are frozen. All methods in a row use the same contexts and budget. The endpoint is analytically measured conditional raw-gradient variance; actor backward, forward probes and exact evaluation work are charged in the preserved records. No sampled learner-update or future task-quality endpoint is reported here. Direct policy and risk-MSE use different probability parameterizations.'),
            ('p','Constructive transfer/stability/future-learning theory: experiments/theory/RECIPROCAL_CREDIT_TRANSFER_STABILITY_FUTURE_LEARNING.md and report/reciprocal_credit_theory.pdf. Current-state packet geometry transports reusable influence; audits contract credit-tracking error without erasing forward memory; future-demand-weighted information gain selects useful exploration. For a quadratic future loss with predictable update geometry, weighted correction-variance reduction gives an exact expected future-loss improvement. These sufficient conditions define the next native integration.'),
            ('p','Evidence: '+', '.join(tags)+'.')]


def packet_blocks():
    tags=['aws_native_cached_jvp_20261011_v2','aws_native_cached_linearization_20261011_v1','aws_native_packet_actor_update_20261011_v1','aws_packet_actor_step_20261011_v1','aws_packet_actor_teaching_step_20261011_v1']
    if not all((ROOT/'experiments/results/credit'/f'{t}.json').exists() for t in tags):return []
    records=[]
    for tag in tags:
        r=json.loads((ROOT/'experiments/results/credit'/f'{tag}.json').read_text());assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        records.append(r)
    jvp,vjp,actor,ordinary,teaching=records
    rows=[[str(f['events']),str(f['input_dimension']),f"{f['whole_speedup']:.2f}x",f"{b['whole_reverse_speedup']:.2f}x",str(b['cached_floats'])] for f,b in zip(jvp['rows'],vjp['rows'])]
    return [('p','A shared native primal packet now supports exact input tangents for forward teaching and exact input cotangents for producer credit, without constructing the dense cut Jacobian. Temporal decay/rotation, gated writes, key-independent local memory and normalization derivatives are retained. This is an economical exact local reference, not a learned or hard-route credit result.'),
            ('table',(['Events','Cut dimension','Forward probe speed','Reverse speed','Cached floats'],rows,[21,29,35,31,37])),
            ('p','Speed is measured whole capture plus16 batched directions/cotangents versus native batched autodiff (forward reference disables unnecessary reverse-graph recording), warmed5-repeat CPU medians. At32 events the cache contains7,744 floats; a dense1,024-dimensional Jacobian would contain1,048,576. Cache counts exclude shared model, input/dt and query buffers; autograd memory is unmeasured.'),
            ('p',f"Real Taxi TRAIN actor integration: four updates on one four-event prefix, all parameter gradients agree within {max(h['gradient_error'] for h in actor['history']):.2e} normalized error, Adam within {max(h['adam_error'] for h in actor['history']):.2e}. One tracked forward capture retains the layer/head parameter graph; its packet cotangent reconnects embedding/gap producers. No extra forward probe sweep. Gate wall {actor['wall_s']:.3f}s, peak RSS {actor['max_rss_kb']/1024:.1f}MiB; graph and packet work are charged."),
            ('p','Scope: dense-write native layer, fixed elapsed times, input derivatives only. Four actor updates score12 TRAIN target presentations per arm; no DEV/TEST quality endpoint, full fitting FLOPs, per-target fitting work or serving-work measurement. Parameter derivatives remain exact autodiff, with its graph retained. No whole-fit/depth/sparse-learning efficiency claim. Stale parameter versions are rejected. Next: use this exact packet reference for measured compact-credit teaching/audit integration.'),
            ('p','Complete actor-step measurement (aws_packet_actor_step_20261011_v1): reference/packet 2.432/3.063ms at B1,d8,n4,L4 and7.030/9.136ms at B4,d32,n16,L32. Retaining the parameter graph plus packet makes ordinary learning26–30% slower; native autodiff remains the ordinary-fit reference. Extra teaching-query gains are separate.'),
            ('p','Complete actor update plus16 forward teaching responses: '+', '.join(f"B{r['batch']},L{r['events']}: {r['reference_step_s']*1000:.3f}/{r['packet_step_s']*1000:.3f}ms reference/packet ({r['whole_step_speedup']:.2f}x)" for r in teaching['rows'])+'. Response parity <=3.56e-15 and parameter parity <=5.56e-17. Single-layer real Taxi TRAIN, five warmed timings; critic fitting/use is excluded. This is the label-production boundary, not whole-fit learning efficiency.'),
            ('p','Shared-packet evidence: '+', '.join(tags)+'.')]


def crossover_blocks():
    tags=['aws_packet_teaching_crossover_20261011_v1','aws_connected_teaching_crossover_20261011_v1','aws_connected_teaching_zero_query_20261011_v1','aws_connected_deep_teaching_20261011_v1','aws_connected_deep_teaching_scale_20261011_v1','aws_connected_teaching_size_scale_20261011_v2']
    if not all((ROOT/'experiments/results/credit'/f'{t}.json').exists() for t in tags):return []
    records=[]
    for tag in tags:
        r=json.loads((ROOT/'experiments/results/credit'/f'{tag}.json').read_text());assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        records.append(r)
    rows=[]
    for a,b in zip(*[r['rows'] for r in records[:2]]):
        rows.append([f"B{a['batch']}/L{a['events']}",str(a['teaching_directions']),f"{a['whole_step_speedup']:.2f}x",f"{b['whole_step_speedup']:.2f}x"])
    return [('p','Teaching-only actor execution now keeps the native producer graph connected while capturing the forward packet. This removes manual producer reconnection; all parameter gradients and Adam moments match the native reference exactly in the connected sweep.'),
            ('table',(['Shape','Queries','Detached packet speed','Connected packet speed'],rows,[29,20,49,50])),
            ('p','Composed native packets at depths2/4/8/16/32: '+', '.join(f"depth{x['depth']},queries{x['teaching_directions']}: {x['whole_step_speedup']:.2f}x" for x in records[3]['rows']+records[4]['rows'])+'. All parameter gradients and Adam moments match exactly; full composed tangent responses are checked against native forward autodiff. B4/L32/d32/n16, complete actor-update and teaching-label boundary, not learned-credit or task-quality scaling.'),
            ('p','Larger d64/n32/B4/L32 models: '+', '.join(f"depth{x['depth']},{x['parameters']:,} parameters,{x['teaching_directions']} queries: {x['whole_step_speedup']:.2f}x" for x in records[5]['rows'])+'. Recorded gradient/Adam/parameter differences0; maximum teaching response error9.77e-15. Whole diagnostic31.60s, peak RSS617.4MiB. Initial64-event fixture failed before fitting because Taxi TRAIN ends at38 events; corrected v2 holds length32. This measures model width/memory/depth scaling of label production, not sequence-length scaling.'),
            ('p','Complete actor/update plus1–32 teaching directions, five warmed CPU timings, real Taxi TRAIN and one dense-write layer. Captured packet, retained graph and symmetric response copies are charged; connected timing additionally charges symmetric gradient copies. No critic fitting, task-quality or whole-fit learning-efficiency endpoint. Zero-query capture runs5–7% slower than native, while1/16-query confirmations run2.40–3.89x faster. Enable capture for teaching work, keep native execution for ordinary updates. Evidence: '+', '.join(tags)+'.')]

def forward_risk_blocks():
    tag='aws_native_forward_residual_risk_20261011_v1';path=ROOT/'experiments/results/credit'/f'{tag}.json'
    if not path.exists():return []
    r=json.loads(path.read_text());assert r['status']=='completed'
    for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
    return [('p','Forward teaching now supplies credit-risk observations: delta=c^T(Jv)-predicted_credit^T v. Isotropic probe squared delta estimates cut residual energy; producer-tangent probes v=A xi estimate the actual producer correction energy ||A^T residual||^2 without storing A. The same observations train conditional audit risk; no reverse teacher is used in label construction.'),
            ('p',f"Native four-event Taxi TRAIN/one-layer contract, actual observed-loss cotangent: {r['input_probe_count']} cut and {r['producer_probe_count']} producer directions/{r['producer_parameter_count']} producer parameters. Mean input-risk error {r['input_risk_error']:.2e}, producer-risk error {r['producer_risk_error']:.2e}; issued softplus risk-training gradient error {r['risk_training_gradient_error']:.2e}. Label acquisition {r['label_s']*1000:.3f}ms, separate reverse validation {r['reverse_validation_s']*1000:.3f}ms, whole gate {r['wall_s']:.3f}s."),
            ('p','Exhaustive orthogonal probes validate the expectation identities; predicted_credit=0.5c is a diagnostic. No learned-risk calibration, sparse-audit utility, quality or efficiency win. Existing teaching responses can be reused, but fresh risk validation, producer directions and label reductions are paid. Predictor/actor versions are frozen before observation; no reuse of fit probes as independent validation. Theory: FORWARD_OBSERVED_CREDIT_RISK. Evidence: '+tag+'.')]

def temporal_packet_blocks():
    tag='aws_native_temporal_teaching_packet_20261011_v1';path=ROOT/'experiments/results/credit'/f'{tag}.json'
    if not path.exists():return []
    r=json.loads(path.read_text());assert r['status']=='completed'
    for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
    return [('p','Joint content/elapsed-time teaching extends the shared native packet: exact tangent responses and both input/time cotangents from existing primal caches, with2*B*L*n additional forcing floats. Native autodiff already supports time; this exposes its derivative through the reusable teaching interface, without another forward capture.'),
            ('p',f"One layer at4/8/32 events with real Taxi TRAIN gaps: JVP error {max(x['jvp_error'] for x in r['rows']):.2e}, VJP {max(x['vjp_error'] for x in r['rows']):.2e}, joint reciprocity {max(x['reciprocity_error'] for x in r['rows']):.2e}. Zero time directions exactly preserve input-only responses; clock changes invalidate issued packets. Cache280/560/2240 floats, excluding shared/query/graph storage."),
            ('p','Elapsed intervals use layer-scaled time; upstream physical-time/delay producers need their chain rule. Eval/no-dropout first-order contract, no hard-route/parameter/higher-order packet or learned-credit utility/speed/quality claim. Active fits unchanged. Evidence: '+tag+'.')]

def timed_producer_blocks():
    tag='aws_timed_producer_actor_20261011_v1';path=ROOT/'experiments/results/credit'/f'{tag}.json'
    if not path.exists():return []
    r=json.loads(path.read_text());assert r['status']=='completed'
    for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
    h=r['history']
    return [('p',f"Joint content/time packet reconnects a diagnostic positive internal-tempo producer on a real four-event Taxi TRAIN prefix across four Adam updates. Maximum gradient error {max(x['gradient_error'] for x in h):.2e}, Adam-state error {max(x['adam_error'] for x in h):.2e}, parameter error {max(x['parameter_error'] for x in h):.2e}; temporal producer gradients are nonzero. Native layer/head parameter graph retained; physical race/addressed-memory clocks unchanged."),('p','Exact producer-credit execution contract, no active architecture substitution or learned-credit quality/speed claim. Evidence: '+tag+'.')]

def deep_joint_blocks():
    tags=['aws_deep_temporal_packet_20261011_v1','aws_fused_deep_temporal_packet_20261011_v1',
          'aws_fused_deep_temporal_teaching_cost_20261011_v1','aws_joint_forward_risk_20261011_v1']
    records=[]
    for tag in tags:
        path=ROOT/'experiments/results/credit'/f'{tag}.json'
        if not path.exists():return []
        r=json.loads(path.read_text());assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        records.append(r)
    exact,fused,cost,risk=records
    rows=cost['rows'];teaching=[x for x in rows if x['teaching_directions']];ordinary=[x for x in rows if not x['teaching_directions']]
    return [('p','Shared elapsed-time credit across depth sums each layer contribution once: v_l=A_l v_(l-1)+B_l w, c_(l-1)=A_l^T c_l, g_time=sum_l B_l^T c_l. Native depths2/4/8 at4/32 events pass joint transport, suffix causality and stale intermediate-clock rejection. Four real eight-event Taxi TRAIN Adam updates per depth reconnect all content/time producers, layer and head parameters; maximum gradient error '+format(max(x['gradient_error'] for x in exact['actor_rows']),'.2e')+'.'),
      ('p',f"Fused joint transport shares recurrence/normalization/readout work. Complete actor/loss/backward/Adam plus1/16 joint teaching queries at depths2/4/8/32 is {min(x['whole_step_speedup'] for x in teaching):.2f}–{max(x['whole_step_speedup'] for x in teaching):.2f}x faster than native joint forward AD. Five timed steps after three warmups, alternating order, same batches/updates, exact gradient/Adam/response parity. Whole gate {cost['wall_s']:.2f}s/{cost['max_rss_kb']/1024:.1f}MiB. Retained cache grows70,144 to1,122,304floats; shared weights/graph additional."),
      ('p',f"Zero-query speed ratio {min(x['whole_step_speedup'] for x in ordinary):.3f}–{max(x['whole_step_speedup'] for x in ordinary):.3f}: capture is slower for ordinary fitting. Enable it for needed teaching. Native dense-write diagnostic, physical race/addressed-memory clocks retained; no critic fitting/use, task quality, full-learning or Transformer efficiency claim."),
      ('p',f"Forward-only joint residual labels retain shared-producer content/time cross terms at depths2/4/8. Combined producer-risk error {max(x['producer_risk_error'] for x in risk['rows']):.2e}, conditional risk-training gradient error {max(x['risk_training_gradient_error'] for x in risk['rows']):.2e}. Separate reverse validation paid; exhaustive orthogonal probes and diagnostic positive internal tempo. No learned calibration/audit utility claim. Evidence: "+', '.join(tags)+'.')]

def learned_joint_blocks():
    tags=['aws_joint_credit_training_20261011_v1','aws_joint_credit_identity_20261011_v1']
    tags += [f'aws_joint_credit_acquisition_20261011_s{seed}' for seed in range(3)]
    tags += [f'aws_native_joint_audit_risk_20261011_s{seed}' for seed in range(3)]
    records=[]
    for tag in tags:
        path=ROOT/'experiments/results/credit'/f'{tag}.json'
        if not path.exists():return []
        r=json.loads(path.read_text());assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        records.append(r)
    contract,identity=records[:2];acquired=records[2:5];risk=records[5:]
    rows=[row for record in acquired for row in record['rows']]
    out=[('p','Forward responses now train the joint credit estimator directly: squared mismatch between predicted input/time directional credit and the observed-loss response c^T Jv. Actor responses, probes and terminal cotangent are frozen for each issued observation; only estimator predictions receive local training gradients. At depths2/4/8, all1716parameter neural-estimator gradients match exact cut regression and shared-producer regression (including cross terms) within '+format(max(x['producer_gradient_error'] for x in contract['rows']),'.2e')+'. Four local SGD steps decrease in-sample error; no held-out utility is inferred from that contract.'),
      ('p',f"Local acquisition certificate: three seeds, depths2/4/8,512 artificial terminal cotangents with four fresh full-support probes each,400Adam updates,1152parameter linear credit map.64fresh cotangents and probes after fit. Joint-field residual energy versus zero falls {100*min(x['residual_energy_reduction'] for x in rows):.3f}–{100*max(x['residual_energy_reduction'] for x in rows):.3f}%. One frozen four-event Taxi TRAIN context/three unique task targets per actor, not512new task examples. Forward-only fit labels; reverse validation separate. All acquisition/fitting investment charged; no whole-learning or state-transfer claim."),
      ('p','Portable fields/actors restored without refitting, actual observed-loss cotangents: learned shared-producer credit beats identity transport in9/9same-context cases and9/9one-Adam model-update cases;16/18new-context cases improve and2worsen. At depth8/seed1, both new contexts lose to zero/identity. This diagnoses missing state conditioning; it is not a depth/family task-loss verdict. The actor predictions are identical across credit evaluations. Expired maps require revalidation/audits.'),
      ('p','First native conditional audit-risk fit:13parameter positive model,12pre-label state/clock/normalization features,64four-event TRAIN contexts,8forward shared-producer probes per context,300Adam updates.32new TRAIN contexts, expected4audits/floor0.01. Metric is conditional raw-gradient trace sum R(1/pi-1), analytic at identical expected audit budget; full actor backward used to obtain observed-loss cotangent and charged. No sampled sparse updates or future task learning measured.'),
      ('table',(['seed','depth','reduction vs uniform %','reduction vs cotangent norm %'],
       [[str(r['seed']),str(x['depth']),f"{100*x['reduction_vs_uniform']:.2f}",f"{100*x['reduction_vs_cotangent_norm']:.2f}"] for r in risk for x in r['rows']],[20,20,54,64])),
      ('p','The64-context pilot is mixed, including a depth8 deterioration; preserve every row. Next registered64/256/1024-context acquisition ladder keeps the estimator/configuration fixed, reuses forward labels and reserves a new128-context private TRAIN tail. All sizes reported, no best-size selection. Evidence: '+', '.join(tags)+'.')]
    return out + audit_ladder_blocks()


def audit_ladder_blocks():
    records=[]
    for seed in range(3):
        path=ROOT/'experiments/results/credit'/f'aws_native_joint_audit_risk_ladder_20261011_s{seed}.json'
        if not path.exists():return []
        r=json.loads(path.read_text());assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        records.append(r)
    return [('p','Registered acquisition-scale ladder completed: unchanged13parameter risk model/features/300steps at64/256/1024contexts; shared8-probe forward labels,128new private TRAIN tail contexts, expected16audits/12.5% with floor0.01. Earlier exploratory validation explicitly enters acquisition. Every size retained; exact references are neither fitting targets nor features. Full raw records and shared/per-size standalone work retained. Metric is analytic conditional raw-gradient trace, not future task loss or complete fitting efficiency.'),
      ('table',(['seed','depth','TRAIN contexts','vs uniform %','vs cotangent norm %'],
       [[str(r['seed']),str(x['depth']),str(x['train_contexts']),f"{100*x['reduction_vs_uniform']:.2f}",f"{100*x['reduction_vs_cotangent_norm']:.2f}"] for r in records for x in r['rows']],[16,16,34,38,54])),
      ('p','Measured whole ladder wall/RSS: '+', '.join(f"seed{r['seed']} {r['wall_s']:.2f}s/{r['max_rss_kb']/1024:.1f}MiB" for r in records)+'. Earlier local-field fits,64-context pilots and publication are additional. No Transformer/task-quality/whole-learning claim. Evidence: '+', '.join(r['tag'] for r in records)+'.')]

def pages():
    records=[]
    for tag in RESULTS:
        path=ROOT/'experiments/results/credit'/f'{tag}.json'
        if not path.exists():return []
        r=json.loads(path.read_text());assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        records.append(r)
    tracking,scalar,projected=records
    reduction=100*(1-tracking['mean_tracked_relative_error']/tracking['mean_frozen_relative_error'])
    rows=[]
    for label,r in [('Scalar teaching',scalar),('Packet projection',projected)]:
        rows.append([label,str(r['native_actor_updates']),str(r['forward_probes']),str(r['local_teaching_updates']),
                     f"{max(x['normalized_gradient_error'] for x in r['history']):.2e}"])
    return [[('h1',TITLE),
             ('p','Forward tangent packets now teach a backward transport map, which carries credit into native actor input parameters. '
              'Four successive parameter versions match the exact-gradient reference, including Adam state and parameter updates. '
              'The local teaching rule uses forward responses and outer products; it has no third learned teacher.'),
             ('table',(['Implementation','Actor steps','Probes','Teaching updates','Gradient error'],rows,[39,22,22,31,39])),
             ('p',f"Rank-one packet projection reduces teaching updates from {scalar['local_teaching_updates']:,} to {projected['local_teaching_updates']:,} with unchanged forward probes. "
              f"Maximum parameter difference {max(x['parameter_error'] for x in projected['history']):.2e}; maximum Adam-state difference {max(x['adam_state_error'] for x in projected['history']):.2e}. "
              f"In a separate 12-version tracking diagnostic, refreshing two input directions per version reduces mean relative squared transport error by {reduction:.1f}% versus a frozen map."),
             ('p',f"Accounting: projection gate forward teaching {projected['forward_teaching_s']:.3f}s; both actor/reference arms and audits {projected['two_arm_actor_and_audit_s']:.3f}s; "
              f"whole gate {projected['wall_s']:.3f}s, peak RSS {projected['max_rss_kb']/1024:.1f}MiB. Native forward loss is {projected['history'][0]['loss']:.6f} to {projected['history'][-1]['loss']:.6f} "
              'on the repeated TRAIN prefix. Dense cut transport stores 1,024 values; full fitting FLOPs and inference work are unmeasured.'),
             ('p','Scope: one four-event Taxi TRAIN prefix, one temporal layer. Cut credit trains embedding/gap producers; layer/head derivatives remain exact local operations. '
              'Every cut direction is refreshed in the actor-parity gate. This is a reciprocal-update contract, not a DEV-quality, sparse-learning, efficiency or benchmark win. '
              'Next: context-conditioned transport, cheaper probes and task-valued acquisition under moving weights.'),
             ('p','Evidence: '+', '.join(RESULTS)+'. Sources, data hashes, timings and all gate inputs are retained in completed JSON records.')] + packet_blocks() + crossover_blocks() + forward_risk_blocks() + temporal_packet_blocks() + timed_producer_blocks() + deep_joint_blocks() + learned_joint_blocks() + direct_policy_blocks()]
