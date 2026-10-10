"""Completed native future-credit dynamics; no benchmark verdicts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. R1 future-directed credit: connected critic and candidate pools'


def pages():
    folder=ROOT/'experiments/results/credit';prefix='curie_future_credit_v1_20261010T1140Z'
    files=[folder/(prefix+f'_pilot_s{s}.json') for s in (170,171,172)]
    if not all(p.exists() for p in files):return []
    records=[json.loads(p.read_text()) for p in files]
    assert all(r['status']=='completed' for r in records)
    rows=[];work=[];pool=[]
    for seed,r in zip((170,171,172),records):
        for a in r['metrics']['arms']:
            rows.append([str(seed),a['arm'],f'{a["train"]["relative_mse"]:.3f}',
                         f'{a["dev"]["relative_mse"]:.3f}',f'{a["dev"]["selected_future_advantage"]:.5f}'])
            if 'learned_pool' in a:
                q=a['learned_pool'];pool.append([str(seed),f'{q["mean_pool_coverage_improvement"]:.6f}',
                    f'{q["mean_selection_regret"]:.5f}',f'{q["selected_future_advantage"]:.5f}'])
        targets=r['metrics']['teacher_scored_targets'];mf=r['supported_flops']/1e6
        work.append([str(seed),str(targets),f'{mf:.2f}',f'{mf/targets:.4f}',
                     f'{r["wall_s"]:.2f}',f'{r["peak_rss_kb"]/1024:.1f}'])
    return [[('h1',TITLE),
        ('p','A shared critic learns the future loss consequences of controlled interventions in an addressed key write. '
         'Both isolated and connected critics acquire TRAIN structure and select beneficial held-out interventions in '
         'three seeded pilots. Connected transfer is weaker than isolated transfer in this small-data regime; '
         'the connected variant uses 7,361 parameters versus 7,297 for isolated. Zero-prediction relative MSE is 1.0. '
         'Future advantage is realized continuation loss minus no-op loss, in nats per future target; negative is better.'),
        ('table',(['Seed','Critic','TRAIN MSE','DEV MSE','DEV advantage'],rows,[17,34,35,35,40])),
        ('small','Each seed uses a frozen initialized depth-two native temporal actor, 16 TRAIN and 16 DEV contexts, '
         '128 critic steps and fixed final checkpoints. Inputs include only revealed prefix information. '
         'Future branch outcomes supervise fitting and assess DEV choices, without entering those choices. '
         'Nine basis/no-op interventions are enumerated. These are state-intervention dynamics, not trained '
         'forward-weight adaptation, hard-routing or efficiency wins.'),
        ('p','A bounded proposal trained through the connected critic adds one candidate to the retained pool. '
         'Its realized coverage improvement is near zero. Separate coverage and selection-regret measurements '
         'expose whether the proposer finds useful alternatives and whether the critic ranks them correctly.'),
        ('table',(['Seed','Coverage gain','Selection regret','Pool advantage'],pool,[17,48,48,48])),
        ('p','The exact native receiver cotangent contract passes at depths 2, 4 and 8 with maximum error 1.39e-17. '
         'The future-intervention contract passes prefix causality, zero-action parity, unchanged earlier losses '
         'and directional finite differences, maximum error 1.39e-10.'),
        ('table',(['Seed','Teacher targets','Whole work MF','MF/teacher target','Wall s','RSS MiB'],work,[16,34,35,42,24,30])),
        ('small','Each work row covers shared TRAIN/DEV branch generation, both critic fits and proposal fitting/evaluation. '
         'The denominator is all teacher scored-target presentations, including context and branch recomputation; '
         'it is not unique training data. Supported PyTorch FLOPs exclude special/unsupported operations; wall includes '
         'profiling. Inference service work is not separately measured.'),
        ('p','Next diagnostic is admitted: 128 TRAIN and 64 DEV contexts, same actor and critic mathematics, '
         'three seeds, and a TRAIN-fitted mean-advantage reference that chooses one fixed action in every context. '
         'This distinguishes conditional credit from global action preference. A small-data isolated advantage is '
         'a testable sample-efficiency hypothesis; redundant history and recurrent optimization are competing explanations. '
         'Coupled forward learning, regime adaptation and depth/work scaling follow a useful replicated signal.')]] + diagnostic_pages(folder) + noise_pages(folder) + cpu_scan_pages(folder) + integrated_scan_pages(folder) + averaged_credit_pages(folder) + depth_credit_pages(folder)


def diagnostic_pages(folder):
    path=folder/'curie_future_credit_v2_20261010T1240Z_decision.json'
    if not path.exists():return []
    result=json.loads(path.read_text())
    if result['status']!='completed':raise ValueError('Completed credit decision evidence required')
    rows=[]
    for r in result['rows']:
        rows.append([str(r['seed']),r['arm'],f'{r["dev_relative_mse"]:.3f}',
            f'{r["constant_relative_mse"]:.3f}',f'{r["loss_reduction"]:.5f}'])
    work=[]
    for r in result['work']:
        targets=r['teacher_scored_target_presentations'];mf=r['whole_supported_flops']/1e6
        work.append([str(r['seed']),str(targets),f'{mf:.2f}',f'{mf/targets:.4f}',
                     f'{r["wall_s"]:.2f}',f'{r["peak_rss_kb"]/1024:.1f}'])
    decisions=' '.join(r['arm']+': '+r['decision'].replace('_',' ')+'.' for r in result['verdicts'])
    return [[('h1','Appendix. R1 conditional future credit: larger-data diagnostic'),
        ('p','128 TRAIN and 64 DEV contexts per fixed native teacher, three seeds, same critic architecture. '
         'The reference fits one mean-advantage vector on TRAIN and chooses one fixed action for every DEV context. '
         'Positive future-loss reduction means the critic improves on that action; MSE is relative to zero prediction.'),
        ('table',(['Seed','Critic','DEV MSE','Constant MSE','Future-loss reduction'],rows,[15,31,32,36,48])),
        ('p',decisions),
        ('small',result['scope']),
        ('table',(['Seed','Teacher targets','Whole work MF','MF/teacher target','Wall s','RSS MiB'],work,[16,34,35,42,24,30])),
        ('small','Whole supported work includes paired native branches, both critic fits and proposal training/evaluation. '
         'Teacher-target presentations are the common denominator; unsupported/special operation arithmetic is excluded. '
         'Passing the conditional gate enables a coupled-learning design, not an efficiency or benchmark claim.')]]


def noise_pages(folder):
    prefix='curie_credit_noise_v1_20261010T1450Z'
    files=[folder/(prefix+f'_measure_s{s}.json') for s in (170,171,172)]
    contract=folder/(prefix+'_contract.json')
    if not contract.exists() or not all(p.exists() for p in files):return []
    records=[json.loads(p.read_text()) for p in files];c=json.loads(contract.read_text())
    assert c['status']=='completed' and all(r['status']=='completed' for r in records)
    error=max(r['maximum_branch_parity_error'] for r in c['metrics']);rows=[];work=[]
    for seed,r in zip((170,171,172),records):
        m=r['metrics'];rows.append([str(seed),f'{m["context_signal_energy"]:.6g}',
            f'{m["mean_conditional_noise_energy"]:.6g}',f'{m["context_signal_fraction_of_raw_energy"]:.3f}'])
        targets=m['scored_target_presentations'];mf=r['supported_flops']/1e6
        work.append([str(seed),str(m['actor_forwards']),str(targets),f'{mf:.2f}',
                     f'{mf/targets:.4f}',f'{r["wall_s"]:.2f}'])
    return [[('h1','Appendix. R1 conditional credit: signal, noise and cached response'),
        ('p',f'Exact cached key-write intervention values match complete native branch replay at depths 2/4/8 '
         f'to {error:.2g}. This reuse applies because the intervention changes addressed mark scores while factual '
         'future hidden states, queries and timing clocks stay unchanged on fixed external inputs. General internal '
         'routing that alters those states requires a different calculation.'),
        ('p','Each teacher uses 16 independently generated prefixes and two independent groups of eight legitimate '
         'continuations per prefix. Cross-products of the two group means estimate conditional mean energy; '
         'cross-prefix products estimate global mean energy. Their difference measures signal beyond a constant '
         'predictor. Sample covariance measures unpredictable continuation noise. Finite signal estimates are '
         'reported without clipping; full-prefix conditioning does not prove current packet sufficiency.'),
        ('table',(['Seed','Context signal','Continuation noise','Signal/raw energy'],rows,[17,48,48,48])),
        ('table',(['Seed','Actor calls','Teacher targets','Whole MF','MF/target','Wall s'],work,[15,28,35,33,29,23])),
        ('small','Non-fitting synthetic diagnostic, no TEST. Actor calls include all causal-prefix checks; '
         'supported work includes all native calls and analytic candidate responses, with special/unsupported '
         'operation arithmetic excluded. No complete-learning efficiency or benchmark win is claimed.')]]


def cpu_scan_pages(folder):
    prefix='curie_cpu_scan_dispatch2_20261010T1640Z'
    cp=folder/(prefix+'_contract.json');sp=folder/(prefix+'_smoke.json')
    if not cp.exists() or not sp.exists():return []
    c=json.loads(cp.read_text());r=json.loads(sp.read_text())
    assert c['status']=='completed' and r['status']=='completed'
    native=c['metrics']['native'];error=max(v['maximum_parameter_gradient_error'] for v in native)
    rows=[]
    for v in r['metrics']['rows']:
        rows.append([str(v['batch']),str(v['length']),str(v['modes']),
            f'{v["timing_s"]["reference"]:.4f}',f'{v["timing_s"]["scan"]:.4f}',
            f'{v["runtime_reference_over_scan"]:.2f}'])
    return [[('h1','Appendix. R1/B10 exact CPU temporal execution'),
        ('p',f'Compiled complex recurrence and its first-order adjoint preserve native depths 2/4/8 '
         f'with maximum parameter-gradient error {error:.2g}. Explicit initial state and chunked carry '
         'preserve both numerical state and gradients. Projections, gates, physical decay/rotation, '
         'norms, feedforward network, addressed mark memory and race heads are unchanged.'),
        ('table',(['Batch','Length','Modes','Reference s','Scan s','Time ratio'],rows,[20,23,23,38,30,32])),
        ('small',r['metrics']['scope']),
        ('p','Kernel arithmetic is 8 scalar FLOPs per mode/event forward and 14 backward. Source '
         'coefficient arrays use 4 batch × length × modes values and the stored state trajectory 2. '
         'Array traffic, native projections, heads, mark-memory gradients, credit learning and optimizer '
         'work must be included before any complete-fit advantage is claimed.'),
        ('small',f'Contract wall including compilation {c["wall_s"]:.1f} s; parent peak RSS '
         f'{c["peak_rss_kb"]/1024:.1f} MiB; maximum individual compiler-child RSS '
         f'{c["peak_child_rss_kb"]/1024:.1f} MiB. Runner guards the whole process group. '
         'One compiler worker; CPU float64. Higher-order backward/meta-Hessians are unsupported by this '
         'backend and keep the established differentiable reference. No fitting run has adopted it.')]]


def integrated_scan_pages(folder):
    p=folder/'curie_integrated_scan_v2_20261010T1710Z.json'
    if not p.exists():return []
    r=json.loads(p.read_text());assert r['status']=='completed';rows=[]
    for v in r['metrics']:
        rows.append([str(v['depth']),str(v['length']),f'{v["median_step_s"]["reference"]*1000:.2f}',f'{v["median_step_s"]["scan"]*1000:.2f}',f'{v["speedup"]:.2f}'])
    return [[('h1','Appendix. R1 native learning execution across depth'),
        ('p','The compiled temporal scan accelerates complete native likelihood/backward/Adam steps while preserving losses, gradients, forward parameters and optimizer state with zero observed error. Temporal blocks, keyed memory, predecessor messages and silence-aware race likelihood remain unchanged.'),
        ('table',(['Depth','Events','Reference ms','Scan ms','Speedup'],rows,[23,25,44,40,34])),
        ('small','CPU float64, one thread, batch 1, random marked streams; four matched updates per arm/shape, first warmup, median of three timings with alternating execution order. Identical gradient-copy instrumentation is included. This is execution throughput, not benchmark quality, sparse-credit savings or hardware energy.'),
        ('p','The gain increases with depth in this bounded measurement: 2.47 to 5.95 times at 128 events and 2.89 to 7.79 at 512. All factual first-order derivatives are retained. Dense addressed-key work remains charged; higher-order meta-gradients require the reference backend.'),
        ('small',f'Whole diagnostic wall {r["wall_s"]:.2f} s; peak RSS {r["peak_rss_kb"]/1024:.1f} MiB. No FLOP reduction is claimed.')]]


def averaged_credit_pages(folder):
    prefix='curie_averaged_credit_v3_20261010T1720Z'
    files=[folder/(prefix+f'_pilot_s{s}.json') for s in (170,171,172)]
    if not all(p.exists() for p in files):return []
    records=[json.loads(p.read_text()) for p in files];assert all(r['status']=='completed' for r in records)
    rows=[];work=[]
    for seed,r in zip((170,171,172),records):
        for v in r['metrics']['arms']:
            rows.append([str(seed),v['history']+'/'+v['labels'],f'{v["mse_gain_vs_averaged_train_constant"]:.6g}',f'{v["utility_gain_vs_averaged_train_constant"]:.6g}'])
        targets=r['metrics']['teacher_target_presentations'];work.append([str(seed),str(targets),f'{r["supported_flops"]/1e9:.3f}',f'{r["supported_flops"]/targets/1e6:.4f}',f'{r["wall_s"]:.1f}'])
    return [[('h1','Appendix. R1 future credit: label averaging and prefix history'),
        ('p','Four crossed arms use identical prefix packets, continuation samples, fitting steps, initialization and a common TRAIN-derived target scale. Single-continuation and averaged-continuation labels are crossed with isolated and connected encoders. Positive gains indicate improvement against the common averaged TRAIN-mean fixed-action reference.'),
        ('table',(['Seed','Arm','MSE reduction','Future loss reduction'],rows,[17,64,43,43])),
        ('small',records[0]['metrics']['scope']),
        ('table',(['Seed','Teacher targets','Total GFLOPs','MFLOPs/target','Wall s'],work,[17,39,39,39,33])),
        ('small',records[0]['flop_scope']),
        ('p','The admission gate requires improved DEV calibration and action utility in all three seeded replicates before depth scaling. This frozen-actor synthetic intervention diagnostic does not demonstrate an online asynchronous learner or deployment access to a true continuation generator.')]]


def depth_credit_pages(folder):
    prefix='curie_averaged_credit_depth_v4_20261010T1750Z';records=[]
    for depth in (4,8):
        for seed in (170,171,172):
            p=folder/(prefix+f'_d{depth}_pilot_s{seed}.json')
            if not p.exists():return []
            r=json.loads(p.read_text());assert r['status']=='completed';records.append((depth,seed,r))
    rows=[];work=[]
    for depth,seed,r in records:
        for v in r['metrics']['arms']:
            if v['labels']=='averaged':rows.append([str(depth),str(seed),v['history'],f'{v["dev_relative_mse"]:.3f}',f'{v["utility_gain_vs_averaged_train_constant"]:.6g}'])
        targets=r['metrics']['teacher_target_presentations'];work.append([str(depth),str(seed),f'{r["supported_flops"]/1e9:.3f}',f'{r["supported_flops"]/targets/1e6:.4f}',f'{r["wall_s"]:.1f}'])
    return [[('h1','Appendix. R1 denoised intervention credit at depth'),
        ('p','The same crossed history/label construction is applied to deeper frozen native actors. Averaged-label arms are shown; each is compared with its own TRAIN-mean action. All single-label diagnostics remain in result files.'),
        ('table',(['Depth','Seed','History','DEV relative MSE','Future loss gain'],rows,[18,18,35,47,49])),
        ('small',records[0][2]['metrics']['depth_scope']),
        ('table',(['Depth','Seed','Total GFLOPs','MFLOPs/target','Wall s'],work,[18,18,43,45,43])),
        ('small','All four critic fits and shared teacher generation charged together. Supported arithmetic excludes special/unsupported operations. Different frozen actor depths have different target distributions; this is intervention-interface transfer, not a matched-quality actor-learning scaling result.')]]
