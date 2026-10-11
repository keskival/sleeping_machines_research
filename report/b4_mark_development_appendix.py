"""Current source-bound B4 numerical and mark-memory development evidence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. B4 repaired learning and mark-memory development'


def bound(tag):
    p=ROOT/'experiments/results/tpp_b4'/f'{tag}.json';r=json.loads(p.read_text());assert r['status']=='completed'
    for s,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
    return r



def confirmation_blocks():
    tag='aws_b4_calibration_confirmation_20261011'
    tags=[tag+'_decision']+[tag+f'_serving_s{s}' for s in range(3)]
    if not all((ROOT/'experiments/results/tpp_b4'/f'{t}.json').exists() for t in tags):return []
    decision=bound(tags[0]);serving=[bound(t) for t in tags[1:]]
    for path,digest in decision['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    rows=[[str(r['seed']),f"{r['baseline']['L_M']:.3f}",f"{r['calibrated']['L_M']:.3f}",f"{r['gain']:.3f}",f"{r['prerequisite_fitting_s']/60:.1f}"] for r in decision['rows']]
    verdict='passes' if decision['gate'] else 'fails'
    return [('p',f"Fixed-recipe three-seed calibration confirmation {verdict}: mean DEV mark gain {decision['mean_mark_gain']:.3f} nats per sequence, sample SD {decision['sample_sd']:.3f}. Registered gate: mean gain >=1.000 and every seed positive; TRAIN-only global coefficients, repaired timing unchanged."),
            ('table',(['Seed','Baseline marks','Calibrated marks','Gain','Prerequisite fit min'],rows,[16,35,35,25,37])),
            ('p','Warm sequential DEV serving through both complete predictors: '+', '.join(f"seed{r['seed']} {r['dev_seconds_per_target']*1e6:.2f} microseconds/target; helps {sum(g>0 for g in r['sequence_mark_gains'])}/34 sequences" for r in serving)+'. Query, scalar-fit and prerequisite work are separate; both predictors remain necessary.'),
            ('p','Scope: one Github split1; seed0 selected the diagnostic and seeds1/2 confirm its fixed recipe on the same DEV split. No new TEST scoring, independent-dataset generalization, efficiency or integrated sparse-learning win. Failed joint/residual gates remain failed. Serving and per-sequence errors determine the next single-readout engineering decision.')]

def single_readout_blocks():
    tags=['aws_single_readout_decision_20261011','aws_single_readout_serving_20261011_v1']
    if not all((ROOT/'experiments/results/tpp_b4'/f'{t}.json').exists() for t in tags):return []
    decision,serving=[bound(t) for t in tags]
    for r in (decision,serving):
        for path,digest in r['input_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    for row in serving['rows']:
        assert hashlib.sha256((ROOT/row['bundle']).read_bytes()).hexdigest()==row['bundle_sha256']
    rows=[[str(r['seed']),f"{r['student']['L_M']:.3f}",f"{r['mark_gain']:.3f}",f"{100*r['retention']:.1f}%",f"{r['teacher_serving_s']/r['student_serving_s']:.2f}x"] for r in decision['rows']]
    return [('p',f"One-core conditional mark readout passes the unchanged gain>=1 and retention>=80% gate in all three seeds. Mean DEV gain {decision['mean_mark_gain']:.3f} nats per sequence, sample SD {decision['sample_sd']:.3f}; mean per-seed teacher-gain retention {100*decision['mean_retention']:.1f}%."),
            ('table',(['Seed','Student marks','Gain','Retention','Pair/student speed'],rows,[16,35,26,31,40])),
            ('p','TRAIN soft targets reuse the confirmed native pair. Linear ridge and linear KL pilots retained only60.1/60.3% and failed. A fixed64-wide Tanh residual on144 frozen native features fits50 passes,665,350 cached target presentations and1,300 updates; seed0 selected, seeds1/2 fixed before DEV scoring. Student32,938 versus pair50,826 parameters. One Github split1,34 DEV sequences/2,094 targets, CPU float64 full conditional-density interface. Teacher quality remains higher; this is a scoped quality/work tradeoff.'),
            ('p','Teacher-free serialized bundles reproduce all three scores exactly, normalization error <=4.45e-16, first-target suffix-mark error0, original time density unchanged. Each bundle290,035 bytes contains one native core and readout metadata; no teacher or TRAIN loaded for restoration. Prerequisite fits97–117 minutes per seed are charged, plus~2.53s teacher preparation,~1.18s linear fitting and~1.43s residual fitting. No TEST, deployment, full-fit efficiency or integrated sparse-learning claim. Leading joint-model DEV total and frozen benchmark verdicts stay current.'),
            ('p','Evidence: '+', '.join(tags)+', aws_nonlinear_mark_readout_20261011_s0/s1/s2. Next: sequence-level residual diagnosis before broader-domain development.')]

def observation_anchor_blocks():
    tag='aws_observation_readout_decision_20261011'
    if not (ROOT/'experiments/results/tpp_b4'/f'{tag}.json').exists():return []
    decision=bound(tag)
    for path,digest in decision['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    for seed in range(3):bound(f'aws_observation_anchored_readout_20261011_s{seed}')
    rows=[[str(r['seed']),f"{r['student']['L_M']:.3f}",f"{r['gain_over_prior_student']:.3f}",f"{r['baseline_gain']:.3f}",f"{100*r['teacher_retention']:.1f}%"] for r in decision['rows']]
    blocks=[('p',f"Observation-anchored one-core readout {'passes' if decision['gate'] else 'fails'} its preregistered gate: mean DEV improvement over the previous compact student {decision['mean_gain_over_prior_student']:.3f} nats per sequence, sample SD {decision['sample_sd']:.3f}. All three seeds improve. TRAIN target90% teacher probability plus10% actual observed-mark one-hot; coefficient fixed before new DEV scores, no DEV selection or early stopping."),
            ('table',(['Seed','Anchored marks','Gain over student','Baseline gain','Teacher-gain ratio'],rows,[16,35,36,29,32])),
            ('p','Same one-core32,938-parameter readout,144 frozen features/64-wide residual/50 passes. Observed target marks enter TRAIN loss only; normalized serving probabilities use causal history and candidate time. Two seeds exceed teacher DEV mark quality. Fitting2.60–2.62s after2.58–2.61s teacher preparation per seed, plus original97–117-minute core fits and prior readout development. One Github split1 DEV; no TEST, independent-dataset, future-intervention or sparse-learning claim. Leading selected joint-model seed0 DEV total is unchanged.'),
            ('p','Registered adoption gate: mean gain>=0.1 over saved student, each seed gain>=-0.25 and original baseline gain>=1/teacher-retention>=80% gates all pass. Evidence: '+tag+'.')]
    serving_path=ROOT/'experiments/results/tpp_b4/aws_observation_readout_serving_20261011.json'
    if serving_path.exists():
        serving=bound('aws_observation_readout_serving_20261011')
        for path,digest in serving['input_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        for r in serving['rows']:assert hashlib.sha256((ROOT/r['bundle']).read_bytes()).hexdigest()==r['bundle_sha256']
        blocks.append(('p','Anchored teacher-free bundle restoration passes all three seeds: score/time parity0, normalized mark probabilities<=4.45e-16 error and first-target suffix-mark error0. Bundle source/checkpoint hashes are verified; no teacher/TRAIN loaded to restore.'))
    return blocks

def pages():
    path=ROOT/'experiments/results/tpp_b4/aws_b4_marks_diagnosis_20261011T0303Z.json'
    if not path.exists():return []
    repair=bound('aws_b4_logspread_develop_decision_20261010T2113Z')
    mark=bound('aws_b4_marks_decision_20261010T2301Z');diag=bound('aws_b4_marks_diagnosis_20261011T0303Z')
    frozen=bound('aws_b4_frozen_marks_20261011T0307Z')
    transfer=bound('aws_b4_mark_transfer_20261011T0442Z')
    private=bound('aws_b4_private_inputs_20261011T0615Z')
    confidence=bound('aws_b4_mark_confidence_20261011T0619Z')
    b=mark['selected_repaired_baseline'];m=mark['selected_mark_memory']
    rows=[[label,f"{x['L_T']:.3f}",f"{x['L_M']:.3f}",f"{x['total']:.3f}"] for label,x in [('Repaired baseline',b),('Plus mark memory',m)]]
    work=[]
    for row in mark['accounting']:
        work.append([row['arm'],str(row['epochs']),f"{row['train_targets']:,}",f"{row['fitting_wall_s']/60:.2f}",f"{row['seconds_per_train_target']*1000:.3f}"])
    return [[('h1',TITLE),
        ('p',f"The log-domain clock repair completes {repair['repaired_epochs']} TRAIN passes with zero skipped updates, versus {repair['guarded_skips']} skips in the saved guarded run. "
             f"Selected DEV total loss improves by {repair['repaired_dev_loss_reduction']:.3f} nats per sequence. The original guarded run skipped a padded batch containing most TRAIN events."),
        ('table',(['Model','Time loss','Mark loss','Total loss'],rows,[52,32,32,32])),
        ('p',f"The mark-memory variant improves selected mark loss by {mark['selected_mark_loss_reduction']:.3f} and total loss by {mark['selected_total_loss_reduction']:.3f}. "
             f"Its registered stability gate fails: the last 20 common epochs improve marks by {mark['common_window_mark_loss_reduction']:.3f}, below the required 1.000; total gain is {mark['common_window_total_loss_reduction']:.3f}. No adoption or family-loss verdict follows."),
        ('p',f"At the selected mark checkpoint, disabling only its mark read raises mark loss from {diag['on_mean_mark_loss']:.3f} to {diag['off_mean_mark_loss']:.3f}. "
             f"The direct read helps {diag['sequences_helped']} of 34 sequences and gains {diag['direct_read_mark_gain']:.3f} nats per sequence, with timing unchanged. "
             'The residual cross-fit difference includes shared learning and RNG changes; it is not a separately identified causal effect.'),
        ('table',(['Arm','Passes','Scored TRAIN targets','Fit minutes','ms / target'],work,[43,18,37,26,24])),
        ('p','Parameters: 21,768 baseline, 29,057 mark variant. Both score every clock; the auxiliary temporal memory updates densely. '
             'Whole fitting FLOPs are unmeasured. Saved DEV likelihood-forward latency is 73.03 versus 86.41 microseconds per target; no matched-compute or energy claim.'),
        ('p','Scope: one Github split1 seed0 TRAIN/DEV development comparison, original core recipe and validation patience. '
             'Frozen B4 TEST verdicts remain unchanged.'),
        ('p',f"Frozen-core residual fitting completes {frozen['epochs']} passes, with zero skips and unchanged timing, but fails its development gate: selected mark gain {frozen['selected']['mark_gain']:.3f}; last-20 mean {frozen['mean_last20_mark_gain']:.3f}. Residual fitting {frozen['residual_fit_s']:.2f}s; cumulative fitting including the repaired warm start {frozen['cumulative_fit_s']:.2f}s."),
        ('p','Checkpoint swaps show that the joint branch transfers poorly to the repaired core. Preserving its original mark-input embedding/gap weights changes transferred DEV mark loss only 71.615 to 71.592, versus repaired baseline 67.237. The input-representation hypothesis is not supported by this diagnostic. Late residual training improves TRAIN loss while worsening DEV loss.'),
        ('p',f"A TRAIN-only scalar confidence fit combines the two complete native conditional-mark predictors with coefficient {confidence['alpha_train_locked']:.6f}, retaining the repaired timing density. DEV mark loss {confidence['dev']['L_M']:.3f} versus baseline {confidence['baseline_dev']['L_M']:.3f}, gain {confidence['dev_mark_gain']:.3f}; total loss {confidence['dev']['total']:.3f}. One coefficient is used for every candidate mark; there is no target-dependent inference selection."),
        ('p',f"Scalar fitting {confidence['scalar_fit_s']*1000:.2f}ms after {confidence['predictor_forward_s']:.3f}s of predictor evaluation. Both native models remain necessary: {confidence['parameters']:,} parameters and {confidence['prerequisite_fitting_s']:.2f}s of prerequisite fits are charged. This is a saved-predictor calibration diagnostic, not an integrated sparse-substrate or efficiency win. Neither failed mark-fit gate is relaxed. The completed confirmation and single-readout results follow.")] + confirmation_blocks() + single_readout_blocks() + observation_anchor_blocks()]
