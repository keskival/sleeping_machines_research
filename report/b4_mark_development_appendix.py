"""Current source-bound B4 numerical and mark-memory development evidence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. B4 repaired learning and mark-memory development'


def verify_artifact(path,digest):
    """Verify present raw artifacts; remote renderers retain recorded binary digests.
    Completed JSON evidence and driver sources remain mandatory on every host.
    """
    artifact=ROOT/path
    if artifact.exists():assert hashlib.sha256(artifact.read_bytes()).hexdigest()==digest
    else:
        assert artifact.suffix=='.pt', f'Missing mandatory evidence: {path}'
        assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)


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
        verify_artifact(path,digest)
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
            verify_artifact(path,digest)
    for row in serving['rows']:
        verify_artifact(row['bundle'],row['bundle_sha256'])
    rows=[[str(r['seed']),f"{r['student']['L_M']:.3f}",f"{r['mark_gain']:.3f}",f"{100*r['retention']:.1f}%",f"{r['teacher_serving_s']/r['student_serving_s']:.2f}x"] for r in decision['rows']]
    return [('p',f"One-core conditional mark readout passes the unchanged gain>=1 and retention>=80% gate in all three seeds. Mean DEV gain {decision['mean_mark_gain']:.3f} nats per sequence, sample SD {decision['sample_sd']:.3f}; mean per-seed teacher-gain retention {100*decision['mean_retention']:.1f}%."),
            ('table',(['Seed','Student marks','Gain','Retention','Pair/student speed'],rows,[16,35,26,31,40])),
            ('p','TRAIN soft targets reuse the confirmed native pair. Linear ridge and linear KL pilots retained only60.1/60.3% and failed. A fixed64-wide Tanh residual on144 frozen native features fits50 passes,665,350 cached target presentations and1,300 updates; seed0 selected, seeds1/2 fixed before DEV scoring. Student32,938 versus pair50,826 parameters. One Github split1,34 DEV sequences/2,094 targets, CPU float64 full conditional-density interface. Teacher quality remains higher; this is a scoped quality/work tradeoff.'),
            ('p','Teacher-free serialized bundles reproduce all three scores exactly, normalization error <=4.45e-16, first-target suffix-mark error0, original time density unchanged. Each bundle290,035 bytes contains one native core and readout metadata; no teacher or TRAIN loaded for restoration. Prerequisite fits97–117 minutes per seed are charged, plus~2.53s teacher preparation,~1.18s linear fitting and~1.43s residual fitting. No TEST, deployment, full-fit efficiency or integrated sparse-learning claim. Leading joint-model DEV total and frozen benchmark verdicts stay current.'),
            ('p','Reports verify mandatory completed JSON/source evidence on every host and available raw binary hashes; checkpoint/bundle digests remain recorded when binaries reside only on the experiment host.'),
            ('p','Evidence: '+', '.join(tags)+', aws_nonlinear_mark_readout_20261011_s0/s1/s2. Next: sequence-level residual diagnosis before broader-domain development.')]

def observation_anchor_blocks():
    tag='aws_observation_readout_decision_20261011'
    if not (ROOT/'experiments/results/tpp_b4'/f'{tag}.json').exists():return []
    decision=bound(tag)
    for path,digest in decision['input_sha256'].items():
        verify_artifact(path,digest)
    for seed in range(3):bound(f'aws_observation_anchored_readout_20261011_s{seed}')
    rows=[[str(r['seed']),f"{r['student']['L_M']:.3f}",f"{r['gain_over_prior_student']:.3f}",f"{r['baseline_gain']:.3f}",f"{100*r['teacher_retention']:.1f}%"] for r in decision['rows']]
    blocks=[('p',f"Observation-anchored one-core readout {'passes' if decision['gate'] else 'fails'} its preregistered gate: mean DEV improvement over the previous compact student {decision['mean_gain_over_prior_student']:.3f} nats per sequence, sample SD {decision['sample_sd']:.3f}. All three seeds improve. TRAIN target90% teacher probability plus10% actual observed-mark one-hot; coefficient fixed before new DEV scores, no DEV selection or early stopping."),
            ('table',(['Seed','Anchored marks','Gain over student','Baseline gain','Teacher-gain ratio'],rows,[16,35,36,29,32])),
            ('p','Same one-core32,938-parameter readout,144 frozen features/64-wide residual/50 passes. Observed target marks enter TRAIN loss only; normalized serving probabilities use causal history and candidate time. Two seeds exceed teacher DEV mark quality. Fitting2.60–2.62s after2.58–2.61s teacher preparation per seed, plus original97–117-minute core fits and prior readout development. One Github split1 DEV; no TEST, independent-dataset, future-intervention or sparse-learning claim. Leading selected joint-model seed0 DEV total is unchanged.'),
            ('p','Registered adoption gate: mean gain>=0.1 over saved student, each seed gain>=-0.25 and original baseline gain>=1/teacher-retention>=80% gates all pass. Evidence: '+tag+'.')]
    serving_path=ROOT/'experiments/results/tpp_b4/aws_observation_readout_serving_20261011.json'
    if serving_path.exists():
        serving=bound('aws_observation_readout_serving_20261011')
        for path,digest in serving['input_sha256'].items():verify_artifact(path,digest)
        for r in serving['rows']:verify_artifact(r['bundle'],r['bundle_sha256'])
        blocks.append(('p','Anchored teacher-free bundle restoration passes all three seeds: score/time parity0, normalized mark probabilities<=4.45e-16 error and first-target suffix-mark error0. Bundle source/checkpoint hashes are verified; no teacher/TRAIN loaded to restore.'))
    return blocks

def readout_curve_blocks():
    tag='aws_readout_curve_decision_20261011'
    if not (ROOT/'experiments/results/tpp_b4'/f'{tag}.json').exists():return []
    decision=bound(tag)
    for path,digest in decision['input_sha256'].items():
        verify_artifact(path,digest)
        record=json.loads((ROOT/path).read_text())
        for source,sha in record['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
    rows=[]
    for r in decision['rows']:
        t=r['arms']['teacher']['first_scheduled_point'];a=r['arms']['anchor']['first_scheduled_point']
        rows.append([str(r['seed']),f"{r['target']:.3f}",str(t['passes']) if t else 'unreached',str(a['passes']) if a else 'unreached',f"{r['incremental_work_ratio']:.3f}",f"{r['complete_supported_training_work_ratio']:.6f}"])
    return [('p','Fixed-target readout learning curves confirm earlier scheduled attainment in two seeds and equal attainment in the third. Targets come from completed teacher-only compact students before new runs; snapshots0/5/10/20/35/50, no early stopping or DEV selection. Both final arms reproduce their saved50-pass scores within1e-9.'),
            ('table',(['Seed','Target marks','Teacher passes','Anchor passes','Incremental fit ratio','Inherited-fit ratio'],rows,[14,28,30,29,34,39])),
            ('p','Ratios are anchor/teacher training wall at the first scheduled target. Incremental fitting includes teacher preparation, linear fitting and residual updates; inherited ratio also charges original core training and both earlier readout-development rounds. Those large prerequisites make the complete supported training ratio near1: this is an incremental readout-learning improvement, not whole-pipeline/depth/sparse-credit supremacy. DEV evaluation is separately charged in the curve records; inherited core evaluation, failed initial pilots and other diagnostic overhead are additional costs, not zero.'),
            ('p',f"All six curve diagnostics charge {decision['whole_six_fit_diagnostic_wall_s']:.2f}s wall; mean final mark gain remains {decision['mean_final_gain']:.3f} nats/sequence. Same frozen native core, identical TRAIN order/initialization/optimizer, three seeds on one Github DEV split; no TEST. Driver legacy human-readable rule string says90/10 for both arms; actual observed_train_weight fields and executed source are0/0.1, verified by saved-score reproduction. Evidence: "+tag+'.')]

def split2_development_blocks():
    tag='aws_b4_split2_logspread_decision_20261011_v2'
    if not (ROOT/'experiments/results/tpp_b4'/f'{tag}.json').exists():return []
    pilot=bound(tag)
    for path,digest in pilot['input_sha256'].items():
        verify_artifact(path,digest)
        record=json.loads((ROOT/path).read_text())
        for source,sha in record['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
    rows=[[str(r['epoch']+1),f"{r['guarded_total']:.3f}",f"{r['repair_total']:.3f}",f"{r['gain']:.3f}"] for r in pilot['paired_epoch_comparison']]
    blocks=[('p','Prospective Github split2 numerical-repair pilot passes: five fresh TRAIN passes with zero skips,223.35s including DEV. The first four epoch scores match the saved guarded run within3.3e-13; at the fifth pass repaired total DEV loss is5.346 nats per sequence lower. Original model/settings and data-derived clocks retained; full-backward contract passed.'),
            ('table',(['Pass','Guarded total','Repaired total','DEV gain'],rows,[17,44,44,43])),
            ('p','One seed0/new partition of the same Github cohort; not an independent dataset or additional unique-data result. Original2GB full-backward gate stopped at2.36GB RSS before pilot fitting; measured demand justified an explicit4GB cap on~29GB available host memory with8GiB reserve. Gate passed under the amended cap. Frozen TEST scores remain unchanged.'),
            ('p','The completed pilot admits one fresh100-pass/30-pass-patience fit at the original lr0.003/batch32. Pilot restart work is charged. Require zero skips/finite completed fit and selected DEV gain>=0.5 over saved guarded split2 before further mark fitting. Pending scores are not evidence. Source-bound evidence: '+tag+'.')]
    full_tag='aws_b4_split2_development_decision_20261011'
    if (ROOT/'experiments/results/tpp_b4'/f'{full_tag}.json').exists():
        blocks[3]=('p','Completed pilot justified the fresh100-pass/30-pass-patience fit at original lr0.003/batch32; duplicated pilot work is charged. Registered zero-skip/finite completion and selected DEV gain>=0.5 gates are evaluated below. Frozen TEST unchanged.')
        full=bound(full_tag)
        for path,digest in full['input_sha256'].items():verify_artifact(path,digest)
        if full['repaired_fit_completed']:
            blocks.append(('p',f"Full split2 repaired fit completes {full['epochs']} passes with zero skips: selected DEV total {full['selected_repaired_dev']['total']:.3f}, gain {full['selected_total_gain']:.3f} versus guarded. Registered quality gate {'passes' if full['quality_gate'] else 'fails'}. Fitting {full['fitting_wall_s']:.2f}s, DEV evaluation {full['evaluation_wall_s']:.2f}s, paid pilot {full['paid_pilot_wall_s']:.2f}s; peak fit RSS {full['fit_max_rss_kb']/1024:.1f}MiB. One seed/split development, no new TEST."))
        else:blocks.append(('p','Full split2 repaired fit fails its finite/no-skip completion gate; current compact readout and benchmark scores retained. Evidence: '+full_tag+'.'))
    return blocks

def split2_marks_blocks():
    tag='aws_b4_split2_marks_decision_20261011';path=ROOT/'experiments/results/tpp_b4'/f'{tag}.json'
    if not path.exists():return []
    d=bound(tag)
    for path,digest in d['input_sha256'].items():
        verify_artifact(path,digest);record=json.loads((ROOT/path).read_text())
        for source,sha in record['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
    if not d['fits_completed_without_skips']:return [('p','Split2 mark fit fails finite/no-skip completion; repaired core and current compact readout retained. No family-loss or benchmark verdict. Evidence: '+tag+'.')]
    rows=[[label,f"{r['L_T']:.3f}",f"{r['L_M']:.3f}",f"{r['total']:.3f}"] for label,r in [('Repaired core',d['selected_repaired']),('Mark memory',d['selected_marks'])]]
    work=[[r['arm'],str(r['epochs']),f"{r['targets']:,}",f"{r['fit_s']:.1f}",f"{r['fit_s']/r['targets']*1000:.3f}",f"{r['inference']['seconds_per_target']*1e6:.2f}"] for r in d['accounting']]
    return [('p',f"Prospective split2 mark-memory gate {'passes' if d['gate'] else 'fails'}: last20 common-epoch mean mark gain {d['last20_common_mark_gain']:.3f}, required1.000. Selected mark gain {d['selected_mark_gain']:.3f}, selected total gain {d['selected_total_gain']:.3f}. Selected-checkpoint improvement does not override the stability gate."),
            ('table',(['Model','Time loss','Mark loss','Total loss'],rows,[49,33,33,33])),
            ('table',(['Arm','Passes','TRAIN targets','Fit seconds','Fit ms/target','DEV us/target'],work,[36,17,32,26,30,31])),
            ('p',f"Same Github cohort/split2/seed0, original100pass/30patience/lr0.003/batch32, no TEST. Core21,768/mark29,057 parameters; auxiliary temporal memory updates densely, all clocks scored. Shared pilot {d['paid_pilot_wall_s']:.2f}s and DEV evaluation remain additional. Full fitting FLOPs unmeasured; no sparse-credit, matched-quality efficiency, independent-dataset or benchmark claim. Both complete predictors remain evidence for a separately registered TRAIN-only calibration/readout decision. Evidence: "+tag+'.')]

def split2_readout_blocks():
    tags=['aws_b4_split2_mark_confidence_20261011','aws_b4_split2_observation_readout_20261011']
    if not all((ROOT/'experiments/results/tpp_b4'/f'{t}.json').exists() for t in tags):return []
    teacher,student=[bound(t) for t in tags]
    for record in (teacher,student):
        for path,digest in record['input_sha256'].items():verify_artifact(path,digest)
    rows=[[label,f"{r['L_T']:.3f}",f"{r['L_M']:.3f}",f"{r['total']:.3f}"] for label,r in [('Repaired core',student['baseline_dev']),('Calibrated pair',student['teacher_dev']),('Anchored one-core',student['student_dev'])]]
    blocks=[('p',f"Split2 TRAIN-only global confidence alpha {teacher['alpha_train_locked']:.6f} improves DEV marks {teacher['dev_mark_gain']:.3f} over repaired core/time fixed. Fixed split1 compact recipe {'passes' if student['gate'] else 'fails'} unchanged mark gain>=1 and teacher-retention>=80% gate on this new partition."),
            ('table',(['Model','Time loss','Mark loss','Total loss'],rows,[49,33,33,33])),
            ('p',f"Compact readout gain {student['mark_gain']:.3f} nats per sequence versus repaired core; gain over calibrated pair {student['teacher_dev']['L_M']-student['student_dev']['L_M']:.3f}, teacher-gain ratio {100*student['retained_teacher_gain']:.1f}%. Student {student['parameters']:,} versus pair {student['teacher_parameters']:,} parameters. Warm one-core/pair-predictor forward {student['warm_single_dev_s']:.3f}/{student['warm_pair_dev_s']:.3f}s ({student['warm_pair_dev_s']/student['warm_single_dev_s']:.2f}x) on2,778 DEV targets/34 sequences; final scalar-mixture reduction excluded from this conservative pair-forward measurement."),
            ('p',f"One seed0/new partition of same Github cohort, fixed144 features/64-wide residual/50 passes and90% teacher/10% observed TRAIN targets. No DEV tuning, new TEST or independent-dataset/general training claim. Teacher fits {student['prerequisite_fitting_s']:.2f}s, teacher preparation {student['teacher_preparation_s']:.3f}s, linear fit {student['readout_fit_s']:.3f}s and residual fit {student['nonlinear_fit_s']:.3f}s; pilot/validation/failed investigations are additional. Native core/time fixed, outputs normalized; observed labels enter TRAIN loss only. Full fitting FLOPs unmeasured. Evidence: "+', '.join(tags)+'.')]
    serving_path=ROOT/'experiments/results/tpp_b4/aws_b4_split2_observation_serving_20261011.json'
    if serving_path.exists():
        serving=bound('aws_b4_split2_observation_serving_20261011')
        for path,digest in serving['input_sha256'].items():verify_artifact(path,digest)
        for row in serving['rows']:verify_artifact(row['bundle'],row['bundle_sha256'])
        blocks.append(('p','Teacher-free split2 portable bundle reproduces score/time exactly, normalized mark probabilities and first-target suffix invariance. One core/readout metadata, no teacher or TRAIN loaded for restoration.'))
    tradeoff_path=ROOT/'experiments/results/tpp_b4/aws_b4_split2_serving_tradeoff_20261011.json'
    if tradeoff_path.exists():
        tradeoff=bound('aws_b4_split2_serving_tradeoff_20261011')
        for path,digest in tradeoff['input_sha256'].items():verify_artifact(path,digest)
        blocks.append(('p',f"Full normalized mixture/scoring boundary: student {tradeoff['student_median_s']:.3f}s vs pair {tradeoff['pair_median_s']:.3f}s, {tradeoff['speedup']:.2f}x faster; five warmed CPU timings with alternating order. Includes scalar-mixture reduction and reproduces both saved scores within1e-9. Student35.2% fewer parameters and0.158 lower DEV mark loss than this calibrated pair, with identical time density. Development quality/serving point against our own teacher, not Transformer or total-training superiority."))
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
        ('p',f"Scalar fitting {confidence['scalar_fit_s']*1000:.2f}ms after {confidence['predictor_forward_s']:.3f}s of predictor evaluation. Both native models remain necessary: {confidence['parameters']:,} parameters and {confidence['prerequisite_fitting_s']:.2f}s of prerequisite fits are charged. This is a saved-predictor calibration diagnostic, not an integrated sparse-substrate or efficiency win. Neither failed mark-fit gate is relaxed. The completed confirmation and single-readout results follow.")] + confirmation_blocks() + single_readout_blocks() + observation_anchor_blocks() + readout_curve_blocks() + split2_development_blocks() + split2_marks_blocks() + split2_readout_blocks()]
