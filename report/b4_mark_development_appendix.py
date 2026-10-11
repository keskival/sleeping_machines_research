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
        ('p',f"Scalar fitting {confidence['scalar_fit_s']*1000:.2f}ms after {confidence['predictor_forward_s']:.3f}s of predictor evaluation. Both native models remain necessary: {confidence['parameters']:,} parameters and {confidence['prerequisite_fitting_s']:.2f}s of prerequisite fits are charged. This is a saved-predictor calibration diagnostic, not an integrated sparse-substrate or efficiency win. Neither failed mark-fit gate is relaxed. Next: independent confirmation of calibration before engineering a single integrated readout.")]]
