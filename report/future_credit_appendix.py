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
         'Coupled forward learning, regime adaptation and depth/work scaling follow a useful replicated signal.')]] + diagnostic_pages(folder)


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
