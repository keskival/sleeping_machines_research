"""Source-bound AWS block-credit execution and full-fit quality evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = 'Appendix. B1 exact deep online credit: structural blocks and real-data execution'
PREFIX = 'experiments/results/credit/aws_deep_blocks_real_20261010T180155Z_smoke.json'
DIAGNOSIS = 'experiments/results/credit/aws_deep_blocks_diagnosis_20261010T181802Z.json'
DECISION = 'experiments/results/credit/aws_deep_blocks_full_20261010T181802Z_decision.json'


def load_bound(path):
    result = json.loads(path.read_text())
    if result['status'] != 'completed': raise ValueError('Incomplete block-credit evidence')
    for source, digest in result['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest() != digest:
            raise ValueError('Block-credit source changed: '+source)
    return result


def pages():
    path = ROOT/PREFIX
    if not path.exists(): return []
    r = load_bound(path); targets = r['train_scored_events']; rows = []
    figure(r, path)
    for label in ('dense', 'blocks'):
        wall = r['training_wall_s'][label]
        rows.append([label, f'{wall:.2f}', f'{wall*1e6/targets:.1f}',
                     f"{r['trace_floats_per_stream'][label]*8/2**20:.3f}"])
    diagnostic = ''
    if (ROOT/DIAGNOSIS).exists():
        d = load_bound(ROOT/DIAGNOSIS)
        diagnostic = (f"Same-incoming-state audit at batch {d['audit']['batch']} gives gradient error "
                      f"{d['audit']['same_incoming_state_errors']['gradients']:.2g}, parameter error "
                      f"{d['audit']['same_incoming_state_errors']['parameters']:.2g}, zero mean-likelihood difference. ")
    return [[('h1', TITLE),
        ('p', f"Removing structural-zero sensitivity blocks accelerates four complete Taxi TRAIN batches "
         f"by {r['training_speedup']:.2f}× and reduces stored traces by 44.5%. The same two-layer forward model, "
         'initialized weights, shuffled batches and per-event Adam recipe are used in both arms.'),
        ('table', (['Backend', 'TRAIN seconds', 'µs/scored TRAIN target', 'Trace MiB/stream'], rows, [32, 39, 56, 45])),
        ('figure', ('aws_deep_credit_blocks', 170)),
        ('p', 'Layer-1 state cannot depend on downstream parameters; its trace omits those columns. Input derivatives '
         'are scattered into their embedding/gap support. Layer-2 own-weight traces retain one vector per mode; '
         'the upstream cross-layer trace stays fully dense. Every retained sensitivity undergoes the same temporal '
         'decay and rotation. The fixed-weight every-parameter BPTT contract passes with 9.56e-16 relative error.'),
        ('small', f"One CPU thread, float64, seed 0; {targets:,} scored TRAIN events, "
         f"{r['optimizer_updates_per_arm']} updates per arm. Whole train steps include forward, exact credit transport, "
         'local backward and Adam. Trace storage is a tensor-element count, not total RSS. Larger-width synthetic '
         'batches give 3.05× speed, at depth two. These are execution measurements on a dense-write diagnostic member; '
         'no hard-routing, arbitrary-depth, whole-fit FLOP, energy or new benchmark-quality claim.'),
        ('p', diagnostic+'Independent floating-point trajectories later crossed both absolute-gradient and '
         'per-target-likelihood identity guards. Both stopped logs are preserved; no completed paired epoch is claimed. '
         'A three-seed 20-pass final-DEV comparison uses the original model and recipe. Its predeclared adoption rule '
         'allows at most 0.002 mean quality loss and 0.005 on any seed. TEST remains untouched.')]] + full_fit_pages()


def full_fit_pages():
    if not (ROOT/DECISION).exists(): return []
    decision = load_bound(ROOT/DECISION)
    assert decision['stage'] == 'quality_decision'
    rows = []
    for name, digest in decision['inputs'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest: raise ValueError(name)
        r = load_bound(ROOT/name)
        seed = r.get('seed', r.get('args', {}).get('seed'))
        label = 'blocks' if r.get('stage') == 'full_fit' else 'saved dense'
        rows.append([str(seed), label, f"{r['final_dev_ll']:.6f}", f"{r['wall_s']/60:.2f}"])
    verdict = 'passes' if decision['quality_preservation_gate_pass'] else 'fails'
    return [[('h1', 'Appendix. B1 block-credit adoption: three-seed full-fit quality'),
        ('p', f"The fixed-recipe 20-pass quality preservation gate {verdict}: mean final DEV "
         f"{decision['mean_final_dev_ll']:.6f} versus saved dense traces "
         f"{decision['reference_mean_final_dev_ll']:.6f}. {decision['decision']}."),
        ('table', (['Seed', 'Backend', 'Final DEV nats/event', 'Whole-fit minutes'], sorted(rows), [20, 40, 58, 54])),
        ('small', decision['scope']+' Full-fit wall includes loading, evaluation and checkpoint work. '
         'The saved walls are historical; controlled speed comes from the paired prefix above. '
         'All parameter/Adam/RNG final checkpoints are published. Inference, FLOPs and energy are unchanged or unmeasured; '
         'this adoption gate is not a leaderboard result.')]]


def figure(r, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    data = [([r['training_wall_s'][x] for x in ('dense', 'blocks')], 'Four complete TRAIN batches', 'seconds'),
            ([r['trace_floats_per_stream'][x]*8/2**20 for x in ('dense', 'blocks')], 'Persistent trace state', 'MiB per stream')]
    for ax, (values, title, unit) in zip(axes, data):
        ax.bar(['Dense traces', 'Structural blocks'], values, color=['#8a8984', '#2a78d6'], width=.6)
        for i, value in enumerate(values): ax.text(i, value, f'{value:.2f}', ha='center', va='bottom')
        ax.set_title(title); ax.set_ylabel(unit); ax.set_ylim(0, max(values)*1.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"Paired two-layer Taxi execution · {r['training_speedup']:.2f}× faster TRAIN", fontsize=12)
    fig.tight_layout()
    folder = ROOT/'report/figures'
    for ext in ('png', 'svg'): fig.savefig(folder/f'aws_deep_credit_blocks.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)
    (folder/'aws_deep_credit_blocks.inputs.json').write_text(json.dumps(dict(
        result=PREFIX, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scope=r['scope']), indent=2)+'\n')
