"""Plot completed paired DEV curves and payload utility, without work estimates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = ROOT / 'experiments/results/token_language'
    stem = 'aws_private_bank_time_256k_20261006T041606Z'
    payload_path = ROOT / 'experiments/results/diagnostics/aws_private_bank_time_256k_audit_20261006T041757Z_payload.json'
    payload = json.loads(payload_path.read_text())
    assert payload['status'] == 'completed'
    utility = {r['tag']: r for r in payload['rows']}
    sources = [payload_path]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    colors = ('#225ea8', '#d95f0e')
    bars = [[], []]
    for index, seed in enumerate((6, 7)):
        fits = []
        for arm, mode in enumerate(('scale1', 'scale1over16')):
            tag = stem + '_' + mode + '_s' + str(seed)
            path = folder / (tag + '.aws.json')
            fit = json.loads(path.read_text())
            assert fit['status'] == 'completed'
            assert fit['args']['train_tokens'] == 262144 and fit['presentations_total'] == 524272
            row = utility[tag]
            assert row['partition_parity'] and row['matched_rng'] and all(row['intervention_contracts'].values())
            assert row['memory_time_scale'] == fit['memory_time_scale']
            assert row['selected']['dev_nll'] == fit['best_dev_nll']
            curve = fit['curve']
            selected = min(curve, key=lambda r: (r['dev_nll'], r['step']))
            assert selected['dev_nll'] == fit['best_dev_nll']
            label = 'Scale 1' if arm == 0 else 'Scale 1/16'
            axes[index].plot([r['step'] for r in curve], [r['dev_nll'] for r in curve],
                             color=colors[arm], label=label, linewidth=1.8)
            axes[index].scatter([selected['step']], [selected['dev_nll']], color=colors[arm], s=34, zorder=3)
            bars[arm].append(row['erasure_increase']['payload'])
            fits.append(fit); sources.append(path)
        assert fits[0]['identity'] == fits[1]['identity']
        gain = fits[0]['best_dev_nll'] - fits[1]['best_dev_nll']
        axes[index].set_title(f"Seed {seed}: scale 1/16 {'win' if gain > 0 else 'loss'} {abs(gain):.5f} NLL")
        axes[index].set_xlabel('Optimizer update')
        axes[index].set_ylabel('Development NLL (lower is better)')
        axes[index].grid(alpha=.2)
        axes[index].legend(frameon=False)
    for arm, offset in enumerate((-.18, .18)):
        positions = [x + offset for x in (0, 1)]
        axes[2].bar(positions, bars[arm], width=.36, color=colors[arm], label=('Scale 1', 'Scale 1/16')[arm])
    axes[2].set_xticks([0, 1], ['Seed 6', 'Seed 7'])
    axes[2].set_ylabel('NLL increase after payload erasure')
    axes[2].set_title('Useful stored content in both seeds')
    axes[2].legend(frameon=False)
    axes[2].grid(axis='y', alpha=.2)
    fig.suptitle('Positive temporal evolution: split quality, stronger payload dependence', fontsize=13)
    fig.text(.5, .01, '262,144 GPT-2 TRAIN tokens; two passes / 524,272 fitting targets; 2,040 DEV targets.\n'
             'Two exploratory paired seeds; frozen erasure retains timing/message state. Complete work audits pending.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .93))
    target = ROOT / 'report/figures/aws_private_bank_time_256k_20261006'
    for suffix in ('.svg', '.png', '.inputs.json'):
        if target.with_suffix(suffix).exists(): raise FileExistsError(target.with_suffix(suffix))
    fig.savefig(target.with_suffix('.svg'))
    fig.savefig(target.with_suffix('.png'), dpi=180)
    target.with_suffix('.inputs.json').write_text(json.dumps(dict(
        status='completed', input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        producer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
    plt.close(fig)


if __name__ == '__main__': main()
