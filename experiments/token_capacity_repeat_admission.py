"""Read-only completed-evidence gates, then admit one independent-seed queue."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def admit():
    candidates = []
    protocol = None
    for width in (16, 24, 32):
        tag = f'curie_data_growth_tokens_64k_b64_c16_p{width}_s6_20261005_v1'
        folder = ROOT / 'experiments/results/token_language'
        result = json.loads((folder / (tag + '.json')).read_text())
        selected = json.loads((folder / (tag + '.selection.json')).read_text())
        assert result['status'] == selected['status'] == 'completed', tag
        args = {k: v for k, v in result['args'].items() if k not in ('payload', 'tag')}
        if protocol is None:
            protocol = args
        assert args == protocol, ('Changed protocol', tag)
        assert result['presentations_total'] == 131056
        assert abs(selected['selected']['dev_nll'] - min(x['dev_nll'] for x in result['curve'])) < 2e-6
        for name, digest in result['identity']['source_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        suffix = '' if width == 16 else f'_p{width}'
        utility = json.loads((ROOT / f'experiments/results/diagnostics/curie_data_growth_tokens_64k{suffix}_utility_20261005_v1.json').read_text())
        assert utility['status'] == 'completed'
        assert utility['source_sha256'] == hashlib.sha256((ROOT / 'experiments/token_stage_utility.py').read_bytes()).hexdigest()
        row = utility['rows'][0]
        assert row['tag'] == tag and row['selected'] == selected['selected']
        assert row['partition_parity'] and row['matched_rng']
        assert row['checkpoint_sha256'] == hashlib.sha256(Path(row['selected']['checkpoint']).read_bytes()).hexdigest()
        candidates.append((selected['selected']['dev_nll'], result['parameters'], width, row))
    loss, parameters, width, row = min(candidates, key=lambda x: (x[0], x[1]))
    assert row['promotion']['small_fit_promotable'] and row['context_gain'] > 0
    tag = f'curie_data_growth_tokens_64k_b64_c16_p{width}_s7_20261005_v1'
    queue = ROOT / 'experiments/queue' / (tag + '.txt')
    words = queue.read_text().split()
    assert words[0] == tag and words[words.index('--payload')+1] == str(width)
    assert words[words.index('--seed')+1] == '7'
    return dict(status='admitted', queue=str(queue.relative_to(ROOT)), selected_payload=width,
                seed6_selected_nll=loss, seed6_parameters=parameters,
                candidate_losses={str(w): v for v, _, w, _ in candidates},
                scope='One seed7 repeat of the best initial-inclusive completed width; same64K/two-pass recipe and all mechanisms. Estimates selected-member reliability; capacity improvement versus P16 remains seed6 evidence until paired replication. No public scoring or matched-work claim.')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    receipt = admit()
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(receipt['queue'])
