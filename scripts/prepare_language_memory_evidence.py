"""Source-bound investor packet: addressed memory improves native token language (completed Curie results)."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cross = ROOT/'experiments/results/diagnostics/curie_cross_member_language_protocol_20261015_v1.json'
    epi = ROOT/'experiments/results/token_language/curie_episodic_memory_2m_episodic_20261015_v1.json'
    ctl = ROOT/'experiments/results/token_language/curie_episodic_memory_2m_none_20261015_v1.json'
    c, e, n = (json.loads(p.read_text()) for p in (cross, epi, ctl)); assert c['status'] == e['status'] == n['status'] == 'completed'
    rows = {(r['keyed'], r['train_tokens'], r['seed']): r['flagship_protocol_nll'] for r in c['rows']}
    flagship = c['flagship_reference']['nll']; keyed4m = [rows[(1, 4194304, 0)], rows[(1, 4194304, 1)]]
    packet = dict(status='completed', flagship_nll=flagship, keyed_member_4M_nll=keyed4m, keyed_member_4M_mean=sum(keyed4m)/2,
                  keyed_read_ablation_1M=dict(with_read=rows[(1, 1048576, 0)], without_read=rows[(0, 1048576, 0)]),
                  episodic_nll=e['scored']['nll'], episodic_gain_vs_parent=flagship - e['scored']['nll'], identical_stream_plain_continuation=n['scored']['nll'],
                  scope='Same 8,192 FineWeb GPT-2 development targets with 1,024-token history. Keyed member: 3.33M parameters, two seeds, repeated epochs over 4M tokens; flagship and episodic arm: single seed. Quality placement, not matched compute.',
                  input_sha256={str(p.relative_to(ROOT)): sha(p) for p in (cross, epi, ctl)})
    (ROOT/'investment/language_memory_evidence.json').write_text(json.dumps(packet, indent=2) + '\n'); print(json.dumps({k: v for k, v in packet.items() if k not in ('scope', 'input_sha256')}))


if __name__ == '__main__':
    main()
