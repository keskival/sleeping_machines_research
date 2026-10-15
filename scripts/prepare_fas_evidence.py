"""Source-bound investor packet for the FAS v2.1 sealed confirmation (industrial fault detection in interleaved logs)."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = ROOT/'experiments/results/fas/aws_b3_v21_sealed_20261015_v1.json'; r = json.loads(p.read_text())
    assert r['status'] == 'completed' and r['verdict'] == 'win' and all(r['criteria'].values())
    prev = ROOT/'experiments/results/fas/curie_b3_sealed_v4_20261010T0220Z_decision.json'; v2 = json.loads(prev.read_text())
    arms = r['arms']; classical = max((k for k in arms if k.startswith('classical:')), key=lambda k: arms[k]['auroc'])
    packet = dict(status='completed', evidence_level='confirmed win: three native replicates, preregistered sealed fresh draw, frozen B3 Stage 5 rule',
                  native_auroc=r['native_mean'], native_replicates=r['native_seed_aucs'], transformer_auroc=arms['transformer[total:total]']['auroc'],
                  lstm_auroc=max(v['auroc'] for k, v in arms.items() if k.startswith('lstm[')), best_classical=classical, best_classical_auroc=arms[classical]['auroc'],
                  margin=r['difference'], bootstrap_95=r['bootstrap_95'], per_fault_kind={'wear_and_tear': r['per_fault_kind']['1']['native_delay_race'], 'retry_delay': r['per_fault_kind']['2']['native_delay_race'],
                  'transformer_wear_and_tear': r['per_fault_kind']['1']['transformer[total:total]'], 'transformer_retry_delay': r['per_fault_kind']['2']['transformer[total:total]']},
                  samples=r['eligible_samples'], previous_sealed=dict(native=v2['native_mean'], transformer=v2['strongest_auroc'], verdict=v2['verdict']),
                  scope='FAS v2 interleaved assembly-line logs (2 lines, 30 items each, anonymous events); detection at 512 events per line; fresh 2,000+2,000-run test draw generated after the committed preregistration; no item identities used; weights fixed during scoring.',
                  input_sha256={str(p.relative_to(ROOT)): sha(p), str(prev.relative_to(ROOT)): sha(prev), 'experiments/FAS_V21_CONFIRMATION.md': sha(ROOT/'experiments/FAS_V21_CONFIRMATION.md')})
    (ROOT/'investment/fas_evidence.json').write_text(json.dumps(packet, indent=2) + '\n'); print(json.dumps({k: v for k, v in packet.items() if k not in ('scope', 'input_sha256')}))


if __name__ == '__main__':
    main()
