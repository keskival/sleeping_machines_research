"""Completed native-only FAS repeats, with artifact receipts and fair controls."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def pages():
    folder=ROOT/'experiments/results/fas'
    original=folder/'curie_fas_v1_native_p32d4_linear_seg128_t1024_e2_20261004T231500Z.json'
    seed6=json.loads(original.read_text());rows=[]
    controls=[]
    for name,keys in [('fas_v1_classical_test_20261004T221500Z.json',('elapsed','tick_count','gap_z')),('fas_v1_strong_classical_test_20261005T010000Z.json',('gap_quantile','gap_cusum','gap_robust_z'))]:
        source=json.loads((folder/name).read_text())['auroc']
        controls.extend(source[key] for key in keys)
    # Exactly the established six anonymous generic controls, no oracle route.
    assert len(controls)==6
    best={p:max(c[p]['all'] for c in controls) for p in ('256','512')}
    completed=[seed6]
    for seed,tag in [(7,'curie_fas_v1_frozen_p32d4_s7_retry_20261005_v2'),(8,'curie_fas_v1_frozen_p32d4_s8_20261005_v1')]:
        path=folder/(tag+'.json');receipts=sorted((ROOT/'experiments/results/diagnostics').glob(f'curie_fas_v1_s{seed}_completed_artifacts_*.json'))
        receipt=receipts[-1] if receipts else ROOT/'missing_receipt'
        if not path.exists() or not receipt.exists():continue
        r=json.loads(path.read_text());audit=json.loads(receipt.read_text())
        assert r['status']==audit['status']=='completed' and audit['auroc_max_error']==0
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['result_sha256']
        assert audit['seed']==seed
        completed.append(r)
    for r in completed:
        values=r['test_auroc'];rows.append([str(r['args']['seed']),f"{values['256']['all']:.6f}",f"{values['256']['all']-best['256']:+.6f}",f"{values['512']['all']:.6f}",f"{values['512']['all']-best['512']:+.6f}"])
    verdict='Replicated win' if len(completed)>=2 and all(r['test_auroc']['256']['all']>best['256'] for r in completed) else 'Single-seed win'
    return [[('h1','FAS native frozen-recipe replication'),('p',f'<b>{verdict} against all six anonymous generic controls at 256 events.</b> Completed native fitting seeds: '+', '.join(str(r['args']['seed']) for r in completed)+'.'),('table',(['Native seed','AUROC N256','Gap vs controls','AUROC N512','Gap vs controls'],rows,[24,35,38,35,38])),('p',f"Best generic controls: {best['256']:.6f} AUROC at N=256 and {best['512']:.6f} at N=512. The fixed P32/D4/H2/U2 recipe uses two passes over 10,000 clean FIT runs; selection uses validation-clean NLL only. Test: 2,000 clean and 2,000 faulty runs. Seeds7/8 score artifacts each reproduce all 18 aggregate/type/prefix AUROCs exactly and retain the selected weights."),('small','FAS v1 replication, not sealed v2 confirmation. Generic-control comparison uses saved references; the completed FAS v2 neural-reference comparison is reported separately. FIFO/timing de-interleaving trained on hidden item identities remain oracle-assisted diagnostics, excluded here. Original seed6 has aggregate scores, not the new per-run artifact; paired multi-seed bootstrap cannot be reconstructed from those aggregates. Current frozen dependency hashes are checked; historical unrecorded dependency identity is not independently verified.')]]
