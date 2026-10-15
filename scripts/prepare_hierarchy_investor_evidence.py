"""Bind completed language-depth evidence and untrained hierarchy contracts."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    paths=[f'experiments/results/token_language/aws_r1_deep_keyed_L{depth}_1M_s0_20261015_v3.json' for depth in (2,4)]
    results=[json.loads((ROOT/p).read_text()) for p in paths]
    for d in results:
        assert d['status']=='completed' and d['score_65528']['targets']==65528
    for k in ['seed','train_tokens','lane','d','modes','heads','dk','dropout','lr','wd','batch']:
        assert results[0]['args'][k]==results[1]['args'][k],k
    contract='experiments/results/diagnostics/addressable_parent_contracts_20261015_v1.json'
    receipt=json.loads((ROOT/contract).read_text()); assert receipt['status']=='completed' and receipt['tests']==29
    theory='experiments/theory/ADDRESSABLE_COMPOSITION_MEMORY.md'
    packet=dict(status='completed',language_depth=dict(evidence_level='single-seed within-family development',
        targets=65528,unique_training_tokens=results[0]['args']['train_tokens'],seed=0,
        rows=[dict(layers=d['args']['layers'],nll=d['score_65528']['nll'],parameters=d['parameters'],epochs=d['epochs_run']) for d in results],
        nll_improvement=results[0]['score_65528']['nll']-results[1]['score_65528']['nll'],
        scope='Same scored slice and unique training corpus; different epochs and total work. Quality gain only; no matched-compute or semantic-hierarchy claim.'),
        hierarchical_memory=dict(status='implemented, untrained',abstraction_levels=8,refinements_per_level=4,cells=32,passed_contracts=29,
        capability='Higher parents query retained input and lower-level memories, select complementary child roles, compose relations and refine persistent state.',
        scope='Synthetic numerical contracts, not learned semantic concepts, compositional generalization or measured sparse execution.',
        next_gate='Real-TRAIN resource/credit preflight, then bounded learned composition and controlled read-source evaluation; no current fit result inferred.'),
        input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [*paths,contract,theory]})
    (ROOT/'investment/hierarchy_development_evidence.json').write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps(packet['language_depth']))
if __name__=='__main__':main()
