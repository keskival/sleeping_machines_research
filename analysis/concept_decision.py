"""R1 concept-hierarchy decision (theory/PART_WHOLE_COMPOSITION.md section 5, P7a-c). Reads completed one-pass records
and the document-concept curve of the same runs; no fitting.
P7a: concept hierarchy + tree read beats its plan-B L2 base on 1024-token lanes (flagship protocol and last-quarter
     prequential loss, same 4M fresh tokens, seed and schedule).
P7b: its document-context value (word-initial targets, document positions 256-1024, history cut to 64 tokens) exceeds
     the base's and the no-tree hierarchy's.
P7c: the tree read is responsible: with tree < without tree on the flagship protocol.
"""
import argparse, json, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--base', required=True); p.add_argument('--tree', required=True)
    p.add_argument('--no-tree', required=True); p.add_argument('--curve', required=True); a = p.parse_args(); began = time.monotonic()
    rec = lambda t: (json.loads((ROOT / f'experiments/results/token_language/{t}.json').read_text()) if (ROOT / f'experiments/results/token_language/{t}.json').exists() else None)
    curve = json.loads((ROOT / f'experiments/results/diagnostics/{a.curve}.json').read_text())['rows']
    rows = {}
    for name, t in (('base', a.base), ('tree', a.tree), ('no_tree', a.no_tree)):
        r = rec(t)
        rows[name] = None if r is None or r.get('status') != 'completed' else dict(tag=t, flagship=r['flagship_protocol']['nll'], prequential_last_quarter=r['prequential_last_quarter'],
            score_65528=r['score_65528']['nll'], train_wall_s=r['train_wall_s'], parameters=r['parameters'], concept_slots_used=r.get('concept_slots_used'),
            document_context_value_word_initial_256_1024=curve.get(t, {}).get('256-1024', {}).get('word_initial_document_context_value'))
    B, T, N = rows['base'], rows['tree'], rows['no_tree']; decision = {}
    if B and T:
        decision.update(P7a_flagship_gain=B['flagship'] - T['flagship'], P7a_prequential_gain=B['prequential_last_quarter'] - T['prequential_last_quarter'],
                        P7a=T['flagship'] < B['flagship'] and T['prequential_last_quarter'] < B['prequential_last_quarter'])
        dv = lambda r: r and r['document_context_value_word_initial_256_1024']
        if dv(B) is not None and dv(T) is not None:
            decision['P7b_document_context_value'] = dict(base=dv(B), tree=dv(T), no_tree=dv(N) if N else None)
            decision['P7b'] = dv(T) > dv(B) and (N is None or dv(T) > dv(N))
    if T and N: decision.update(P7c_tree_gain=N['flagship'] - T['flagship'], P7c=T['flagship'] < N['flagship'])
    out = dict(status='completed', battle='R1', tag=a.tag, rows=rows, decision=decision, curve=a.curve,
               source_sha256={'analysis/concept_decision.py': hashlib.sha256((ROOT / 'analysis/concept_decision.py').read_bytes()).hexdigest()},
               process_wall_s=time.monotonic() - began, scope='Single seed, one pass over the same 4M fresh tokens on 1024-token lanes; development protocol.')
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
