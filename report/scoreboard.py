"""Scoreboard (experiments/PRODUCT_ORDERS.md P0-5): every headline comparison as a win, loss, efficiency point or pending,
per experiments/WIN_CRITERIA.md, generated from completed result files on every report build."""
import math


def eligible(row):
    """Never replace a T256 comparison with a training-window or pending score."""
    q = row.get('test256')
    return (row.get('comparison_eligible', True) and isinstance(q, (int, float))
            and math.isfinite(q))


def best_within(rows, budget, key):
    if not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('Finite positive reference budget required')
    candidates = [r for r in rows if eligible(r) and isinstance(r.get(key), (int, float))
                  and math.isfinite(r[key]) and 0 < r[key] <= budget]
    return min(candidates, key=lambda r: r['test256']) if candidates else None


def page(native, public):
    rows = []
    # 10M matched-compute comparisons
    nat = [r for r in native['native'] if 'route credit' in r['label']]
    for c in native['controls']:
        for axis, key, unit, scale in (('training', 'whole', 'TF', 1e12), ('inference', 'infer', 'MF/pos', 1e6)):
            best = best_within(nat, c[key], key if axis == 'training' else 'sparse')
            if best is None:
                rows.append([f"10M vs {c['label']} at ≤ its {axis} compute", f"{c['test']:.3f}", 'no native row within budget',
                             'Pending']); continue
            q = best['test256']
            used = best[key] if axis == 'training' else best['sparse']
            rows.append([f"10M vs {c['label']} at ≤ its {axis} compute ({c[key] / scale:.1f} {unit})", f"{c['test']:.3f}",
                         f"{q:.3f} ({best['label']}; {used / scale:.1f} {unit})", 'WIN' if q < c['test'] else 'Loss'])
    # P0-6 tuned dense references at the native budgets (TUNED_BASELINES.md): validation-selected arm per budget
    from pathlib import Path
    import runpy
    admit=runpy.run_path(str(Path(__file__).with_name('tuned_reference_admission.py')))['load_group']
    for budget, member in (('A','p96/d4 + route credit, 6 passes'),('B','p64/d4 + route credit, 4 passes')):
        best=next((r for r in nat if r['label']==member and eligible(r)),None)
        if best is None:
            rows.append([f'10M vs tuned dense budget {budget} (P0-6)','pending','native anchor missing','Pending']);continue
        cap=best['whole'];label=f'≤ {cap/1e12:.0f} TF';group=admit(budget,cap)
        if group['status']!='completed':
            detail=(f"{group['completed']}/{group['required']} arms complete" if group['status']=='pending'
                    else 'align LSTM validation/test contexts')
            rows.append([f"10M vs tuned dense at {label} (P0-6)",'pending',detail,'Pending']);continue
        ref=group['selected'];q=best['test256']
        a = ref['args']; name = f"{a['model']}{a['size']}" + (f"x{a['layers']}" if a['model'] == 'tf' else '') + f" {a['passes']:g}p lr{a.get('lr')}"
        rows.append([f"10M vs tuned dense at {label} (P0-6; {group['required']} arms, validation-selected)",
                     f"{ref['test_bpc']:.3f} ({name})", f"{q:.3f} ({best['label']}; {best['whole'] / 1e12:.0f} TF)",
                     'WIN' if q < ref['test_bpc'] else 'Loss'])
        tfs=[r for r in group['rows'] if r['args']['model']=='tf']
        if tfs:                                   # architecture sub-comparison, also validation-selected
            t=min(tfs,key=lambda r:r['selection_valid']);ta=t['args'];ratio=best['whole']/t['whole']
            verdict=('WIN' if q<t['test_bpc'] and ratio<=1 else
                     f"Better quality at {100*ratio:.0f}% of its compute (not matched)" if q<t['test_bpc'] else 'Loss')
            rows.append([f"10M vs tuned Transformers at {label} (P0-6; {len(tfs)} arms, validation-selected)",
                         f"{t['test_bpc']:.3f} (tf{ta['size']}x{ta['layers']} {ta['passes']:g}p lr{ta.get('lr')})",
                         f"{q:.3f} ({best['label']}; {best['whole'] / 1e12:.0f} TF)",verdict])
    # 90M
    if native.get('native90'):
        for c in native['controls90']:
            dominated=[r for r in native['native90'] if eligible(r)
                       and c['whole'] <= r['whole'] and c['test'] < r['test256']]
            if dominated:
                strongest=min(dominated,key=lambda r:r['test256'])
                rows.append([f"90M four-pass vs {c['label']}",
                    f"{c['test']:.3f} ({c['whole']/1e12:.0f} TF)",
                    f"{strongest['test256']:.3f} ({strongest['label']}; {strongest['whole']/1e12:.0f} TF)",
                    'LOSS (single seed)'])
                continue
            b = best_within(native['native90'], c['whole'], 'whole')
            if b is None:
                rows.append([f"90M vs {c['label']}", f"{c['test']:.3f}", 'no native T256 row within budget', 'Pending'])
                continue
            ratio = c['whole'] / b['whole']
            rows.append([f"90M vs {c['label']}", f"{c['test']:.3f}",
                         f"{b['test256']:.3f} ({b['label']}; {ratio:.0f}× less training compute)",
                         'WIN' if b['test256'] < c['test'] else 'Efficiency point; run queued'])
    # NeuroBench
    mg = public['mg']
    if mg['n']:
        q = mg['modes']['mix8']
        verdict = ('WIN' if q < 13.37 else 'Loss vs LSTM') if mg.get('protocol_claim_eligible', False) else f"Pending ({mg['n']}/30 repeats)"
        rows.append(['NeuroBench Mackey-Glass (sMAPE; LSTM 13.37, ESN 14.79)', '13.37', f"{q:.2f} (57.6 KB vs 490 KB)", verdict])
    if public['primate']:
        b = max(public['primate'], key=lambda r: r['val'])          # validation-selected, never by test
        rows.append(['NeuroBench primate reaching (R²; leaderboard 0.71 six-session)', '0.710',
                     f"{b['test']:.3f} (validation-selected arm, development session indy_20170131_02, also one of the six "
                     f"official sessions; tinyRSNN .746 there)", 'Pending (six-session run; also report the five untouched sessions)'])
    rows.append(['SHD (accuracy; best published 96.4%)', '96.4%', 'development queued', 'Pending'])
    wins = sum(r[3] == 'WIN' for r in rows)
    return [('h1', 'Scoreboard — wins, losses and open targets'),
            ('p', f"<b>{wins} wins against saved references</b> in {len(rows)} headline comparisons. Definitions: experiments/WIN_CRITERIA.md; orders: "
                  'experiments/PRODUCT_ORDERS.md. Single seeds unless stated; tuned references and confirming seeds are pending.'),
            ('table', (['Comparison', 'Reference', 'Ours', 'Verdict'], rows, [66, 20, 64, 28])),
            ('small', 'Native compute is traced (fitting extrapolated from traced windows; inference from the exact winner-only '
                      'trace); references use the saved shape estimates or the leaderboard\'s published counts. Multi-pass native '
                      'rows may use more optimizer updates than one-pass references. A native row qualifies for a budget only if '
                      'its own estimate does not exceed the reference\'s. Native and Transformer score the same 999,936 targets '
                      'with reset T256 windows; saved LSTMs carry state across 999,999 targets of the same test interval. '
                      'LSTM rows are saved-reference quality/work wins or losses; identical-context rescoring is pending.')]
