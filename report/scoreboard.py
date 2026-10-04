"""Scoreboard (experiments/PRODUCT_ORDERS.md P0-5): every headline comparison as a win, loss, efficiency point or pending,
per experiments/WIN_CRITERIA.md, generated from completed result files on every report build."""


def page(native, public):
    rows = []
    # 10M matched-compute comparisons
    nat = [r for r in native['native'] if 'route credit' in r['label']]
    for c in native['controls']:
        for axis, key, unit, scale in (('training', 'whole', 'TF', 1e12), ('inference', 'infer', 'MF/pos', 1e6)):
            within = [r for r in nat if (r[key] if axis == 'training' else (r.get('sparse') or float('inf'))) <= c[key]]
            if not within:
                rows.append([f"10M vs {c['label']} at ≤ its {axis} compute", f"{c['test']:.3f}", 'no native row within budget',
                             'Loss']); continue
            best = min(within, key=lambda r: r['test256'] if r['test256'] is not None else r['test'])
            q = best['test256'] if best['test256'] is not None else best['test']
            used = best[key] if axis == 'training' else best['sparse']
            rows.append([f"10M vs {c['label']} at ≤ its {axis} compute ({c[key] / scale:.1f} {unit})", f"{c['test']:.3f}",
                         f"{q:.3f} ({best['label']}; {used / scale:.1f} {unit})", 'WIN' if q < c['test'] else 'Loss'])
    # P0-6 tuned dense references at the native budgets (TUNED_BASELINES.md): validation-selected arm per budget
    import glob, json, os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments/results/aws_20260929')
    for budget, cap, label in (('A', 352.1e12, '≤ 352 TF'), ('B', 107.2e12, '≤ 107 TF')):
        arms = []
        for f in glob.glob(os.path.join(root, f'aws_tuned_ref_10M_{budget}_*', '*_D10000000_*.json')):
            r = json.load(open(f))
            if r.get('best_valid_bpc') is not None and r['training_flops_estimate']['total_training_flops'] <= cap:
                arms.append(r)
        within = [r for r in nat if r['whole'] <= cap]
        if not arms or not within:
            rows.append([f"10M vs tuned dense at {label} (P0-6)", 'pending', 'queued on AWS', 'Pending']); continue
        ref = min(arms, key=lambda r: r['best_valid_bpc'])          # selected by validation, never by test
        best = min(within, key=lambda r: r['test256'] if r['test256'] is not None else r['test'])
        q = best['test256'] if best['test256'] is not None else best['test']
        a = ref['args']; name = f"{a['model']}{a['size']}" + (f"x{a['layers']}" if a['model'] == 'tf' else '') + f" {a['passes']:g}p lr{a.get('lr')}"
        rows.append([f"10M vs tuned dense at {label} (P0-6; {len(arms)} arms, validation-selected)",
                     f"{ref['test_bpc']:.3f} ({name})", f"{q:.3f} ({best['label']}; {best['whole'] / 1e12:.0f} TF)",
                     'WIN' if q < ref['test_bpc'] else 'Loss'])
    # 90M
    if native.get('native90'):
        b = min(native['native90'], key=lambda r: r['test256'])
        for c in native['controls90']:
            ratio = c['whole'] / b['whole']
            rows.append([f"90M vs {c['label']}", f"{c['test']:.3f}",
                         f"{b['test256']:.3f} ({b['label']}; {ratio:.0f}× less training compute)",
                         'WIN' if b['test256'] < c['test'] else 'Efficiency point (matched-compute run queued)'])
    # NeuroBench
    mg = public['mg']
    if mg['n']:
        q = mg['modes']['mix8']
        verdict = ('WIN' if q < 13.37 else 'Loss vs LSTM') if mg['n'] == 30 else f"Pending ({mg['n']}/30 repeats)"
        rows.append(['NeuroBench Mackey-Glass (sMAPE; LSTM 13.37, ESN 14.79)', '13.37', f"{q:.2f} (57.6 KB vs 490 KB)", verdict])
    if public['primate']:
        b = max(public['primate'], key=lambda r: r['test'])
        rows.append(['NeuroBench primate reaching (R²; leaderboard 0.71 six-session)', '0.710',
                     f"{b['test']:.3f} (one development session; tinyRSNN .746 there)", 'Pending (six-session run)'])
    rows.append(['SHD (accuracy; best published 96.4%)', '96.4%', 'development queued', 'Pending'])
    wins = sum(r[3] == 'WIN' for r in rows)
    return [('h1', 'Scoreboard — wins, losses and open targets'),
            ('p', f"<b>{wins} wins</b> in {len(rows)} headline comparisons. Definitions: experiments/WIN_CRITERIA.md; orders: "
                  'experiments/PRODUCT_ORDERS.md. Same data and test sets; T256 language evaluation; single seeds unless stated.'),
            ('table', (['Comparison', 'Reference', 'Ours', 'Verdict'], rows, [66, 20, 64, 28])),
            ('small', 'Native compute is traced (fitting extrapolated from traced windows; inference from the exact winner-only '
                      'trace); references use the saved shape estimates or the leaderboard\'s published counts. Multi-pass native '
                      'rows may use more optimizer updates than one-pass references. A native row qualifies for a budget only if '
                      'its own estimate does not exceed the reference\'s.')]
