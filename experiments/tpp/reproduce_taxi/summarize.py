"""Summarize a Taxi reproduction against the published reference numbers."""
import json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REF = {'AWS (ours, 7 Oct 2026)': (0.5250, 0.0010), 'curie (ours, independent hardware)': (0.5252, 0.0007),
       'S2P2 (best published, EasyTPP protocol)': (0.522, 0.004)}

rows = []
for tag in sys.argv[1:]:
    d = json.loads((ROOT / f'experiments/results/tpp/{tag}.json').read_text())
    t = d['test']; rows.append(t['ll'])
    print(f"{tag}: test ll {t['ll']:.5f} nats/event (time {t['time_ll']:.4f}, mark {t['mark_ll']:.4f}), "
          f"rmse {t['rmse']:.4f}, acc {t['acc']:.4f}, events {t['events']}, wall {d['wall_s']:.0f} s")
if rows:
    sd = st.stdev(rows) if len(rows) > 1 else float('nan')
    print(f"\nThis run: {st.mean(rows):.4f} +- {sd:.4f} (n={len(rows)})")
for k, (m, s) in REF.items():
    print(f"{k}: {m:.4f} +- {s:.4f}")
