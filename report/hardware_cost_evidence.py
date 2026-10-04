"""Report page: P2 hardware cost model (per-character traffic and priced energy) from its saved result file."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'experiments/results/diagnostics/hardware_cost_model_20261004.json'


def pages():
    if not SRC.exists():
        return []
    r = json.loads(SRC.read_text())
    rows = [[x['model'], f"{x['parameters'] / 1e6:.2f}M", f"{x['weights_read'] / 1e3:.0f}K ({100 * x['weights_read_fraction']:.0f}%)",
             f"{x['state_read'] / 1e3:.1f}K", f"{x['bytes_moved_per_char_int8'] / 1e3:.0f}K",
             f"{x['pj_per_char_sram_local_8KB'] / 1e6:.2f}", f"{x['pj_per_char_dram'] / 1e6:.0f}"] for x in r['rows']]
    return [[('h1', 'Hardware cost model: bytes moved per character'),
             ('p', 'Per-character steady-state inference reads/writes, priced with Horowitz 45 nm anchors (int8, batch 1). '
                   'Native rows use the exact cached-key winner-only execution (§414); its p96 count agrees with the recorded '
                   '1.3 MF/position trace. The 1.888-bpc native model moves 5.8× fewer bytes than the Transformer-256×4 it beats '
                   '(1.908) and 1.9× fewer than LSTM-512 (1.799, better quality). From pool 2 to pool 32, parameters grow 10.5× '
                   'and bytes per character 5%: capacity beyond activity at the traffic level (pool-32 quality not measured).'),
             ('table', (['Model', 'Params', 'Weights read/char', 'State read/char', 'Bytes/char', 'µJ/char SRAM', 'µJ/char DRAM'],
                        rows, [62, 18, 30, 24, 20, 20, 20])),
             ('small', r['scope'] + ' Source: ' + str(SRC.relative_to(ROOT)) + '; experiments/hardware_cost_model.py; '
                       'investment/HARDWARE_THESIS.md. Not a chip, cache or joule measurement.')]]
