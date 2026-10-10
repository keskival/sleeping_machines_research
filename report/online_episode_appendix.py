"""Completed causal real-data episode bridge; no benchmark win claim."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. Real market and token episodes: causal online adaptation'

def pages():
    path=ROOT/'experiments/results/credit/curie_real_online_bridge_v1_20261010T2050Z_measure.json'
    if not path.exists():return []
    r=json.loads(path.read_text());assert r['status']=='completed';rows=[]
    for domain,arms in r['metrics']['domains'].items():
        for a in arms:rows.append([domain,a['arm'].replace('stateful_','').replace('online_native_','online '),f'{a["nll"]:.5f}',str(a['updates']),f'{a["wall_s"]:.2f}',f'{1000*a["wall_s"]/a["targets"]:.3f}'])
    return [[('h1',TITLE),
        ('p','Both real streams admit useful within-episode adaptation in this short bridge. Issued forecasts are scored before feedback updates; forward numerical state survives credit boundaries. Stateful frozen weights are the control, and the current learning arm is native online BPTT.'),
        ('table',(['Domain','Arm','NLL/target','Updates','Wall s','ms/target'],rows,[22,42,30,21,23,29])),
        ('small','One run, 1,024 fresh TRAIN targets per domain, 128-context observations, update groups of 32, CPU float64 and dropout disabled for both arms. Market model is initialized using only prior context gap statistics; language uses an available 4M-token pretrained keyed member. Language fast weights/optimizer/state reset after five observed EOS boundaries. Full vocabulary output and dense optimizer work are charged. These different warm starts are not compared across domains.'),
        ('p','Online adaptation lowers market NLL by 0.38668 and language NLL by 0.02497 nats/target. Its increased wall cost motivates an economical learned-credit comparison and model development. Cold market parameter fitting is one possible source of the gain; this bridge does not identify nonstationarity as the cause.'),
        ('small',f'Whole bridge including setup/copies and both domains {r["metrics"]["whole_wall_s"]:.2f} s; peak RSS {r["peak_rss_kb"]/1024:.1f} MiB. Per-arm wall includes context replay/scoring/backward/updates, but excludes model copy and initial optimizer construction. Language warm-start fitting cost {r["metrics"]["warm_start"]["language"]["prior_fit_wall_s"]:.1f} s is separate and must be included in a complete-system comparison.'),
        ('p','Real-market state and fixed-parameter gradient contracts pass at depths 2/4/8, with saved-market score parity error 1.9e-10. A trained proper-token member preserves lazy addressed key transport, scores and every factual parameter gradient in float64 to 3.7e-15. Detachment preserves numbers while truncating credit; issued forecasts cannot be rewritten by later updates.'),
        ('small','No public language scoring interval or market TEST is read. Cached key-response algebra is valid within its fixed-learning-path assumptions; across changed optimizer updates the total effect needs versioned replay or audited transition prediction. Serial event-time CPU processing is not clockless hardware measurement.')]]
