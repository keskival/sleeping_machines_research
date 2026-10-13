"""Publish the complete admitted native addressed-credit cohort."""
import json,subprocess
from pathlib import Path
import build_credit_report_parts as builder
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'experiments/results/market/curie_b10_addressed_clock_20261013_confirmation_v1.json';d=json.loads(p.read_text());assert d['status']=='completed'
verdict='Wins both controls for all three saved parents.' if d['all_parent_wins'] and d['all_uniform_wins'] else 'Mixed cohort outcomes; keep per-parent evidence, no universal promotion.'
summary=f"**Curie addressed credit confirmation — 13 October, completed.** Three saved independently initialized parentcritics, one shared actor/data; native64available/4selected fixed-support residuals. Mean16384DEV NLL{d['mean_nll']:.9f}±{d['sample_sd_nll']:.9f}, meanparent gain{d['mean_parent_gain']:+.9f}/uniform energy-bound gain{d['mean_uniform_gain']:+.9f}. {verdict} Fixed256receipts8192TRAIN, source seed/radius/optimizer unchanged, all outcomes retained; no TEST/independentactor replication/fullwork or energyclaim. Eachfit512extra worlds12288score+8192refresh/4096own-gradient targets within scoring, actor/parent acquisition additional. Maximum residual L2bound matched, realizedupdates/headcapacity differ. Exact fitted selected-readout output/VJP/RNG contract and16xreadout component reduction retained. Next diagnose cohort differences or integrate query-conditioned economical target retrieval; no blind larger grid. AllCurie jobs complete.\n\n"
h=ROOT/'experiments/HANDOFF.md';s=h.read_text();prefix='**Curie fixed parent confirmation — 13 October, admitted/running.**';start=s.find(prefix)
if start>=0:
    end=s.find('\n\n',start);s=s[:start]+('' if end<0 else s[end+2:])
h.write_text(summary+s)
p=ROOT/'report/I_SCIENCE.md';s=p.read_text();prefix='**Addressed credit development (13 October):**';start=s.find(prefix)
if start>=0:
    end=s.find('\n\n',start)
    paragraph=f"**Addressed credit development (13 October):** the fixed three-parent native cohort achieves mean16384DEV NLL{d['mean_nll']:.9f}±{d['sample_sd_nll']:.9f}, mean gain{d['mean_parent_gain']:+.9f}over own parent and{d['mean_uniform_gain']:+.9f}over the uniform energy-bound control. {verdict}64available/4selected clock residuals, native temporal races/private memory;256TRAINreceipts perfit. One sharedactor/reuseddata, sampleSD across critic parents; actor/parent and512extra teacher worlds perfit additionally charged. Maximum residual L2bound matched, not realized updates/headcapacity. Exact selected-row output/gradient/RNG contract cuts readout component arithmetic/operands16fold, not whole training work. [Completed cohort](../experiments/results/market/curie_b10_addressed_clock_20261013_confirmation_v1.json)."
    s=s[:start]+paragraph+('' if end<0 else s[end:]);p.write_text(s)
builder.main()
paths=['scripts/finalize_addressed_credit_confirmation.py','scripts/continue_addressed_credit_confirmation.py','experiments/credit/market_addressed_credit_confirmation.py','experiments/credit/analyze_addressed_credit_confirmation.py','experiments/HANDOFF.md','experiments/market/ADDRESSED_CLOCK_CREDIT_PROTOCOL.md','report/market_future_credit_appendix.py','REPORT.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md','report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf','report/split_manifest.json']
for directory,pattern in [('experiments/queue/market','curie_b10_addressed_clock_20261013_confirm*.txt'),('experiments/results/market','curie_b10_addressed_clock_20261013_confirm*.json')]:paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob(pattern))
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True);subprocess.run(['git','add',*sorted(set(paths))],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Publish fixed three-parent native addressed credit confirmation'],cwd=ROOT,check=True)
