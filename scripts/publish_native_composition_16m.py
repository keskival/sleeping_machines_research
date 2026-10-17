"""Refresh current investor text from completed source-bound 16M development runs."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 paths=['experiments/results/token_language/aws_r1_rmr_compact_K4_16M_s0_20261016_v2.json','experiments/results/token_language/aws_r1_cerl_16M_s0_20261016_v1.json']
 rows=[]
 for name in paths:
  p=ROOT/name;d=json.loads(p.read_text());assert d['status']=='completed'
  rows.append(dict(result=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),flagship_nll=d['flagship']['nll'],long_slice_nll=d['score_65528']['nll']))
 packet=dict(scope='One seed; one pass over 16M FineWeb GPT-2 tokens; development evaluation, no public TEST. Carrier-backed current-slot residual-memory composition; old-slot rewrites disabled. Full learning arithmetic and hardware benefit not audited.',results=rows,flagship_gain=rows[1]['flagship_nll']-rows[0]['flagship_nll'])
 (ROOT/'investment/native_composition_16m_evidence.json').write_text(json.dumps(packet,indent=2)+'\n')
 paragraph='**Completed native language composition at 16M tokens.** Four-level residual memory scores **5.0433 NLL**, improving the native causal-race member\'s **5.0634** by **0.0201 nats per token**. The saved sixteen-layer dense hybrid scores **5.2034**. One seed, one pass over 16M FineWeb GPT-2 tokens; the flagship evaluation uses 8,192 development targets in 1,024-token windows. The separate 65,528-target slice improves from 5.0282 to 5.0244. This supports useful composition in the carrier-backed current-slot memory variant; old-slot rewrites are disabled, and it does not establish fully asynchronous deep learning or a public leaderboard win. Complete learning arithmetic is not audited. [Bound evidence](native_composition_16m_evidence.json).'
 for name in ['investment/PITCH.md','investment/INVESTMENT_CASE.md','investment/INVESTOR_PROOF_PLAN.md','report/I_SCIENCE.md']:
  p=ROOT/name;s=p.read_text();label='**Completed native language composition at 16M tokens.**'
  if label in s:s=re.sub(r'\*\*Completed native language composition at 16M tokens\.\*\*[^\n]*',lambda _:paragraph,s)
  else:s=re.sub(r'\*\*Completed 16M hybrid diagnostic\.\*\*[^\n]*',lambda _:paragraph,s)
  if name.startswith('report/'):s=s.replace('[Bound evidence](native_composition_16m_evidence.json)','[Bound evidence](../investment/native_composition_16m_evidence.json)')
  p.write_text(s)
 p=ROOT/'investment/brief/evidence_brief.tex';s=p.read_text()
 tex=r"\textbf{Native language composition at 16M tokens.} Four-level residual memory scores \textbf{5.0433 NLL}, improving the native causal-race member's \textbf{5.0634}; the saved sixteen-layer dense hybrid scores 5.2034. One seed, one pass over 16M FineWeb GPT-2 tokens; 8,192 development targets in 1,024-token windows. Carrier-backed current-slot memory composition, with old-slot rewrites disabled. This is a development accuracy result; fully asynchronous deep learning and complete learning arithmetic are separate tests. Source: \path{investment/native_composition_16m_evidence.json}."
 s=re.sub(r'\\textbf\{(?:Completed 16M hybrid diagnostic\.|Native language composition at 16M tokens\.)\}[^\n]*',lambda _:tex,s);p.write_text(s)
if __name__=='__main__':main()
