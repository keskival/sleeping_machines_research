"""Build a source-bound two-page investor demonstration from saved evidence only."""
import argparse,hashlib,json,math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def prepare():
 race='experiments/results/token_language/curie_r1_cerl_4M_s0_20261016_v1.json';dense='experiments/results/token_language/curie_r1_deep_keyed_stream_L8_4M_s0_20261015_v1.json';analysis='experiments/results/diagnostics/curie_cerl_vs_dense_4M_error_analysis_20261016_v1.json'
 a=json.loads((ROOT/race).read_text());b=json.loads((ROOT/dense).read_text());e=json.loads((ROOT/analysis).read_text())
 assert a['status']==b['status']=='completed'
 assert all(a['args'][k]==b['args'][k] for k in ['seed','train_tokens','lane','batch','lr','wd'])
 assert a['flagship']['targets']==b['flagship_protocol']['targets']==8192
 gate='experiments/results/diagnostics/aws_r1_compositional_producer_credit_gate_20261016_v1.json'
 gate_status='queued'
 inputs=['scripts/build_composition_investor_demo.py',race,dense,analysis,'experiments/theory/COMPOSITION_LEARNING_CONTRACT.md','sleeping_machines/compositional_producer_credit.py','tests/test_compositional_producer_credit.py','experiments/queue/aws_model_improvement_repair_20261005T161000Z/addenda/000_aws_r1_000_composition_learning_contract_20261016_v1.json']
 if (ROOT/gate).exists():
  g=json.loads((ROOT/gate).read_text());gate_status='passed' if g.get('status')=='completed' and g.get('gate')=='PASS' else 'requires_review';inputs.append(gate)
 row=dict(status='completed',packet_kind='completed-evidence-and-current-demonstration-plan',battle='R1',date='2026-10-16',prediction=dict(race_nll=a['flagship']['nll'],dense_nll=b['flagship_protocol']['nll'],gain_nats=b['flagship_protocol']['nll']-a['flagship']['nll'],targets=8192,train_tokens=a['args']['train_tokens'],seeds=1),error_strata=e['flagship']['by_copyable'],cause_strata=e['flagship']['by_winning_cause'],native_training_flops_per_target=a['train_flops_per_target'],composition_credit=dict(implementation='implemented',numerical_gate=gate_status,max_child_edges_per_root=6,max_child_edges_per_path=2,new_parameters=0),next_proofs=['Deeper models improve same-protocol held-out prediction, including novel tokens and relations.','More stored capacity delivers useful quality while full selected activity and discovery/learning work are reported.','Actual-child credit passes oracle, factual parity, work cap and acquired optimizer/recovery tests.','Independent event rates reduce measured work or traffic while retaining quality.'],scope='Completed race-language development win: one seed, same4M-token data/order/schedule and flagship protocol;different model sizes. Winner retains two temporal-memory carrier layers/BPTT. Fully native deep local-learning and residual-memory scaling remain separately labelled development. No public language leaderboard, hardware energy/latency or scaling-supremacy claim.',input_sha256={p:sha(p) for p in inputs})
 (ROOT/'investment/composition_demonstration_evidence.json').write_text(json.dumps(row,indent=2)+'\n');return row

def build(r):
 out=ROOT/'investment/sleeping_machines_composition_demonstration.pdf';c=canvas.Canvas(str(out),pagesize=A4);W,H=A4
 ink=HexColor('#15243B');blue=HexColor('#175BCE');green=HexColor('#087C65');muted=HexColor('#53637A');pale=HexColor('#EDF3FA')
 styles={k:ParagraphStyle(k,fontName='Helvetica',fontSize=size,leading=leading,textColor=ink) for k,size,leading in [('body',10.5,15),('small',8.5,12),('card',10,14),('title',17,22)]}
 def para(text,x,y,width,kind='body'):
  p=Paragraph(text,styles[kind]);_,height=p.wrap(width,H);p.drawOn(c,x,y-height);return y-height
 def title(kicker,text,sub):
  c.setFillColor(blue);c.setFont('Helvetica-Bold',10);c.drawString(42,H-42,kicker)
  y=para('<b>'+text+'</b>',42,H-65,W-84,'title');return para(sub,42,y-12,W-84)-22
 def footer(n):
  c.setFillColor(muted);c.setFont('Helvetica',8);c.drawString(42,25,'Sleeping Machines | Private investor review | 16 October 2026');c.drawRightString(W-42,25,str(n))
 y=title('A GENERAL-PURPOSE COMPUTING SUBSTRATE','Race prediction works. Now make useful representations deepen.','Our ambition spans language and reasoning, multimodal world models, embodiment, continual learning and specialized hardware. This demonstration connects one measured language result to the next scaling proof.')
 y=para('<b>Measured: better language prediction on the same 4M-token development recipe.</b>',42,y,W-84)-15
 p=r['prediction'];maxv=6.2;chartx=165;chartw=W-chartx-65
 for label,val,color in [('Dense hybrid',p['dense_nll'],muted),('Causal race model',p['race_nll'],green)]:
  c.setFillColor(ink);c.setFont('Helvetica',10);c.drawString(42,y-14,label);c.setFillColor(color);c.roundRect(chartx,y-24,chartw*val/maxv,22,3,fill=1,stroke=0);c.setFillColor(white);c.setFont('Helvetica-Bold',11);c.drawRightString(chartx+chartw*val/maxv-8,y-17,f'{val:.4f}');y-=40
 y=para(f'<b>{p["gain_nats"]:.4f} nats less loss per token.</b> Lower is better. Both models score the same 8,192 development targets in 1,024-token windows; one seed and different model sizes. Bars start at zero.',42,y-3,W-84)-18
 y=para('<b>Why this matters</b><br/>The winning model predicts through competing causes: earlier text, continuing spans, learned concept units and a background route. Its training estimate is 7.74 million FLOPs per target. This is evidence that our prediction law can compete in language.',42,y,W-84)-18
 y=para('<b>Where the advantage comes from</b>',42,y,W-84)-12
 for label,key in [('Tokens already seen in the window','seen_earlier_in_window'),('Tokens new to the window','novel_in_window')]:
  s=r['error_strata'][key];y=para(f'{label}: <b>{s["cerl"]:.2f} versus {s["dense"]:.2f}</b> NLL (race versus dense).',42,y,W-84)-8
 y=para('These are descriptive subsets of the saved predictions. Remembered evidence supplies a strong foothold; novel prediction and relations are the opportunity for deeper composition.',42,y-5,W-84)-16
 y=para('<b>Scope of the win</b><br/>The winner uses two temporal-memory carrier layers and BPTT. The fully native eight-layer event-local learner is a separate development track. This is a development comparison, separate from the 13 public benchmark wins.',42,y,W-84)
 assert y>45,y
 footer(1);c.showPage()
 y=title('THE COMPOSITION DEMONSTRATION','Memories hold meaning. Events schedule refinement.','Higher stages query several lower memories, compose their relationships, and write persistent refinements. Later task success must teach the producers of those memories.')
 def box(x,yy,w,h,heading,body):
  c.setFillColor(pale);c.roundRect(x,yy-h,w,h,7,fill=1,stroke=0);para('<b>'+heading+'</b><br/>'+body,x+12,yy-10,w-24,'card')
 box(90,y, W-180,52,'Task prediction','Read the useful refined memories.');top=y;y-=78
 box(90,y,W-180,68,'Higher memory: a reusable whole','Combine participant identities and relationships; retain the refinement between events.');mid=y;y-=96
 box(42,y,240,65,'Lower memory A','Persistent evidence from many events.');box(W-282,y,240,65,'Lower memory B','A distinct participant with its own history.')
 c.setStrokeColor(blue);c.setLineWidth(2)
 for x in [162,W-162]:c.line(x,y+8,W/2,mid-72)
 c.line(W/2,mid+7,W/2,top-56)
 c.setStrokeColor(green);c.setDash(4,3);c.line(W-30,top-10,W-30,y-42);c.setDash();c.setFillColor(green);c.setFont('Helvetica',8);c.drawRightString(W-70,mid-78,'Delayed task credit');y-=82
 y=para('Architecture schematic; learned semantic hierarchy is the next proof. Events can arrive at different rates; a persistent memory can remain useful between updates.',42,y,W-84,'small')-18
 y=para('<b>Implemented now</b><br/>Bounded credit replay follows the actual lower-memory versions used by a historical composer. It adds no parameters and is designed to preserve factual predictions. Its additional replay is capped at six child reads per root and two edges per path.',42,y,W-84)-15
 label='passed' if r['composition_credit']['numerical_gate']=='passed' else 'queued; not yet passed'
 y=para('<b>Numerical validation: '+label+'.</b><br/>The gate checks an independent finite-computation gradient, unchanged predictions and RNGs, bounded work, and update/recovery on a private copy of the acquired eight-layer learner.',42,y,W-84)-15
 y=para('<b>The investor proof milestone</b><br/>Deeper refinements must improve held-out prediction and relations at disclosed total learning work. Larger memory must produce useful quality while activity, discovery, traffic and storage are measured. Independent firing must save actual work while retaining quality.',42,y,W-84)-12
 y=para('Existing depth/capacity experiments are running; composition-credit validation is queued. Historical route/timing and some input provenance remain outside the new bounded credit path. No new scaling or hardware-energy result is claimed.',42,y,W-84,'small')-12
 y=para('Sources: composition_demonstration_evidence.json binds the original results, error analysis, theory, implementation and immutable gate admission. Use the accompanying evidence brief for the full public benchmark record.',42,y,W-84,'small')
 assert y>45,y
 footer(2);c.save()
if __name__=='__main__':build(prepare())
