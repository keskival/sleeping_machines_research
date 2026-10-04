"""Build local vector documentation with the standard library; no model runtime."""
from html import escape
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLUE, GREEN, ORANGE, GRAY = '#2259a7', '#147d69', '#ad5b16', '#5a6779'


class Diagram:
    def __init__(self, title, height):
        self.height = height
        self.marker_id = 'arrow-'+hashlib.sha256(title.encode()).hexdigest()[:8]
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" role="img" aria-label="{escape(title, quote=True)}">',
                      f'<defs><marker id="{self.marker_id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="context-stroke"/></marker></defs>',
                      f'<rect width="1200" height="{height}" fill="#fafcff"/>']
        self.label(34, 44, title, 26, '#162235', True)

    def label(self, x, y, text, size=16, color='#25354b', bold=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{escape(text)}</text>')

    def box(self, x, y, w, h, title, lines=(), color=BLUE, css=''):
        self.parts.append(f'<g class="{css}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="white" stroke="{color}" stroke-width="2"/>')
        title_size = min(17, (w-28)/max(1,len(title))/.57)
        self.label(x+14, y+27, title, title_size, color, True)
        for i, line in enumerate(lines):
            self.label(x+14, y+53+23*i, line, 15)
        self.parts.append('</g>')

    def arrow(self, points, color=BLUE, dashed=False, css=''):
        coords = ' '.join(f'{x},{y}' for x,y in points)
        self.parts.append(f'<polyline class="{css}" points="{coords}" fill="none" stroke="{color}" stroke-width="2.3" stroke-dasharray="{6 if dashed else 0}" marker-end="url(#{self.marker_id})"/>')

    def finish(self):
        return '\n'.join(self.parts+['</svg>'])


def design_space():
    d = Diagram('Sleeping Machines: one family, many compatible design choices', 1240)
    d.label(34,76,'Define a causal content/time/state interface; choose compatible operators at each level.',16,GRAY)
    d.box(35,115,1130,112,'Common contract: observed event → local state program → timed communication → causal query',[
        'Content, computational time, routes and stored evidence affect one another.',
        'No required network-wide periodic tick. Local timers, joins and causal ordering still matter.',
        'Synchronous barriers and full participation are allowed endpoints, chosen where useful.'])
    rows = [
        ('Unit: what can persist and transform?', 'Vectors / modes / gated programs', 'Counts / protected facts / history', 'Delays / clock responses / keys'),
        ('Module: which evidence participates?', 'One winner / several deliveries', 'Window / silence / exact pooling', 'Observed / learned / indexed access'),
        ('Layer: how do units interact?', 'Private heads / shared receivers', 'Timed join / learned channel mix', 'Lean stage / dense local island'),
        ('Composition: which capabilities meet?', 'Depth / recurrence / optional paths', 'State + statistics + retrieval', 'Dense and sparse modalities together')]
    for i,(title,a,b,c) in enumerate(rows):
        y=270+i*153
        d.label(35,y,title,19,GREEN,True)
        for j,label in enumerate((a,b,c)):
            d.box(35+385*j,y+20,360,65,label,(),GREEN)
        if i<3: d.arrow([(600,y+85),(600,y+141)],GREEN)
    d.box(35,893,550,112,'Review each choice on three axes',[
        'Computation: access, work, traffic and latency',
        'Representation: retained distinctions and useful functions',
        'Trainability: credit, support, exposure and horizon'],ORANGE)
    d.box(615,893,550,112,'Selectively expand the limiting region',[
        'Match/include rich reference operators when necessary.',
        'Keep efficient stages where their quality is sufficient.',
        'Exact containment needs operators and numerical contracts.'],BLUE)
    d.label(35,1045,'Theory and evidence guide choices; the inventory records implementations rather than defining the family.',16,GRAY)
    d.box(35,1090,1130,105,'Further ambition: the model learns some of these design choices from data',[
        'Learned programs, fan-in, optional stages and state allocation can morph regions within the family.',
        'Dynamic routes exist; broader structural learning needs utility, state migration and full cost contracts.'],ORANGE)
    d.label(35,1220,'Static design is a fixed-policy case. Matching a reference function, learning it and beating its cost are separate.',16,GRAY)
    return d.finish()


def shared_world():
    d = Diagram('One-model aspiration: mixed arrivals, shared state, local computation', 1030)
    d.label(34,76,'Input density, computation density, arrival cadence and hardware clocking are independent axes.',16,GRAY)
    inputs = [('Dense periodic frames',['regular arrival; many pixels']),('Language / queries',['sequence order; declared time']),('Sparse sensor events',['irregular source/time/content'])]
    for i,(title,lines) in enumerate(inputs):
        x=35+385*i
        d.box(x,120,360,100,title,lines)
        d.box(x,268,360,100,'Causal modality adapter',['preserve relevant evidence', 'align time/address/reference semantics'])
        d.arrow([(x+180,220),(x+180,268)])
        d.arrow([(x+180,368),(x+180,407),(600,407),(600,450)])
    d.box(190,450,820,135,'Explicit shared information paths + persistent world state',[
        'Private facts / learned vectors / causal statistics / historical evidence',
        'Cross-source receivers or retrieval let modalities inform the same query.',
        'Shared parameters alone do not share facts.'],GREEN)
    d.box(35,640,550,122,'Selective temporal region',[
        'Addressed state programs; learned clocks and routes',
        'Small messages; local timers, joins and deadlines',
        'No required global periodic sweep'],GREEN)
    d.box(615,640,550,122,'Rich interaction where needed',[
        'Multiple deliveries, exact aggregation or dense island',
        'Wider state/message only at a diagnosed bottleneck',
        'Pay its full key/value, learning and traffic cost'])
    d.arrow([(420,585),(310,640)],GREEN)
    d.arrow([(780,585),(890,640)])
    d.arrow([(585,700),(615,700)],GREEN)
    d.box(190,820,820,100,'Queries, predictions and actions',[
        'Regular readouts or event-triggered decisions over shared causal evidence',
        'Preserve state across arrivals; supervise useful factual and alternative consequences'])
    d.arrow([(310,762),(310,790),(600,790),(600,820)],GREEN)
    d.arrow([(890,762),(890,790),(600,790),(600,820)])
    d.label(35,962,'Full joint multimodal training is pending. Current emulators and optimizer use clocked/coordinated execution.',15,GRAY)
    d.label(35,994,'Clock independence is architectural; measured physical clockless energy and fully local online learning are open.',15,GRAY)
    return d.finish()


def learning():
    d = Diagram('One structure: stateful prediction, online learning and batching', 970)
    d.label(34,76,'TTT and online updates use the same trainable units, memories, clocks and routes; define the causal objective.',16,GRAY)
    d.box(35,130,290,120,'Observed input event',[
        'source + content + time',
        'available self-supervised evidence',
        'independent stream or batch lane'])
    d.box(395,130,370,155,'Same stateful model',[
        'private state + shared/local maps',
        'selected temporal programs',
        'keys, values, routes and clocks',
        'predictions and persistent writes'],GREEN)
    d.box(840,130,325,155,'Prediction / action',[
        'causal query and output',
        'record score before label update',
        'continue inference / retain state',
        'later observed outcome'])
    d.arrow([(325,192),(395,192)])
    d.arrow([(765,192),(840,192)])
    d.box(840,365,325,120,'Loss / learning event',[
        'observed target or self-supervision',
        'no unavailable future label',
        'declared online or inner objective'],ORANGE,'learning')
    d.box(395,365,370,155,'Credit on the model structure',[
        'actual content / state / time paths',
        'route alternatives and eligibility',
        'bounded extra proposals or replay',
        'future-write utility still incomplete'],ORANGE,'learning')
    d.box(35,365,290,155,'Parameter update',[
        'local ownership or shared rules',
        'declared optimizer / learning rate',
        'retain needed versions / credit',
        'affect subsequent computation'],ORANGE,'learning')
    d.arrow([(1000,285),(1000,365)],ORANGE,True,'learning')
    d.arrow([(840,425),(765,425)],ORANGE,True,'learning')
    d.arrow([(395,425),(325,425)],ORANGE,True,'learning')
    d.arrow([(180,365),(180,310),(500,310),(500,285)],ORANGE,True,'learning')
    d.box(35,595,550,170,'Streamed / local update policy',[
        'Credit may arrive later than the prediction event.',
        'Select eligible units and useful alternative support.',
        'Shared keys/maps and stale versions need coordination.',
        'Sparse forward activity is not total training sparsity.',
        'Fully asynchronous physical training is pending.'],ORANGE)
    d.box(615,595,550,170,'Batching / compiled execution',[
        'Independent lanes retain independent state.',
        'Group compatible operations into dense kernels.',
        'Masks / resets / event order preserve lane semantics.',
        'Batch and sequential updates can differ numerically.',
        'Current clip / Adam steps remain coordinated.'])
    d.label(35,825,'State adaptation ≠ parameter learning. TTT is a supported family capability, not a measured project TTT win.',16,GRAY)
    d.label(35,860,'Charge selected inference + candidate discovery + alternative credit + optimizer + retained state + traffic.',16,GRAY)
    d.label(35,895,'Learning may be online, batched or mixed without replacing the model by a separate dense predictor.',16,BLUE,True)
    d.label(35,934,'Reference loop: native integrated units. This figure specifies admissible learning policies, not an on-chip implementation.',14,GRAY)
    return d.finish()


def receiver():
    d = Diagram('1. Receiver unit: selection, message, private state and time', 820)
    d.label(34, 76, 'Logical dependencies of the current native program; one race chooses among U such receivers.', 16, GRAY)
    d.box(35, 140, 215, 115, 'Incoming event', ['content x: P numbers', 'arrival t; source address'])
    d.box(35, 310, 215, 145, 'Private stored state', ['memory m; last write t_prev', 'seen flag; learned key', 'decay/rotation parameters'], GREEN)
    d.box(305, 135, 270, 130, 'Key / compatibility branch', ['query from layer content', 'key + key_read(stored m)', 'score: dot product + bias'])
    d.box(640, 135, 240, 130, 'Head race', ['rate = exp(clamped score)', 'first time + winner identity', 'all U keys are scored'])
    d.box(930, 135, 235, 130, 'Outgoing time', ['bounded delay(first time)', 't_out = t + delay', 'time changes later content'])
    d.box(305, 330, 270, 160, 'Value / state proposal', ['forget and write gates from x', 'rotate + damp m over age', 'add write × Input(x)', 'gated residual value v'])
    d.box(640, 330, 240, 160, 'Hard selected commit', ['winner: store proposed m', 'store incoming t; set seen', 'losers retain old state', 'deliver one value per head'], GREEN)
    d.box(930, 330, 235, 160, 'Outgoing content', ['selected P-vector v', 'other heads align to join', 'next layer + future context'])
    d.box(305, 565, 575, 110, 'Training credit: three distinct consequences', ['delivered-value alternatives; common computational time', 'future memory/write utility needs additional credit'], ORANGE, 'learning')
    d.arrow([(250,185),(305,185)])
    d.arrow([(250,350),(278,350),(278,240),(305,240)], GREEN)
    d.arrow([(250,225),(275,225),(275,355),(305,355)])
    d.arrow([(250,395),(305,395)], GREEN)
    d.arrow([(575,185),(640,185)])
    d.arrow([(880,185),(930,185)])
    d.arrow([(760,265),(760,330)])
    d.arrow([(575,410),(640,410)])
    d.arrow([(880,410),(930,410)])
    d.arrow([(680,490),(680,525),(140,525),(140,455)], GREEN)
    d.label(34, 553, 'Only selected state is committed', 15, GREEN, True)
    d.arrow([(1045,490),(1045,620),(880,620)], ORANGE, True, 'learning')
    d.arrow([(450,565),(450,490)], ORANGE, True, 'learning')
    d.arrow([(735,565),(900,565),(900,285),(755,285),(755,265)], ORANGE, True, 'learning')
    d.label(34, 722, 'Blue: event computation    Green: persistent state    Orange dashed: learning-only signals', 16, GRAY)
    d.label(34, 754, 'Standard training evaluates U proposals; cached winner-only inference evaluates one selected proposal.', 16)
    d.label(34, 784, 'Trained FP32 sparse parity/cost pending. Stored-key read precedes input-conditioned age evolution.', 16, GRAY)
    return d.finish()


def stack():
    d = Diagram('2. A native layer, stacked model and source-local recurrence', 1020)
    d.label(34, 76, 'Illustration: H=2 heads, U=3 receivers/head. Only one receiver wins in each head.', 16, GRAY)
    d.box(34, 130, 245, 135, 'Input and previous context', ['observed content/time', 'wait for same-source readiness', 'gate aligned previous context'])
    d.box(325, 130, 330, 135, 'Channel mix + head queries', ['mix the H×P incoming vector', 'each query sees mixed channels', 'split content into P per head'])
    d.arrow([(279,195),(325,195)])
    for head, x in enumerate((125,690)):
        d.box(x, 330, 400, 215, f'Head {head+1}: private receiver pool', ['U keys scored; U candidate clocks', 'receiver 0: private memory', 'receiver 1: private memory [winner]', 'receiver 2: private memory', 'one selected value + arrival'], GREEN)
        d.arrow([(490,265),(490,298),(x+200,298),(x+200,330)])
        d.arrow([(x+200,545),(x+200,581),(530,581),(530,605)])
    d.box(325, 605, 415, 125, 'Temporal join: latest head arrival', ['transport earlier values by their elapsed age', 'concatenate aligned head vectors', 'next layer receives content AND time'])
    d.box(835, 605, 330, 125, 'Depth D: repeat this layer', ['new private receiver pool at each depth', 'one selected continuation per head', 'credit and state paths span the stack'])
    d.arrow([(740,668),(835,668)])
    d.box(835, 805, 330, 125, 'Task readout / final context', ['language logits or regression values', 'save final head vectors and times', 'reuse only for the same source'])
    d.arrow([(1000,730),(1000,805)])
    d.arrow([(835,869),(58,869),(58,285),(100,285),(100,265)], GREEN)
    d.box(125, 785, 565, 115, 'Execution boundary', ['Episode lanes: independent state, batched together', 'one-source task recipes reset or carry state as declared', 'local head joins do not imply a physical asynchronous chip'], GRAY)
    d.label(34, 960, 'Available: D×H×U memories   |   Scored: D×H×U keys   |   Selected: D×H writes/input', 17, GREEN, True)
    d.label(34, 990, 'Map sharing can preserve private state. Cross-source grounding needs an explicit shared information path.', 16, GRAY)
    return d.finish()


def comparisons():
    d = Diagram('3. Execution structures: representative model families', 850)
    d.label(34, 76, 'Structural comparison, not a quality or energy ranking. Mature implementations remain required controls.', 16, GRAY)
    rows = [
        ('Dense softmax attention', 'query + cached keys/values', 'all valid key comparisons', 'weighted value aggregation', 'FFN + residual; next layer', 'Every valid value can contribute. FlashAttention improves IO, not the mathematical attention function.'),
        ('Recurrent / selective SSM', 'input + carried state', 'input-conditioned gates/flow', 'update configured state', 'readout; carry to next input', 'Recurrence compresses history. State update structure and parallel scan implementation matter.'),
        ('Sparse expert layer / MoE', 'input representation', 'router + candidate scores', 'selected expert computations', 'combine; residual/next layer', 'Capacity can exceed selected experts. Router, communication, learning and balance remain paid.'),
        ('Native temporal receiver stack', 'content/time + private state', 'key race: identity AND clock', 'selected state/value update', 'timed head join; next layer', 'Time transforms content; writes change future selection. Standard training also computes losing values.'),
        ('Addressed count / backoff memory', 'context address + symbol', 'lookup statistics + escape', 'predict before observing target', 'increment addressed counts', 'Counts store evidence directly. Learned pooling is an extension, not an inherent property of counting.')]
    for i,(title,a,b,c,e,foot) in enumerate(rows):
        y = 112+i*145
        d.label(34,y,title,18,GREEN if i==3 else BLUE,True)
        for j,label in enumerate((a,b,c,e)):
            x = 34+286*j
            d.box(x,y+15,266,60,label,(),GREEN if i==3 else BLUE)
            if j<3: d.arrow([(x+266,y+45),(x+286,y+45)],GREEN if i==3 else BLUE)
        d.label(34,y+102,foot,14,GRAY)
    return d.finish()


def family():
    d = Diagram('0. The whole family: shared toolbox, distinct constructions', 1160)
    d.label(34,76,'Compositions reuse primitives; no single completed model combines and validates every branch below.',16,GRAY)
    d.box(35,115,1130,105,'Observed events: address + arrival time + content',['Tokens, sensor marks, counts and query/action requests need declared causal adapters.',
              'Input precision and packetization set the available information before any layer learns.'])
    primitives = [ ('Timing logic',['delay; first-of / latest-of','coincidence; hold; veto']),
                   ('Temporal state',['exponential flow / rotation','gated writes; affine scans']),
                   ('Choice and credit',['key race; clock + identity','alternative utilities']),
                   ('Evidence memory',['counts; escape; copy','historical keys / values']),
                   ('Reception / readout',['window; silence deadline','affine / joint query']) ]
    for i,(title,lines) in enumerate(primitives):
        x=35+228*i
        d.box(x,280,218,112,title,lines)
        d.arrow([(600,220),(600,248),(x+109,248),(x+109,280)])
    d.label(35,435,'Implemented family branches (different learning and validation scopes)',19,GREEN,True)
    branches = [ ('Timing logic + statistics',['order / phase / finite rules','context counts, copy and backoff','event topology or addressed tables']),
                 ('Carriers + modal encoders',['shared event / SSM-like state','selective temporal carriers','dense local maps at supplied events']),
                 ('Native receiver stacks',['private pools + hard state commits','parallel heads + temporal joins','sparse selected work; paid candidates']),
                 ('Historical / outcome memory',['episodic key/value bank and indexing','protected outcomes; delayed taps','bilinear queries; historical credit']),
                 ('Count / learned-memory hybrids',['native stack + statistical cascade','learned keys + count-valued receivers','mixture or sampled/top write variants']),
                 ('Attention / richer reception',['Transformer import and race delivery','exact delay-coded attention identity','windows / popcorn: partial integration']) ]
    for i,(title,lines) in enumerate(branches):
        x=35+385*(i%3); y=470+195*(i//3)
        d.box(x,y,360,150,title,lines,GREEN)
    d.box(35,875,1130,170,'System composition: choose and connect the required modules',[
        'A stack + evidence memory + reception policy + query/readout + learning algorithm.',
        'Learned content changes time/routes; chosen time/routes change content and persistent future evidence.',
        'Inference, counterfactual learning, optimizer, scheduling, storage and communication have separate costs.',
        'Shared rules with private facts can scale useful state; actual cross-source grounding needs an information path.' ],ORANGE)
    for x in (215,600,985): d.arrow([(x,815),(x,850),(600,850),(600,875)],GREEN)
    d.label(35,1100,'Review every level on: computational capacity/work | representational capacity | trainability.',18,BLUE,True)
    d.label(35,1132,'Known primitives are attributed. The research contribution is their construction and tested consequences.',16,GRAY)
    return d.finish()


LEVELS = [
 ('Event / observation', 'Content, timestamp, observed address; finite numerical precision.', 'What the adapter preserves: identity, order, elapsed time and content.', 'Adapter information loss cannot be repaired by downstream credit.'),
 ('Temporal algebra', 'Min/first-of, sums of delays, exponential flow, coincidence and deadlines.', 'Functions of interval, order, phase and selected evidence.', 'Smooth-history derivatives and event-boundary credit have distinct contracts.'),
 ('Receiver unit', 'Score/read, conditional flow/write, gated residual; small local maps.', 'Private vector state plus keys, timescales and temporal response.', 'Factual content/time derivatives; optional routes need utility credit.'),
 ('Race head / module', 'U candidate scores; one selected state program, or paid multi-reception.', 'Addressed selection, associative access, local timing and alternative programs.', 'Common clock, choice, delivered content and future write are separate consequences.'),
 ('Layer / interaction', 'H heads, cross-channel query/mix, latest-arrival join and transport.', 'Cross-head features and interactions between content and arrival order.', 'Join/transport couples gradients; full sequence influence matters.'),
 ('Deep stack', 'D serial selected transformations with persistent state at each depth.', 'Hierarchical temporal features; conditional composition of stateful programs.', 'Identity conditioning, deep Jacobians, support and future credit horizon.'),
 ('Memory composition', 'Add causal counts, outcome bank or historical KV retrieval.', 'Repeated evidence, long-lived facts, learned pooling and nonlocal binding.', 'Learned writes/retrieval must receive utility; exact count increments need no fitting.'),
 ('Learning / serving system', 'Candidate discovery + proposals + replay + optimizer + communication.', 'Useful capacity depends on precision, storage, access and predictive use.', 'Validated recipes, checkpoints, repeated quality/work and hardware measurements.'),
]


def html_page(figures):
    rows = ''.join(f'<tr><th>{escape(level)}</th><td>{escape(comp)}</td><td>{escape(rep)}</td><td>{escape(train)}</td></tr>' for level,comp,rep,train in LEVELS)
    panels = ''.join(f'<section id="panel-{i}" class="panel" {"hidden" if i else ""}>{svg}</section>' for i,svg in enumerate(figures))
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sleeping Machines: architecture atlas</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f1f5fa;color:#182b43;font:16px/1.5 system-ui,sans-serif}main{max-width:1300px;margin:auto;padding:28px}h1{font-size:32px;margin:8px 0}p{max-width:1040px}a{color:#2259a7}.controls,.card{background:white;border:1px solid #d4dfeb;border-radius:14px;padding:18px;margin:18px 0}.controls{display:flex;flex-wrap:wrap;gap:20px;align-items:center}select,input,button{font:inherit}button{padding:9px 15px;border-radius:8px;border:1px solid #2259a7;background:white;color:#2259a7;cursor:pointer}button[aria-selected=true]{background:#2259a7;color:white}.tabs{display:flex;gap:10px;flex-wrap:wrap}.panel svg{display:block;width:100%;height:auto;border:1px solid #d4dfeb;border-radius:12px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #d4dfeb;vertical-align:top}thead{background:#eaf0f9}.matrix{overflow:auto}th{min-width:150px}td{min-width:225px}.muted{color:#53657d}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:#ecf6f3;padding:12px;min-width:175px;border-radius:10px}.stat strong{display:block;font-size:26px;color:#147d69}.no-credit .learning{display:none}label{display:inline-flex;align-items:center;gap:7px}.scope{border-left:4px solid #ad5b16;padding:12px 18px;background:#fff7ed}input[type=number]{width:65px}footer{margin:25px 0;color:#53657d}[hidden]{display:none!important}
</style><main><h1>What the model computes, stores and learns</h1>
<p>Sleeping Machines is a family of stateful event-processing networks. Content, time, routing and persistent evidence jointly determine computation. Choose compatible operators per unit, module and layer; expand a limiting region while retaining economical computation elsewhere.</p>
<p><a href="model_family_design.md">Family definition and design rationale</a> · <a href="architecture_review.md">Detailed systematic review</a> · <a href="../experiments/theory/152_primitives_integration_and_capability_bounds.md">Derivations and limits</a></p>
<div class="tabs" role="tablist" aria-label="Architecture diagrams"><button role="tab" aria-selected="true" aria-controls="panel-0" data-panel="0">Design space</button><button role="tab" aria-selected="false" aria-controls="panel-1" data-panel="1">One shared world</button><button role="tab" aria-selected="false" aria-controls="panel-2" data-panel="2">Online / batched learning</button><button role="tab" aria-selected="false" aria-controls="panel-3" data-panel="3">Implemented branches</button><button role="tab" aria-selected="false" aria-controls="panel-4" data-panel="4">Native receiver example</button><button role="tab" aria-selected="false" aria-controls="panel-5" data-panel="5">Native layer and stack</button><button role="tab" aria-selected="false" aria-controls="panel-6" data-panel="6">Compare families</button></div>
<div class="controls"><label><input id="credit" type="checkbox" checked>Show learning signals</label><span class="muted">Orange credit arrows do not change hard forward values.</span></div>
''' + panels + '''
<div class="card"><h2>Capacity, candidate work and selected activity</h2><p>One source, one input. Change the dimensions to see structural counts. These are not measured FLOPs, memory bytes or speedups.</p><div class="controls"><label>Layers D <input id="depth" type="number" min="1" max="64" value="4"></label><label>Heads H <input id="heads" type="number" min="1" max="16" value="2"></label><label>Pool U <input id="pool" type="number" min="1" max="1024" value="4"></label><label>Width P <input id="width" type="number" min="2" max="2048" step="2" value="32"></label><label>Proposal program <select id="mode"><option value="dense">Standard training: all candidates</option><option value="sampled">Sampled credit: winner + one alternative</option><option value="inference">Cached winner-only inference</option></select></label></div><div class="stats"><div class="stat"><strong id="memories"></strong>Available memories</div><div class="stat"><strong id="scores"></strong>Keys scored</div><div class="stat"><strong id="writes"></strong>Selected commits</div><div class="stat"><strong id="proposals"></strong>Proposals evaluated</div><div class="stat"><strong id="state"></strong>Memory vector scalars</div></div><p id="mode-note" class="scope" aria-live="polite"></p><p class="muted">Selected work is one part of total cost. All-key reads, matrix setup, queries/channel mixing, optimizer, masks, candidate search, rollout/replay, traffic and energy require their own accounting.</p></div>
<div class="card"><h2>The same review axes at every level</h2><div class="matrix"><table><thead><tr><th>Level</th><th>Computational capacity / work</th><th>Representational capacity</th><th>Trainability</th></tr></thead><tbody>''' + rows + '''</tbody></table></div><p><a href="model_family_inventory.md">Every family branch and its mechanism coverage</a> · <a href="architecture_source_inventory.md">Complete source-module navigation</a></p></div>
<p class="scope">Evidence boundary: completed model scores are not a universal capability proof. Learned reception windows/popcorn, complete future-write credit, shared multimodal grounding and physical clockless energy remain incomplete. Cached trained FP32 parity is pending. No model runs in this page.</p>
<footer>Private local documentation · 4 October 2026 · Self-contained vector diagrams and JavaScript; no external assets, requests or publication.</footer></main>
<script>
const get=id=>document.getElementById(id);
document.querySelectorAll('[data-panel]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-panel]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));document.querySelectorAll('.panel').forEach((p,i)=>p.hidden=String(i)!==button.dataset.panel)}));
get('credit').addEventListener('change',()=>document.body.classList.toggle('no-credit',!get('credit').checked));
function update(){const ids=['depth','heads','pool','width'];if(ids.some(id=>!get(id).checkValidity()||get(id).value===''||!Number.isInteger(Number(get(id).value)))){get('mode-note').textContent='Use dimensions within the stated positive bounds.';return}const [D,H,U,P]=ids.map(id=>Number(get(id).value));if(P%2){get('mode-note').textContent='The implemented paired rotation modes require an even per-head width.';return}const selected=D*H,mode=get('mode').value;get('memories').textContent=selected*U;get('scores').textContent=selected*U;get('writes').textContent=selected;get('proposals').textContent=selected*(mode==='dense'?U:mode==='sampled'?Math.min(U,2):1);get('state').textContent=selected*U*P;get('mode-note').textContent=mode==='dense'?'Main integrated training path: losing proposals are computed for local value credit; only winning memories are committed.':mode==='sampled'?'Implemented alternative training path: unbiased for the local linear teacher at fixed cotangent, not generally for complete future-route risk. Extra variance and all-key/optimizer work remain.':'Optimized cached inference path: selected proposals after scoring all keys. Completed-weight FP32 winner/state/cache/quality parity and practical serving measurements remain pending.';}
['depth','heads','pool','width','mode'].forEach(id=>get(id).addEventListener('input',update));update();
</script></html>'''


def main():
    figures = [design_space(), shared_world(), learning(), family(), receiver(), stack(), comparisons()]
    for name, svg in zip(('design_space', 'shared_world', 'learning', 'family', 'receiver', 'stack', 'comparisons'), figures):
        path = ROOT/'report/figures'/f'architecture_{name}.svg'
        path.write_text(svg+'\n')
    (ROOT/'report/architecture_atlas.html').write_text(html_page(figures)+'\n')
    modules = []
    foundation_names = ('e30_minsky.py','e53_depth3.py','e54_chains.py','e61_race_attention.py',
                        'e120_shared_tasks.py','e173_causal_language.py','e174_aligned_language_baselines.py')
    paths = sorted((ROOT/'sleeping_machines').glob('*.py')) + [ROOT/'experiments'/n for n in foundation_names]
    for path in paths:
        tree = ast.parse(path.read_text())
        classes = [dict(name=n.name,bases=[ast.unparse(b) for b in n.bases],
                        methods=[m.name for m in n.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef))])
                   for n in tree.body if isinstance(n,ast.ClassDef)]
        modules.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            summary=(ast.get_docstring(tree) or '').split('\n')[0],classes=classes,
                            functions=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]))
    (ROOT/'report/architecture_source_inventory.json').write_text(json.dumps(dict(
        scope='AST source navigation; no runtime import, efficacy or compatibility claim',modules=modules),indent=2)+'\n')
    lines=['# Source navigation for the whole family','',
           'AST inventory; file existence and class names do not establish validated capability.',
           'See [semantic family review](model_family_inventory.md) and [source hashes/methods](architecture_source_inventory.json).','',
           '| Module | Classes / top-level functions | Declared source summary |','| --- | --- | --- |']
    for m in modules:
        names=[c['name'] for c in m['classes']] or m['functions']
        lines.append(f"| [{m['path']}](../{m['path']}) | {', '.join(names) or 'Package'} | {m['summary'].replace('|','/')} |")
    (ROOT/'report/architecture_source_inventory.md').write_text('\n'.join(lines)+'\n')
    print(f'Built seven SVG diagrams, interactive atlas and {len(modules)}-module AST inventory; no numerical imports.')


if __name__ == '__main__':
    main()
