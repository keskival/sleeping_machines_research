"""Build local vector documentation with the standard library; no model runtime."""
from html import escape
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLUE, GREEN, ORANGE, GRAY = '#2259a7', '#147d69', '#ad5b16', '#5a6779'
RESULT_CASES = (
    ('timing_only', 'D4 / P32 / U2, timing-only credit', 'curie_language_batched_10M_p32d4_l64_lr004_cmp_s6_20261003T054000Z.json'),
    ('value_credit', 'D4 / P32 / U2, value choice credit', 'curie_language_batched_10M_p32d4_pool2_linear_l64_lr004_cmp_s6_20261003T070000Z.json'),
    ('larger_pool', 'D4 / P32 / U4, value choice credit', 'curie_language_batched_10M_p32d4_pool4_linear_l64_lr004_cmp_s6_20261003T101000Z.json'),
    ('native_90m', 'D4 / P64 / U2, 90M data', 'aws_language_batched_90M_r2_p64d4_linear_l64_lr004_cmp_s6_20261003T103000Z.json'),
)


def evidence_map():
    """Bind a small claim map to completed saved JSON; never import a model."""
    snapshots = []
    for key, label, filename in RESULT_CASES:
        path = ROOT/'experiments/results/language_batched'/filename
        result = json.loads(path.read_text())
        assert result['status'] == 'completed', path
        assert result['eval_segment'] == 256, path
        a, work, protocol = result['args'], result['work'], result['protocol']
        assert protocol['test'] == [95000000,96000000], path
        presentations = result['fitting_chars']
        near_total = work['fit_unit_special_flops_per_char_estimate']*presentations
        assert abs(work['whole_fit_unit_special_flops_estimate']-near_total) <= max(1.,near_total*1e-12)
        snapshots.append(dict(id=key,label=label,result_path=str(path.relative_to(ROOT)),
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            model={k:a[k] for k in ('depth','heads','pool','payload','seed')},
            parameters=result['parameters'],fitting_presentations=presentations,
            test_bpc=result['test_bpc_eval_segment'],test_targets=result['test_targets_eval_segment'],
            available_receivers=a['depth']*a['heads']*a['pool'],
            selected_writes_per_position=a['depth']*a['heads'],
            estimated_whole_fit_tflops=work['whole_fit_unit_special_flops_estimate']/1e12,
            estimated_fit_mflops_per_presentation=work['fit_unit_special_flops_per_char_estimate']/1e6,
            protocol=protocol,work_scope=work['scope'],execution_sources=result['source_sha256']))
    credit_gain = snapshots[0]['test_bpc']-snapshots[1]['test_bpc']
    credit_work_percent = 100*(snapshots[1]['estimated_whole_fit_tflops']/snapshots[0]['estimated_whole_fit_tflops']-1)
    pool_gain = snapshots[1]['test_bpc']-snapshots[2]['test_bpc']
    pool_work_ratio = snapshots[2]['estimated_whole_fit_tflops']/snapshots[1]['estimated_whole_fit_tflops']
    claims = [
        ('Time computes', 'Derived and scoped implementations',
         'Delay/order, analytic transport and reception change the function; physical energy is separate.',
         ['report/model_family_example.md','sleeping_machines/rotating_memory.py','experiments/theory/152_primitives_integration_and_capability_bounds.md']),
        ('Race identity and clock', 'Exact conditional derivation',
         'At fixed entering state exponential race identity is softmax and independent of first time; later trajectories need not stay independent.',
         ['experiments/theory/151_normalized_clock_noise_and_precision_credit.md','sleeping_machines/clock_noise_law.py']),
        ('Value-informed choice credit', 'Completed single-seed language comparison',
         f"{snapshots[0]['test_bpc']:.6f}→{snapshots[1]['test_bpc']:.6f} BPC at T256 for the saved D4/P32/U2 recipe; local message utility is not complete future-write utility.",
         [snapshots[0]['result_path'],snapshots[1]['result_path']]),
        ('Capacity beyond selected activity', 'Completed single-seed language comparison',
         f"U2→U4: {snapshots[1]['test_bpc']:.6f}→{snapshots[2]['test_bpc']:.6f} BPC, 16→32 available receivers and eight writes/position; fitting cost and untied parameters rise.",
         [snapshots[1]['result_path'],snapshots[2]['result_path']]),
        ('Native scale learning', 'Completed restricted text8 evaluation',
         f"90M-trained D4/P64/U2 reaches {snapshots[3]['test_bpc']:.6f} BPC on test[95M:96M], T256; this is not the full official five-million-character test.",
         [snapshots[3]['result_path']]),
        ('Exact attention endpoint', 'Conditional mathematical construction',
         'Delay-coded value/count aggregation is exact over delivered support with its operators/precision; current native winner-only default is a different function.',
         ['experiments/theory/08_vector_memory_and_deep_stacks.md','experiments/theory/152_primitives_integration_and_capability_bounds.md']),
        ('Sparse serving', 'Implemented references; completed-weight FP32 parity pending',
         'Cached selected proposals still score all keys and pay cache/matrix/traffic work; small reference contracts do not establish trained serving quality/energy.',
         ['sleeping_machines/sparse_inference.py','sleeping_machines/prepacked_sparse_inference.py','experiments/theory/148_compact_tied_maps_and_private_state.md']),
        ('Richer reception / silence', 'Reference primitives; integrated learning pending',
         'Windows and silence timeouts define distinct causal programs; membership and future-state utility need explicit credit.',
         ['sleeping_machines/temporal_window.py','sleeping_machines/silence_burst.py','sleeping_machines/race_window.py','report/model_family_example.md']),
        ('Dense / synchronous endpoints', 'Conditional family containment',
         'Full support plus correct aggregation; shared snapshots plus barriers. Function, execution and learning contracts are separate.',
         ['report/model_family_design.md','experiments/theory/152_primitives_integration_and_capability_bounds.md']),
        ('Clock independence', 'Architectural schedule capability; physical benefit unmeasured',
         'No mandatory global periodic tick; local timers/joins and precision remain. Clocked CPU/GPU execution is not measured clockless hardware.',
         ['report/model_family_design.md','report/figures/architecture_shared_world.svg']),
        ('Online learning / TTT', 'Family capability; no completed TTT advantage',
         'The same trainable units can adapt to causal outcomes; state adaptation, parameter updates, asynchronous consistency and batching are distinct.',
         ['report/model_family_design.md','report/figures/architecture_learning.svg']),
        ('Automatic design', 'Dynamic routes implemented; broader structural adaptation aspirational',
         'Operator/allocation changes need task/resource utility and state migration; NAS precedents are attributed.',
         ['report/model_family_design.md','experiments/theory/152_primitives_integration_and_capability_bounds.md']),
        ('One shared multimodal model', 'Integration ambition',
         'Explicit cross-source information paths are required; successes on separate tasks do not establish joint grounding.',
         ['report/model_family_design.md','report/figures/architecture_shared_world.svg']),
    ]
    claim_records = [dict(claim=c,status=s,scope=b,sources=p) for c,s,b,p in claims]
    record = dict(scope='Curated capability map and immutable saved-result bindings; no new benchmark',
                  metric='BPC at T256',work_convention='Saved arithmetic plus unit-counted special functions, estimated from eager full-step traces',
                  snapshots=snapshots,claims=claim_records)
    (ROOT/'report/architecture_evidence.json').write_text(json.dumps(record,indent=2)+'\n')
    lines = ['# Architecture claims and their evidence','',
        'Supporting evidence for the [family definition](model_family_design.md), not its definition or a claim that every branch is integrated.',
        'The [machine-readable bindings](architecture_evidence.json) retain result hashes, actual execution-source hashes, protocols and work conventions. No new model was run.','',
        '## Capability and scope','',
        '| Claim | Evidence level | Precise scope | Basis |','| --- | --- | --- | --- |']
    for claim in claim_records:
        sources = ' · '.join(f'[{Path(p).name}](../{p})' for p in claim['sources'])
        lines.append(f"| {claim['claim']} | {claim['status']} | {claim['scope']} | {sources} |")
    lines += ['', '## Completed native language anchors','',
        'All rows use BPC at a 256-character evaluation window, the same test[95M:96M] region and 999,936 scored targets. D is depth, P per-head width, U receivers per head and H the head count; all have H=2. Results are single-seed. Fit presentations count actual training targets, not distinct corpus bytes.','',
        '| Saved member | Fit presentations (M) | BPC, lower better | Parameters | Available receivers | Writes / position | Estimated whole fit (TFLOPs) | Estimated fit / presentation (MFLOPs) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in snapshots:
        lines.append(f"| [{s['label']}](../{s['result_path']}) | {s['fitting_presentations']/1e6:.6f} | {s['test_bpc']:.6f} | {s['parameters']:,} | {s['available_receivers']} | {s['selected_writes_per_position']} | {s['estimated_whole_fit_tflops']:.6f} | {s['estimated_fit_mflops_per_presentation']:.6f} |")
    lines += ['',
        f'The first three rows share the declared 10M/one-pass recipe. Value credit lowers BPC by {credit_gain:.6f} for about {credit_work_percent:.2f}% extra estimated fitting work. The larger pool then lowers BPC by {pool_gain:.6f} while increasing fitting work by {pool_work_ratio:.3f}×; it also has more untied parameters. The 90M row changes data and width and is scale evidence, not an isolated pool/credit comparison.','',
        'Work uses the saved tracer convention: arithmetic plus unit-counted special functions, eager full fitting steps extrapolated across presentations. Evaluation, compilation/setup, wall time, memory traffic and energy are separate. These numbers are not hardware joules or guaranteed end-to-end savings. Dense controls and inference comparisons retain their existing protocol/accounting in [REPORT.md](../REPORT.md); this table makes no cross-protocol supremacy claim.','',
        'Some saved metadata calls segment credit “exact BPTT within it”. That describes differentiation through the retained factual continuous graph; it does not certify an exact expected gradient over all discrete persistent-write alternatives. The route teacher scope is stated above. Historical execution-source hashes are retained as historical bindings, not assertions that current kernels have the same hashes.','',
        'Target-dependent E63/E79 archives are excluded. Derived containment, implemented references, completed scoped comparisons and aspirations are different evidence levels.']
    (ROOT/'report/architecture_evidence.md').write_text('\n'.join(lines)+'\n')


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


def reception_example():
    d = Diagram('Reception is an operator: same arrivals, different computations', 850)
    d.label(34,76,'Worked scalar example; H=1, query cutoff q=5, arrival-before-timer ties, no damping or trained parameters.',15,GRAY)
    for i,(name,time,value) in enumerate((('A','0.0','1'),('B','0.8','−2'),('C','1.6','3'),('D','4.0','4'))):
        d.box(35+285*i,120,260,95,f'Message {name}',[f'time {time}; content {value}'])
    rows = [
        ('First arrival', 'Accept A', 'emit 1 at time 0.0', 'One delivery selects the continuation.'),
        ('Fixed window from first arrival', 'Accept A + B', 'emit −1 at time 1.0', 'More evidence meets before a fixed local deadline.'),
        ('Silence timeout / popcorn', 'Accept A + B + C', 'emit 2 at time 2.6', 'Every admitted arrival renews the local deadline.'),
        ('All observed arrivals at query', 'Accept A + B + C + D', 'emit 6 at time 5.0', 'Full support within the declared observed prefix.')]
    for i,(title,members,out,reason) in enumerate(rows):
        y=265+i*118
        d.box(35,y,410,65,title,(),GREEN)
        d.box(485,y,275,65,members,(),GREEN)
        d.box(800,y,365,65,out)
        d.arrow([(445,y+32),(485,y+32)],GREEN)
        d.arrow([(760,y+32),(800,y+32)])
        d.label(35,y+93,reason,15,GRAY)
    d.label(35,776,'At q=2, popcorn has pending state 2 and deadline 2.6. A query/EOF does not silently emit early.',16,ORANGE,True)
    d.label(35,815,'A later layer can transform content by elapsed time; learning must credit membership and future-state effects.',15,GRAY)
    return d.finish()


def composition():
    d = Diagram('The whole family: local choices become system capabilities through contracts', 1030)
    d.label(34,76,'One causal content/time/address/state interface; different compatible programs and mixtures in each region.',15,GRAY)
    d.box(35,115,1130,112,'Keep three scales in view',[
        'Family envelope: admissible choices → member: specified operators/contracts → target: integrated temporal selection',
        'Common review axes: computational power + total work; represented distinctions; useful learning.',
        'Implemented branches illustrate choices. Their union is not a completed trained system.'])
    rows = [
        ('Atoms → units', 'Choose evidence + flow + writes + reception + output + credit.', 'Preserve content, timing, state and the relevant event history.'),
        ('Modules → layers → stacks', 'Compose access, deliveries, mixing, joins, depth and recurrence.', 'Preserve required information paths and usable credit through boundaries.'),
        ('Compositions → complete system', 'Combine statistics, retrieval, rich islands, online learning and serving.', 'Specify shared state, causal queries, versions, termination and full work.')]
    for i,(title,choices,contract) in enumerate(rows):
        y=275+165*i
        d.box(35,y,330,110,title,(),GREEN)
        d.box(410,y,755,110,'Design choices + composition contract',[choices,contract],GREEN)
        d.arrow([(365,y+55),(410,y+55)],GREEN)
        if i<2: d.arrow([(200,y+110),(200,y+165)],GREEN)
    d.label(35,800,'Adaptive replacement or growth needs a declared preservation contract:',19,ORANGE,True)
    for i,(title,lines) in enumerate((
        ('Forward continuation',['Future outputs / emissions / state', 'Keep clocks and pending deadlines.']),
        ('Learning transition',['Credit + parameter / optimizer state', 'Equal outputs can learn differently.']),
        ('Resource refinement',['Work / memory / traffic / latency', 'Ignored branches may still cost work.']))):
        d.box(35+385*i,835,360,110,title,lines,ORANGE)
    d.label(35,985,'Expand the limiting region; retain economical regions. Test the integrated consequence rather than counting features.',15,GRAY)
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
    d = Diagram('Native receiver: selection, message, private state and time', 820)
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
    d = Diagram('A native layer, stacked model and source-local recurrence', 1020)
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
    d = Diagram('Execution structures: representative model families', 850)
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
    d = Diagram('Implemented branches: shared toolbox, distinct constructions', 1160)
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
    names = ('Design space','Reception example','Composition contracts','One shared world','Online / batched learning',
             'Implemented branches','Native receiver example','Native layer and stack','Compare families')
    tabs = ''.join(f'<button id="tab-{i}" role="tab" aria-selected="{str(i==0).lower()}" aria-controls="panel-{i}" tabindex="{0 if i==0 else -1}" data-panel="{i}">{name}</button>' for i,name in enumerate(names))
    rows = ''.join(f'<tr><th>{escape(level)}</th><td>{escape(comp)}</td><td>{escape(rep)}</td><td>{escape(train)}</td></tr>' for level,comp,rep,train in LEVELS)
    panels = ''.join(f'<section id="panel-{i}" class="panel" role="tabpanel" aria-labelledby="tab-{i}" tabindex="0" {"hidden" if i else ""}>{svg}</section>' for i,svg in enumerate(figures))
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sleeping Machines: architecture atlas</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f1f5fa;color:#182b43;font:16px/1.5 system-ui,sans-serif}main{max-width:1300px;margin:auto;padding:28px}h1{font-size:32px;margin:8px 0}p{max-width:1040px}a{color:#2259a7}.controls,.card{background:white;border:1px solid #d4dfeb;border-radius:14px;padding:18px;margin:18px 0}.controls{display:flex;flex-wrap:wrap;gap:20px;align-items:center}select,input,button{font:inherit}button{padding:9px 15px;border-radius:8px;border:1px solid #2259a7;background:white;color:#2259a7;cursor:pointer}button[aria-selected=true]{background:#2259a7;color:white}.tabs{display:flex;gap:10px;flex-wrap:wrap}.panel svg{display:block;width:100%;height:auto;border:1px solid #d4dfeb;border-radius:12px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #d4dfeb;vertical-align:top}thead{background:#eaf0f9}.matrix{overflow:auto}th{min-width:150px}td{min-width:225px}.muted{color:#53657d}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:#ecf6f3;padding:12px;min-width:175px;border-radius:10px}.stat strong{display:block;font-size:26px;color:#147d69}.no-credit .learning{display:none}label{display:inline-flex;align-items:center;gap:7px}.scope{border-left:4px solid #ad5b16;padding:12px 18px;background:#fff7ed}input[type=number]{width:65px}footer{margin:25px 0;color:#53657d}[hidden]{display:none!important}
</style><main><h1>What the model computes, stores and learns</h1>
<p>Sleeping Machines is a family of stateful event-processing networks. Content, time, routing and persistent evidence jointly determine computation. Choose compatible operators per unit, module and layer; expand a limiting region while retaining economical computation elsewhere.</p>
<p><a href="model_family_overview.md">Start with the overview</a> · <a href="model_family_design.md">Family definition and choices</a> · <a href="model_family_composition.md">How capabilities compose</a> · <a href="architecture_evidence.md">Claim / evidence map</a> · <a href="architecture_review.md">Detailed review</a></p>
<div class="tabs" role="tablist" aria-label="Architecture diagrams">''' + tabs + '''</div>
<div class="controls"><label><input id="credit" type="checkbox" checked>Show learning signals</label><span class="muted">Orange credit arrows do not change hard forward values.</span></div>
''' + panels + '''
<div class="card"><h2>Try a reception program</h2><p>Same four messages, different first-group computations. The scalar state sums accepted content. This is a causal semantic demonstration with fixed parameters, not an integrated trained layer. Read the <a href="model_family_example.md">worked example</a>.</p><div class="controls"><label>Policy <select id="rx-policy"><option value="winner">First arrival</option><option value="window">Window from first arrival</option><option value="popcorn" selected>Silence timeout / popcorn</option><option value="all">All arrivals at query</option></select></label><label>Interval H <input id="rx-gap" type="number" min="0.1" max="5" step="0.1" value="1"></label><label>C arrival <input id="rx-time" type="number" min="0" max="6" step="0.1" value="1.6"></label><label>Query cutoff q <input id="rx-cutoff" type="number" min="0" max="8" step="0.1" value="5"></label></div><p id="rx-arrivals"></p><div class="stats"><div class="stat"><strong id="rx-members"></strong>First group accepted</div><div class="stat"><strong id="rx-state"></strong>Integrated content</div><div class="stat"><strong id="rx-emission"></strong>Emission time</div></div><p id="rx-note" class="scope" aria-live="polite"></p><p class="muted">Arrival-before-timer ties. Only the first group is shown; later inputs may start another group. Querying does not flush an unfinished silence timeout. H affects window/popcorn; first-arrival and all-at-query use their own stopping rule.</p></div>
<div class="card"><h2>Native-stack example: capacity and activity</h2><p>One source, one input in the winner-per-head native branch. Change dimensions to see structural counts; other reception programs need their own accounting. These are not measured FLOPs, memory bytes or speedups.</p><div class="controls"><label>Layers D <input id="depth" type="number" min="1" max="64" value="4"></label><label>Heads H <input id="heads" type="number" min="1" max="16" value="2"></label><label>Pool U <input id="pool" type="number" min="1" max="1024" value="4"></label><label>Width P <input id="width" type="number" min="2" max="2048" step="2" value="32"></label><label>Proposal program <select id="mode"><option value="dense">Standard training: all candidates</option><option value="sampled">Sampled credit: winner + one alternative</option><option value="inference">Cached winner-only inference</option></select></label></div><div class="stats"><div class="stat"><strong id="memories"></strong>Available memories</div><div class="stat"><strong id="scores"></strong>Keys scored</div><div class="stat"><strong id="writes"></strong>Selected commits</div><div class="stat"><strong id="proposals"></strong>Proposals evaluated</div><div class="stat"><strong id="state"></strong>Memory vector scalars</div></div><p id="mode-note" class="scope" aria-live="polite"></p><p class="muted">Selected work is one part of total cost. All-key reads, matrix setup, queries/channel mixing, optimizer, masks, candidate search, rollout/replay, traffic and energy require their own accounting.</p></div>
<div class="card"><h2>The same review axes at every level</h2><div class="matrix"><table><thead><tr><th>Level</th><th>Computational capacity / work</th><th>Representational capacity</th><th>Trainability</th></tr></thead><tbody>''' + rows + '''</tbody></table></div><p><a href="model_family_inventory.md">Every family branch and its mechanism coverage</a> · <a href="architecture_source_inventory.md">Complete source-module navigation</a></p></div>
<p class="scope">Evidence boundary: completed model scores are not a universal capability proof. Learned reception windows/popcorn, complete future-write credit, shared multimodal grounding and physical clockless energy remain incomplete. Cached trained FP32 parity is pending. No model runs in this page.</p>
<footer>Private local documentation · 4 October 2026 · Self-contained vector diagrams and JavaScript; no external assets, requests or publication.</footer></main>
<script>
const get=id=>document.getElementById(id);
const tabs=Array.from(document.querySelectorAll('[data-panel]'));
function selectTab(button){tabs.forEach(b=>{b.setAttribute('aria-selected',String(b===button));b.setAttribute('tabindex',b===button?'0':'-1')});document.querySelectorAll('.panel').forEach((p,i)=>p.hidden=String(i)!==button.dataset.panel)}
tabs.forEach((button,i)=>{button.addEventListener('click',()=>selectTab(button));button.addEventListener('keydown',e=>{const next=e.key==='ArrowRight'?(i+1)%tabs.length:e.key==='ArrowLeft'?(i+tabs.length-1)%tabs.length:e.key==='Home'?0:e.key==='End'?tabs.length-1:null;if(next!==null){e.preventDefault();selectTab(tabs[next]);tabs[next].focus()}})});
get('credit').addEventListener('change',()=>document.body.classList.toggle('no-credit',!get('credit').checked));
function invalidDimensions(note){['memories','scores','writes','proposals','state'].forEach(id=>get(id).textContent='—');get('mode-note').textContent=note}
function update(){const ids=['depth','heads','pool','width'];if(ids.some(id=>get(id).value===''||!Number.isInteger(Number(get(id).value)))){invalidDimensions('Use dimensions within the stated positive bounds.');return}const [D,H,U,P]=ids.map(id=>Number(get(id).value));if(P%2){invalidDimensions('The implemented paired rotation modes require an even per-head width.');return}if(ids.some(id=>!get(id).checkValidity())){invalidDimensions('Use dimensions within the stated positive bounds.');return}const selected=D*H,mode=get('mode').value;get('memories').textContent=selected*U;get('scores').textContent=selected*U;get('writes').textContent=selected;get('proposals').textContent=selected*(mode==='dense'?U:mode==='sampled'?Math.min(U,2):1);get('state').textContent=selected*U*P;get('mode-note').textContent=mode==='dense'?'Main integrated training path: losing proposals are computed for local value credit; only winning memories are committed.':mode==='sampled'?'Implemented alternative training path: unbiased for the local linear teacher at fixed cotangent, not generally for complete future-route risk. Extra variance and all-key/optimizer work remain.':'Optimized cached inference path: selected proposals after scoring all keys. Completed-weight FP32 winner/state/cache/quality parity and practical serving measurements remain pending.';}
['depth','heads','pool','width','mode'].forEach(id=>get(id).addEventListener('input',update));update();
function traceReception(policy,H,cTime,q){const events=[{name:'A',time:0,value:1},{name:'B',time:.8,value:-2},{name:'C',time:cTime,value:3},{name:'D',time:4,value:4}].sort((a,b)=>a.time-b.time||a.name.localeCompare(b.name));const visible=events.filter(e=>e.time<=q);let accepted=[],deadline=null;if(visible.length){if(policy==='winner'){accepted=[visible[0]];deadline=visible[0].time}else if(policy==='all'){accepted=visible;deadline=q}else{deadline=visible[0].time+H;for(const e of visible){if(e.time>deadline)break;accepted.push(e);if(policy==='popcorn')deadline=e.time+H}}}const emitted=deadline!==null&&deadline<=q;return {events,accepted,state:accepted.reduce((sum,e)=>sum+e.value,0),deadline,emitted,emission:emitted?deadline:null}}
function reception(){const ids=['rx-gap','rx-time','rx-cutoff'];if(ids.some(id=>get(id).value===''||!get(id).checkValidity()||!Number.isFinite(Number(get(id).value)))){['rx-members','rx-state','rx-emission'].forEach(id=>get(id).textContent='—');get('rx-arrivals').textContent='';get('rx-note').textContent='Use finite timing values within the stated bounds.';return}const [H,cTime,q]=ids.map(id=>Number(get(id).value));const r=traceReception(get('rx-policy').value,H,cTime,q);get('rx-arrivals').textContent=r.events.map(e=>e.name+' @ '+e.time.toFixed(1)+' = '+e.value+(e.time>q?' (after query)':'')).join(' · ');get('rx-members').textContent=r.accepted.map(e=>e.name).join(', ')||'None';get('rx-state').textContent=r.state;get('rx-emission').textContent=r.emission===null?'Not emitted':r.emission.toFixed(2);get('rx-note').textContent=r.emitted?'First group emitted. Its content and emission time can both change the next layer.':r.deadline===null?'No event observed: no active reception.':'Pending at query: deadline '+r.deadline.toFixed(2)+'. Its accumulated state is not an emitted message.'}
['rx-policy','rx-gap','rx-time','rx-cutoff'].forEach(id=>get(id).addEventListener('input',reception));reception();
</script></html>'''


def main():
    figures = [design_space(), reception_example(), composition(), shared_world(), learning(), family(), receiver(), stack(), comparisons()]
    for name, svg in zip(('design_space', 'reception', 'composition', 'shared_world', 'learning', 'family', 'receiver', 'stack', 'comparisons'), figures):
        path = ROOT/'report/figures'/f'architecture_{name}.svg'
        path.write_text(svg+'\n')
    (ROOT/'report/architecture_atlas.html').write_text(html_page(figures)+'\n')
    evidence_map()
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
    print(f'Built nine SVG diagrams, reception explorer, evidence map and {len(modules)}-module AST inventory; no numerical imports.')


if __name__ == '__main__':
    main()
