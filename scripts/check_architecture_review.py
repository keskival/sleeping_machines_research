"""Check local documentation contracts and scalar identities without ML imports.

These checks are not model, training, browser-rendering or benchmark tests.
"""
import argparse
from decimal import Decimal
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.elements = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.elements.append((tag,a))
        if 'id' in a:
            self.ids.append(a['id'])
        for name in ('href', 'src'):
            if name in a:
                self.links.append(a[name])


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8), (a, b)


def verify():
    records = {}
    svgs = sorted((ROOT/'report/figures').glob('architecture_*.svg'))
    assert len(svgs) == 10
    svgs += [ROOT/'report/figures/model_family_codec.svg']
    for path in svgs:
        tree = ET.fromstring(path.read_text())
        _, _, width, height = map(float, tree.attrib['viewBox'].split())
        for element in tree.iter():
            if element.tag.endswith('rect'):
                x, y = float(element.get('x', 0)), float(element.get('y', 0))
                assert 0 <= x <= width and 0 <= y <= height
                assert x+float(element.get('width', 0)) <= width
                assert y+float(element.get('height', 0)) <= height
            if element.tag.endswith('text'):
                assert 0 <= float(element.get('x', 0)) <= width
                assert 0 <= float(element.get('y', 0)) <= height
    records['svg_xml_and_canvas_bounds'] = len(svgs)

    html_path = ROOT/'report/architecture_atlas.html'
    html = html_path.read_text()
    page = Page()
    page.feed(html)
    assert len(page.ids) == len(set(page.ids))
    assert not any(re.match(r'^(https?:)?//', link) for link in page.links)
    assert all((html_path.parent/link.split('#')[0]).exists()
               for link in page.links if not link.startswith('#'))
    records['html_unique_ids_and_local_links'] = len(page.ids)
    tabs = [a for _,a in page.elements if a.get('role') == 'tab']
    panels = [a for _,a in page.elements if a.get('role') == 'tabpanel']
    assert len(tabs) == len(panels) == 10
    for i,(tab,panel) in enumerate(zip(tabs,panels)):
        assert tab['aria-controls'] == panel['id']
        assert panel['aria-labelledby'] == tab['id']
        assert tab['aria-selected'] == str(i==0).lower()
        assert tab['tabindex'] == ('0' if i==0 else '-1')
        assert ('hidden' in panel) == (i!=0)
    records['tab_and_panel_accessibility_bindings'] = len(tabs)

    docs = [ROOT/'report'/n for n in (
        'model_family_design.md', 'architecture_review.md', 'model_family_inventory.md',
        'architecture_review_checks.md', 'architecture_source_inventory.md',
        'model_family_overview.md','model_family_example.md','model_family_composition.md',
        'model_family_specification.md','model_family_members.md','model_family_opportunities.md','architecture_evidence.md')]
    docs += [ROOT/'experiments/theory/152_primitives_integration_and_capability_bounds.md']
    docs += [ROOT/'investment/VALUATION_RATIONALE.md',ROOT/'investment/INVESTOR_PROOF_PLAN.md',ROOT/'investment/INVESTMENT_CASE.md',ROOT/'investment/PITCH.md']
    links = 0
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if re.match(r'^[a-z]+:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0])
            assert (path.parent/target).exists(), (path, target)
            links += 1
    records['markdown_local_links'] = links
    fields=list('EOGSIPTQLUBX')
    for name,repeats in (('model_family_specification.md',1),('model_family_members.md',3)):
        declared=re.findall(r'^\| ([A-Z])(?=[: ])',(ROOT/'report'/name).read_text(),re.M)
        assert declared==fields*repeats,(name,declared)
    records['complete_twelve_field_member_specifications']=3
    inventory = json.loads((ROOT/'report/architecture_source_inventory.json').read_text())
    for item in inventory['modules']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256']
    records['source_inventory_hashes'] = len(inventory['modules'])
    evidence = json.loads((ROOT/'report/architecture_evidence.json').read_text())
    snapshots = evidence['snapshots']
    assert len(snapshots) == 4 and len({s['id'] for s in snapshots}) == 4
    for s in snapshots:
        path = ROOT/s['result_path']
        assert 'invalid_protocol' not in str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == s['result_sha256']
        result = json.loads(path.read_text())
        assert result['status'] == 'completed' and result['eval_segment'] == 256
        near(s['test_bpc'],result['test_bpc_eval_segment'])
        assert s['protocol'] == result['protocol']
        assert s['execution_sources'] == result['source_sha256']
        assert s['parameters'] == result['parameters']
        assert s['fitting_presentations'] == result['fitting_chars']
        assert s['test_targets'] == result['test_targets_eval_segment'] == 999936
        assert s['selected_writes_per_position'] == result['args']['depth']*result['args']['heads']
        assert s['available_receivers'] == s['selected_writes_per_position']*result['args']['pool']
        near(s['estimated_whole_fit_tflops']*1e12,result['work']['whole_fit_unit_special_flops_estimate'])
        near(s['estimated_fit_mflops_per_presentation']*1e6,result['work']['fit_unit_special_flops_per_char_estimate'])
        near(s['estimated_whole_fit_tflops']*1e6/s['fitting_presentations'],s['estimated_fit_mflops_per_presentation'])
    matched = [json.loads((ROOT/s['result_path']).read_text()) for s in snapshots[:3]]
    for key in ('fit','test','dev','segment','lanes','passes','lr','clip','payload','depth','heads','seed','eval_segment'):
        assert all(r['args'][key] == matched[0]['args'][key] for r in matched)
    for claim in evidence['claims']:
        assert claim['status'] and claim['scope'] and claim['sources']
        assert all((ROOT/p).exists() and 'invalid_protocol' not in p for p in claim['sources'])
    records['completed_result_bindings'] = len(snapshots)
    online=evidence['online_pilot']
    result=json.loads((ROOT/online['result_path']).read_text())
    assert hashlib.sha256((ROOT/online['result_path']).read_bytes()).hexdigest()==online['result_sha256']
    assert result['status']=='completed' and online['protocol']==result['protocol']
    assert online['execution_sources']==result['source_sha256'] and online['architecture']==result['architecture']
    assert result['protocol']['predict_before_update'] and result['protocol']['scored_targets']==8191
    assert result['protocol']['official_test_read'] is False
    arms={r['arm']:r for r in result['rows']}
    near(online['frozen_bpc'],arms['frozen']['bpc']);near(online['online_bpc'],arms['online']['bpc'])
    assert online['updates']==arms['online']['updates']==512
    near(online['total_processing_work_ratio'],result['work']['online']['unit_special_flops']/result['work']['frozen']['unit_special_flops'])
    records['completed_causal_online_pilot_binding']=1
    records['claims_with_existing_scoped_sources'] = len(evidence['claims'])
    manifest = ROOT/'experiments/queue/native_clock_noise_20261004T065700Z/manifest.json'
    frozen = json.loads(manifest.read_text())['source_sha256']
    for path, expected in frozen.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    records['unchanged_clock_admission_sources'] = len(frozen)
    records['clock_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()

    scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
    assert len(scripts) == 1
    fixture = r'''
const assert=require('node:assert/strict');
const vm=require('node:vm');
const values={depth:4,heads:2,pool:4,width:32,mode:'dense','rx-policy':'popcorn','rx-gap':1,'rx-time':1.6,'rx-cutoff':5};
const limits={depth:[1,64,1],heads:[1,16,1],pool:[1,1024,1],width:[2,2048,2],'rx-gap':[.1,5,.1],'rx-time':[0,6,.1],'rx-cutoff':[0,8,.1]};
const ids=['credit','memories','scores','writes','proposals','state','mode-note','rx-arrivals','rx-members','rx-state','rx-emission','rx-note',...Object.keys(values)];
const elements=Object.fromEntries(ids.map(id=>[id,{value:String(values[id]??''),checked:true,textContent:'',handlers:{},addEventListener(k,f){this.handlers[k]=f},checkValidity(){const v=Number(this.value),limit=limits[id];if(!limit)return true;const [min,max,step]=limit;return this.value!==''&&Number.isFinite(v)&&v>=min&&v<=max&&Math.abs((v-min)/step-Math.round((v-min)/step))<1e-8}}]));
let focused=null;
const buttons=Array.from({length:10},(_,i)=>({dataset:{panel:String(i)},handlers:{},attrs:{},addEventListener(k,f){this.handlers[k]=f},setAttribute(k,v){this.attrs[k]=v},focus(){focused=i}}));
const panels=Array.from({length:10},()=>({hidden:false}));
let noCredit=false;
const document={getElementById:id=>elements[id],querySelectorAll:q=>q==='[data-panel]'?buttons:panels,body:{classList:{toggle(k,v){assert.equal(k,'no-credit');noCredit=v}}}};
const context={document,Number,String,Math};
vm.runInNewContext(SOURCE,context);
function set(id,value){elements[id].value=String(value);elements[id].handlers.input()}
function counts(expected){for(const [id,v] of Object.entries(expected))assert.equal(Number(elements[id].textContent),v)}
counts({memories:32,scores:32,writes:8,proposals:32,state:1024});
set('mode','sampled');counts({proposals:16,writes:8});
set('mode','inference');counts({proposals:8,scores:32});
set('pool',1);set('mode','sampled');counts({proposals:8,memories:8,state:256});
set('width',31);assert.match(elements['mode-note'].textContent,/even/);assert.equal(elements.state.textContent,'—');
set('width','');assert.match(elements['mode-note'].textContent,/bounds/);
set('width',32);set('depth',0);assert.match(elements['mode-note'].textContent,/bounds/);
set('depth',4);set('width',3.5);assert.match(elements['mode-note'].textContent,/bounds/);
for(let i=0;i<10;i++){buttons[i].handlers.click();panels.forEach((p,j)=>assert.equal(p.hidden,j!==i));buttons.forEach((b,j)=>{assert.equal(b.attrs['aria-selected'],String(i===j));assert.equal(b.attrs.tabindex,i===j?'0':'-1')})}
function key(index,name,expected){let prevented=false;buttons[index].handlers.keydown({key:name,preventDefault(){prevented=true}});assert.equal(prevented,true);assert.equal(focused,expected);assert.equal(panels[expected].hidden,false)}
key(9,'ArrowRight',0);key(0,'ArrowLeft',9);key(3,'Home',0);key(3,'End',9);
elements.credit.checked=false;elements.credit.handlers.change();assert.equal(noCredit,true);
elements.credit.checked=true;elements.credit.handlers.change();assert.equal(noCredit,false);
function reception(policy,H,cTime,q,members,state,emission){const r=context.traceReception(policy,H,cTime,q);assert.equal(r.accepted.map(e=>e.name).join(','),members);assert.equal(r.state,state);if(emission===null)assert.equal(r.emission,null);else assert.ok(Math.abs(r.emission-emission)<1e-12);return r}
reception('winner',1,1.6,5,'A',1,0);
reception('window',1,1.6,5,'A,B',-1,1);
reception('popcorn',1,1.6,5,'A,B,C',2,2.6);
reception('all',1,1.6,5,'A,B,C,D',6,5);
reception('popcorn',1,1.6,2,'A,B,C',2,null);
reception('popcorn',1,1.6,2.6,'A,B,C',2,2.6);
reception('window',1,.7,5,'A,C,B',2,1);
reception('popcorn',1,2.3,5,'A,B',-1,1.8);
reception('popcorn',1,1.8,1.8,'A,B,C',2,null);
reception('popcorn',1,1.8,2.8,'A,B,C',2,2.8);
reception('winner',1,0,5,'A',1,0);
reception('all',1,1.6,.8,'A,B',-1,.8);
reception('popcorn',5,1.6,5,'A,B,C,D',6,null);
reception('window',5,1.6,5,'A,B,C,D',6,5);
for(const later of [3,6])reception('popcorn',1,later,2,'A,B',-1,1.8);
assert.equal(elements['rx-members'].textContent,'A, B, C');assert.equal(elements['rx-emission'].textContent,'2.60');
set('rx-cutoff',2);assert.match(elements['rx-note'].textContent,/Pending/);assert.equal(elements['rx-emission'].textContent,'Not emitted');
set('rx-gap','');assert.match(elements['rx-note'].textContent,/bounds/);assert.equal(elements['rx-state'].textContent,'—');
set('rx-gap',1);set('rx-time',.35);assert.match(elements['rx-note'].textContent,/bounds/);
console.log('dimensions, modes, invalid-input clearing, ten keyboard tabs, credit visibility and sixteen causal reception cases pass');
'''
    with tempfile.TemporaryDirectory(prefix='architecture-review-') as temp:
        js_path = Path(temp)/'controls.js'
        js_path.write_text('const SOURCE='+json.dumps(scripts[0])+';\n'+fixture)
        subprocess.run(['node', '--check', str(js_path)], check=True, capture_output=True, timeout=10)
        result = subprocess.run(['node', str(js_path)], check=True, capture_output=True, text=True, timeout=10)
    records['javascript_control_contracts'] = result.stdout.strip()

    # Independent scalar witnesses: these validate statements, not learned models.
    witness_cases = 0
    for scores in ((-1.,0.,2.),(0.,0.,0.),(3.,-2.,1.)):
        rates = [math.exp(s) for s in scores]
        z = sum(rates)
        pi = [r/z for r in rates]
        for i in range(3):
            for j in range(3):
                f_choice = (pi[i] if i == j else 0)-pi[i]*pi[j]
                f_clock = pi[i]*pi[j]  # E[(1-W)^2]=1 for unit exponential W.
                near(f_choice+f_clock, pi[i] if i == j else 0)
        values = [2.,-3.,7.]
        mu = sum(p*v for p,v in zip(pi,values))
        delays = [0.4*(s-min(scores)) for s in scores]
        read_time = max(delays)+0.8
        mass = [math.exp(-(read_time-delay)/0.4) for delay in delays]
        near(sum(w*v for w,v in zip(mass,values))/sum(mass),mu)
        retained = (0,2)
        eps = pi[1]
        truncated = sum(pi[i]*values[i] for i in retained)/(1-eps)
        assert abs(mu-truncated) <= eps*(max(values)-min(values))+1e-12
        # Current values are identical, future write returns are not.
        future = [1.,-2.,4.]
        qbar = sum(p*q for p,q in zip(pi,future))
        analytic = [p*(q-qbar) for p,q in zip(pi,future)]
        assert any(abs(g)>1e-5 for g in analytic)
        for i in range(3):
            outputs = []
            for sign in (-1,1):
                perturbed = [math.exp(s+sign*1e-5*(j==i)) for j,s in enumerate(scores)]
                outputs.append(sum(r*q for r,q in zip(perturbed,future))/sum(perturbed))
            near((outputs[1]-outputs[0])/2e-5,analytic[i])
            near(pi[i]*(5.-5.),0.)
        witness_cases += 1
    for q in (.1,.4,1.):
        g = 3.5
        near(q*(g/q-g)**2+(1-q)*g*g,(1/q-1)*g*g)
        witness_cases += 1
    for nu in (0.,.5,1.):
        cv2 = math.gamma(1+2*nu)/math.gamma(1+nu)**2-1
        assert cv2 >= -1e-14
        near(cv2,0. if nu==0 else 1. if nu==1 else 4/math.pi-1)
        witness_cases += 1
    for alphas in ((.1,.2,.3),(.01,.04,.08)):
        c = .8
        lower, upper = math.prod(1-a*c for a in alphas),math.prod(1+a*c for a in alphas)
        diagonal = [math.prod(1+a*j for a in alphas) for j in (-.6,.4)]
        assert lower <= min(diagonal) <= max(diagonal) <= upper
        witness_cases += 1
    # Simultaneous timestamps need snapshot/commit semantics for synchronous inclusion.
    old = [1.,2.]
    reference = [old[1],old[0]+old[1]]
    staged_events = [old[1],old[0]+old[1]]
    immediate = old.copy()
    immediate[0] = immediate[1]
    immediate[1] = immediate[0]+immediate[1]
    assert staged_events == reference and immediate != reference
    witness_cases += 1
    weights, values, mask = [.2,.3,.5],[2.,-1.,4.],[1,1,1]
    near(sum(m*w*v for m,w,v in zip(mask,weights,values)),
         sum(w*v for w,v in zip(weights,values)))
    witness_cases += 1
    theta, eta, labels = 0.,.1,[1.,2.]
    for y in labels:
        theta -= eta*(theta-y)
    near(theta,.29)
    summed_update = -eta*sum(-y for y in labels)
    near(summed_update,.30)
    assert abs(theta-summed_update)>1e-3
    witness_cases += 1
    # Independent product-state operations commute; interleaved queries can differ.
    def op_a(state):
        a,b = state
        return (a+2,b)
    def op_b(state):
        a,b = state
        return (a,b*3)
    assert op_a(op_b((1,2))) == op_b(op_a((1,2))) == (3,6)
    assert op_a((1,2)) != op_b((1,2))
    witness_cases += 1
    # No positive lower bound: arbitrarily many positive delays fit below one.
    assert all(2.**(-i)>0 for i in range(1,31))
    assert sum(2.**(-i) for i in range(1,31)) < 1
    witness_cases += 1
    # Effective block fan-out, not the degree of internal zero-delay operations.
    for b in (0,1,3):
        c,horizon,delta,seeds = 4,1.,.5,[0.,.25]
        frontier = seeds.copy()
        blocks = 0
        while frontier:
            timestamp = frontier.pop()
            if timestamp>horizon:
                continue
            blocks += 1
            frontier.extend([timestamp+delta]*b)
        limit = c*len(seeds)*sum(b**k for k in range(math.floor(horizon/delta)+1))
        assert c*blocks <= limit
    witness_cases += 1
    # Boundary sufficiency: counts cannot answer a last-symbol query.
    histories = ((0,1),(1,0))
    assert sum(histories[0]) == sum(histories[1])
    assert histories[0][-1] != histories[1][-1]
    witness_cases += 1
    # Isolated width growth preserves sums and a pending timeout; losing the
    # timeout preserves a silent query but changes the future boundary trace.
    parent, child = 2., (2.,0.)
    deadline = 2.6
    for x in (-3.,4.,0.,7.):
        parent += x
        child = (child[0]+x,0.*child[1])
        assert child == (parent,0.)
    def timed_observation(value, due, query):
        return (due,value) if due is not None and due<=query else None
    assert timed_observation(parent,deadline,2.) is None
    assert timed_observation(child[0],None,2.) is None
    assert timed_observation(parent,deadline,3.) == timed_observation(child[0],deadline,3.)
    assert timed_observation(child[0],None,3.) is None
    witness_cases += 1
    # Equal functions under coordinate migration can have unequal SGD paths.
    for theta in (.25,1.,4.):
        phi = math.sqrt(theta)
        near(phi*phi,theta)
        near(2*phi/(2*math.sqrt(theta)),1.)
    theta,phi,eta = 1.,1.,.1
    parent_next = theta-eta*theta
    child_next = (phi-eta*2*phi**3)**2
    near(parent_next,.9)
    near(child_next,.64)
    assert abs(parent_next-child_next)>.2
    witness_cases += 1
    # Function-preserving growth may leave new directions locally dormant.
    x,epsilon = 3.,1e-5
    first_u = ((epsilon*0.*x)-(-epsilon*0.*x))/(2*epsilon)
    first_v = ((0.*epsilon*x)-(0.*-epsilon*x))/(2*epsilon)
    first_w = ((epsilon*x)-(-epsilon*x))/(2*epsilon)
    near(first_u,0.)
    near(first_v,0.)
    near(first_w,x)
    assert 1.*1.*x != 0.  # Finite-change expressivity despite zero tangent.
    witness_cases += 1
    # Complete C example: score/update the observed transition before the next
    # forecast. Known expectations independently identify the causal prefix.
    counts=[[0,0],[0,0]];previous=None;predictions=[]
    for symbol in (0,1,0):
        if previous is not None: counts[previous][symbol]+=1
        predictions.append((counts[symbol][1]+1)/(sum(counts[symbol])+2))
        previous=symbol
    for actual,expected in zip(predictions,(.5,.5,2/3)): near(actual,expected)
    assert counts==[[0,1],[1,0]]
    witness_cases+=1
    # H at q=2.5 with arrivals0,1,2 and payloads(1,0),(0,1),(-1,0):
    # H=.5 closes three bursts and keeps the last two; H=1.5 stays pending.
    # Evaluate the declared initial query maps by an independent closed form.
    d=math.exp(-.1);p=1/(1+math.exp(-3*d))
    short_prediction=math.tanh(2-(1-p)*d)-math.tanh(1+p*math.exp(-.2))
    pending_prediction=math.tanh(2)-math.tanh(1)
    assert abs(short_prediction-pending_prediction)>.1
    losses=[.5*(short_prediction-.5)**2,.5*(pending_prediction-.5)**2]
    for scores in ((0.,0.),(2.,-1.),(-3.,1.)):
        rates=[math.exp(s) for s in scores];pi=[r/sum(rates) for r in rates]
        risk=sum(p*l for p,l in zip(pi,losses))
        analytic=[p*(l-risk) for p,l in zip(pi,losses)]
        near(sum(analytic),0.)
        assert any(abs(g)>1e-5 for g in analytic)
        for i in range(2):
            outcomes=[]
            for sign in (-1,1):
                perturbed=[math.exp(s+sign*1e-5*(j==i)) for j,s in enumerate(scores)]
                outcomes.append(sum(r*l for r,l in zip(perturbed,losses))/sum(perturbed))
            near((outcomes[1]-outcomes[0])/2e-5,analytic[i])
        witness_cases+=1
    # Fixed-query evaluation: conditional completion loss cannot replace full
    # workload loss. Include the explicitly declared fallback for every miss.
    observed_losses=[0.,1.];completed=[True,False];fallback=1.
    coverage=sum(completed)/len(completed)
    conditional=sum(l for l,c in zip(observed_losses,completed) if c)/sum(completed)
    workload=sum(l if c else fallback for l,c in zip(observed_losses,completed))/len(completed)
    near(coverage,.5);near(conditional,0.);near(workload,.5)
    assert conditional<workload
    witness_cases+=1
    # Exact finite route credit for complete query utility, including work.
    # Work1 versus work.2; the preferred route flips at beta=.125.
    for beta in (0.,.125,.2):
        returns=[.4+beta*1.,workload+beta*.2]
        if beta==0.: assert returns[0]<returns[1]
        elif beta==.125: near(*returns)
        else: assert returns[1]<returns[0]
        scores=[.3,-.2];rates=[math.exp(s) for s in scores]
        probabilities=[r/sum(rates) for r in rates]
        utility=sum(p*q for p,q in zip(probabilities,returns))
        gradient=[p*(q-utility) for p,q in zip(probabilities,returns)]
        near(sum(gradient),0.)
        for i in range(2):
            values=[]
            for sign in (-1,1):
                perturbed=[math.exp(s+sign*1e-5*(j==i)) for j,s in enumerate(scores)]
                values.append(sum(r*q for r,q in zip(perturbed,returns))/sum(perturbed))
            near((values[1]-values[0])/2e-5,gradient[i])
        witness_cases+=1
    records['independent_scalar_witness_cases'] = witness_cases
    # Editorial protocol status must reflect the separately source-bound 90M
    # completion, even though the deck's 10M numerical ledger stays frozen.
    pitch=json.loads((ROOT/'investment/PITCH_DECK.json').read_text())
    protocol=next(s for s in pitch['slides'] if s.get('section')=='Appendix B / protocol')
    larger=next(s for s in snapshots if s['fitting_presentations']>80_000_000)
    assert f"{larger['test_bpc']:.6f} BPC" in protocol['notes']
    assert 'test[95M:96M]' in protocol['notes'] and 'completed' in protocol['notes']
    records['investor_protocol_completed_90m_binding']=larger['result_path']
    # Bind the execution plan's evidence table to the already validated result
    # snapshots; editorial changes must not invent quality or cost improvements.
    proof_plan=(ROOT/'investment/INVESTOR_PROOF_PLAN.md').read_text()
    anchors={s['id']:s for s in snapshots}
    for snapshot in snapshots:
        assert f"{snapshot['test_bpc']:.6f}" in proof_plan
    base,credited,pool=(anchors[k] for k in ('timing_only','value_credit','larger_pool'))
    overhead=100*(credited['estimated_whole_fit_tflops']/base['estimated_whole_fit_tflops']-1)
    pool_ratio=pool['estimated_whole_fit_tflops']/credited['estimated_whole_fit_tflops']
    assert f'{overhead:.2f}%' in proof_plan and f'{pool_ratio:.3f}×' in proof_plan
    assert (credited['available_receivers'],pool['available_receivers'])==(16,32)
    assert credited['selected_writes_per_position']==pool['selected_writes_per_position']==8
    pilot=evidence['online_pilot']
    for value in (pilot['frozen_bpc'],pilot['online_bpc']):
        assert f'{value:.6f}' in proof_plan
    assert f"{pilot['total_processing_work_ratio']:.3f}×" in proof_plan
    records['investor_proof_plan_completed_quality_and_work_bindings']=5
    # Independent equity arithmetic, tied to the frozen €3M budget rather than
    # inferred from application breadth or a benchmark score.
    funding=json.loads((ROOT/'investment/pitch_deck_evidence_20261003.json').read_text())
    raised=Decimal(sum(funding['budget_eur'].values()))/Decimal(1_000_000)
    rationale=(ROOT/'investment/VALUATION_RATIONALE.md').read_text()
    prices=re.findall(r'^\| €(\d+)M \| €(\d+)M \| ([\d.]+)% \|',rationale,re.M)
    assert len(prices)==2
    for premoney,postmoney,ownership in prices:
        pre,post=Decimal(premoney),Decimal(postmoney)
        assert post==pre+raised
        actual=Decimal(100)*raised/post
        assert Decimal(ownership)==actual.quantize(Decimal('.0001'))
    near(float(raised/Decimal(prices[0][1])),funding['financial_assumptions']['initial_investor_ownership'])
    records['valuation_equity_cases_and_frozen_raise_budget']=len(prices)
    artifacts = docs+[ROOT/'scripts/build_architecture_atlas.py',ROOT/'scripts/build_opportunity_diagram.py',ROOT/'scripts/check_architecture_review.py',
        ROOT/'report/architecture_evidence.json',ROOT/'report/architecture_atlas.html',
        ROOT/'report/architecture_source_inventory.json',ROOT/'README.md',ROOT/'REPORT.md',
        ROOT/'experiments/THEORY.md',ROOT/'experiments/LOCAL_HANDOFF.md',
        ROOT/'report/family_report.py',ROOT/'report/readable_report.py',ROOT/'report/sleeping_machines_status.pdf',
        ROOT/'scripts/build_pitch_deck.py',ROOT/'scripts/package_pitch_diligence.py',
        ROOT/'investment/README.md',ROOT/'investment/INVESTOR_READING_REVIEW.md',
        ROOT/'investment/PITCH_DECK.json',ROOT/'investment/PITCH_DECK_NOTES.md',
        ROOT/'investment/sleeping_machines_pitch_deck.pdf',ROOT/'investment/sleeping_machines_pitch_deck_main.pdf']+svgs
    report_sha=hashlib.sha256((ROOT/'report/sleeping_machines_status.pdf').read_bytes()).hexdigest()
    deck_sha=hashlib.sha256((ROOT/'investment/sleeping_machines_pitch_deck.pdf').read_bytes()).hexdigest()
    for directory,pattern,expected in (('report','publication_model_family_*.json',report_sha),
                                       ('investment','publication_pitch_deck_*.json',deck_sha)):
        publications=[p for p in (ROOT/directory).glob(pattern) if json.loads(p.read_text()).get('output_sha256')==expected]
        assert len(publications)==1,(directory,publications)
        publication=json.loads(publications[0].read_text())
        assert publication['status']=='completed'
        for path,digest in publication['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,('publication source',path)
        artifacts.append(publications[0])
    records['current_pdf_publications_and_source_bindings']=2
    records['documentation_artifact_sha256'] = {
        str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in artifacts}
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), status='pass',
                scope='Documentation and scalar algebra only; no numerical model, training or benchmark',
                checks=records, browser_pixel_rendering='not performed: no local browser renderer',
                imports='Python standard library and Node built-ins only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
