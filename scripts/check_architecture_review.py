"""Check local documentation contracts and scalar identities without ML imports.

These checks are not model, training, browser-rendering or benchmark tests.
"""
import argparse
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
    assert len(svgs) == 8
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
    assert len(tabs) == len(panels) == 8
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
        'model_family_overview.md','model_family_example.md','architecture_evidence.md')]
    docs += [ROOT/'experiments/theory/152_primitives_integration_and_capability_bounds.md']
    links = 0
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if re.match(r'^[a-z]+:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0])
            assert (path.parent/target).exists(), (path, target)
            links += 1
    records['markdown_local_links'] = links
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
const buttons=Array.from({length:8},(_,i)=>({dataset:{panel:String(i)},handlers:{},attrs:{},addEventListener(k,f){this.handlers[k]=f},setAttribute(k,v){this.attrs[k]=v},focus(){focused=i}}));
const panels=Array.from({length:8},()=>({hidden:false}));
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
for(let i=0;i<8;i++){buttons[i].handlers.click();panels.forEach((p,j)=>assert.equal(p.hidden,j!==i));buttons.forEach((b,j)=>{assert.equal(b.attrs['aria-selected'],String(i===j));assert.equal(b.attrs.tabindex,i===j?'0':'-1')})}
function key(index,name,expected){let prevented=false;buttons[index].handlers.keydown({key:name,preventDefault(){prevented=true}});assert.equal(prevented,true);assert.equal(focused,expected);assert.equal(panels[expected].hidden,false)}
key(7,'ArrowRight',0);key(0,'ArrowLeft',7);key(3,'Home',0);key(3,'End',7);
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
console.log('dimensions, modes, invalid-input clearing, eight keyboard tabs, credit visibility and sixteen causal reception cases pass');
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
    records['independent_scalar_witness_cases'] = witness_cases
    artifacts = docs+[ROOT/'scripts/build_architecture_atlas.py',ROOT/'scripts/check_architecture_review.py',
        ROOT/'report/architecture_evidence.json',ROOT/'report/architecture_atlas.html',
        ROOT/'report/architecture_source_inventory.json']+svgs
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
