"""Publish only completed native world-credit evidence into current documents."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    python = str(ROOT/'.venv-docker/bin/python')
    decision = ROOT/'experiments/results/causal_induction/curie_world_credit_20261012_v2_decision.json'
    data = json.loads(decision.read_text()); assert data['status'] == 'completed'
    for group in ['source_sha256', 'parent_sha256']:
        for name, digest in data[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    rows = {r['arm']: r for r in data['rows']}
    outcome = 'passes' if data['development_gate'] else 'fails'
    numbers = ', '.join(f"{arm} NLL{rows[arm]['pooled_nll']:.6f}/complete stage{rows[arm]['complete_acquisition_and_reuse_stage_wall_s']:.3f}s" for arm in ['learned', 'exact', 'prior'])
    next_step = ('Register unchanged fresh-actor/fresh-system confirmation before scaling.'
                 if data['development_gate'] else
                 'No scale-up or source patch grid. Diagnose conditional outcome sufficiency, route sensitivity and shared-memory information from completed causal traces before the next architectural change.')
    paragraph = ('**Curie native outcome-law acquisition/reuse — 12 October, complete.** WorldCreditSensor retains native two-layer F/C temporal races, independent shared-bank access and future forward-task memory teaching. One target-addressed outcome query/update supplies coherent proper-loss projections into source-choice credit. Common acquisition16nonlinear TRAIN systems/2048unique targets,4096charged presentations. Eight fresh independent factor8/16 and generating-depth4/8 base/hidden-rewire deployments,1024prequential targets per arm, identical incoming actor/world/Adam fingerprints; all arms train the world and expose the same memory capability. '+numbers+'. Development gate '+outcome+'; gains '+', '.join(f"versus{k} {v:+.6f}" for k,v in data['quality_gains'].items())+f". Learned world gain versus causal global frequency{data['learned_world_gain_vs_frequency']:+.6f}; actor versus source-calibrated persistence{data['learned_actor_gain_vs_persistence']:+.6f}. "+next_step+' Native outcome/read/future-task gradients, active relabeling and causal visibility contracts pass. All sources/data/checkpoints/issued traces preserved; oneCPU/RSS1.5GB/9GiBfloor. Proper-loss half blends deliberately change conditional teaching means; no calibrated general route-response geometry, global learned destinations, fulloptimizer teaching, neural-depth replacement, publicTEST or efficiency claim. Old raw-credit TRANSFER.md gate stays closed. Preserve AWS mainR1 original P24D2H2U4 proper-token program. Commands/logs.git/curie-world-credit-preview/20261012_v1; report current; checkpoints preserved.')
    with open('/tmp/aws-language-publication.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'main'
        assert not subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=U'], cwd=ROOT, text=True).strip()
        path = ROOT/'experiments/HANDOFF.md'; text = path.read_text()
        marker = '**Curie native outcome-law acquisition/reuse — 12 October,'
        start = text.index(marker); end = text.index('\n', start)
        path.write_text(text[:start]+paragraph+text[end:])
        text = path.read_text()
        old = '**Curie native reciprocal shared-memory development — 12 October, integration round complete.**'
        if old in text:
            start = text.index(old); end = text.index('\n', start)
            text = text[:start]+text[end+1:]
            path.write_text(text)
        path = ROOT/'experiments/causal_induction/NATIVE_SENSOR.md'; text = path.read_text()
        table = '\n'.join(f"| {arm} | {rows[arm]['pooled_nll']:.6f} | {rows[arm]['world_nll']:.6f} | {rows[arm]['complete_acquisition_and_reuse_stage_wall_s']:.3f} |" for arm in ['exact', 'learned', 'prior'])
        text = f'''# Native world credit and reciprocal memory

B5/R1, Curie,12October2026. The objective is a general-purpose temporal/event substrate. This private causal-system project develops useful learned credit; proper-token language and public benchmarks remain the main value targets.

## Current integrated architecture

WorldCreditSensor uses our AddressedEventHeads for both actor and critic: two temporal layers, two race heads/two receiver alternatives, eight values/head, separate keys/values, decaying/rotating persistent state and hard selected writes. Parameters are shared across public addresses; IDs retrieve numerical state rather than identity embeddings. The critic maintains one target-addressed world state and its own race random stream. It predicts the conditional binary outcome, supervised by actual delayed labels.

One world query consumes target context, pooled discovered candidate context, public bit/age/horizon and relationship evidence. One feedback event carries the actually revealed target. Issued world probabilities are preserved before feedback. The simple pool interface is not proved response-sufficient; it currently omits actual actor/counterfactual logits as query features. Proper-loss algebra projects the common belief into coherent alternative-route risks using those supplied logits.

Both processors choose their own native reads/writes over one versioned observation/world bank. Public relative endpoint/role keys and indexed discovery use up to16visible records. Writes become visible after native completion. Native receiver destinations are learned within addressed pools; global record destinations remain public addresses. Forward reads consume world memory through a bounded residual. Outcome BCE trains the actual native world body/read policy; incoming F features/payloads are detached for that auxiliary objective, while later forward task losses train critic-produced memory. Gradients cover eight-feedback windows; numerical memory persists.

Every learning query decodes its sampled alternative. Exact credit uses its realized BCE contrast; learned uses the half blend of actual label and issued world probability; prior uses the half blend with0.5. All arms retain the same outcome training and memory capability. The blends deliberately change conditional teaching means when their belief is wrong. Known local native derivatives/counterfactual surrogates remain; this changes source-choice teaching, not complete BPTT or optimizer-history credit.

## Completed common acquisition and causal reuse

Source:16nonlinear TRAIN systems,2048unique outcomes, one neural pass plus2048charged confidence presentations. Each target arm deploys eight independent clones of identical acquired actor/world/Adam state, with fresh address-specific memories and causal online updates. Fresh systems vary factors8/16 and generating depth4/8, base/hidden-rewire;1024issued-before-feedback targets per arm. Generating depth is not neural depth.

| Source-choice teaching | Actor NLL | World NLL | Complete source+reuse seconds |
|---|---:|---:|---:|
{table}

Development gate **{outcome}**. {next_step} Learned gains versus controls: {data['quality_gains']['exact']:+.6f} exact, {data['quality_gains']['prior']:+.6f} prior. World gain versus causal global-frequency diagnostic {data['learned_world_gain_vs_frequency']:+.6f}; actor gain versus source-calibrated causal persistence {data['learned_actor_gain_vs_persistence']:+.6f}. Single-actor exploratory evidence; no publicTEST, neural-depth/size scaling or efficiency claim. Full system/half results and issued traces are immutable result files.

## Resource and mechanism evidence

Contracts pass native outcome/read gradients, actual future F-task teaching of C memory versus untaught, label-independent issued forecasts, independent world queries, active public relabeling and future-record invisibility. Source/largest16-target smokes took5.045/4.195s at531MB. A later1492-event burst exceeded the whole-arm1.5GBcap; the watchdog stopped safely. Per-system processes with2.2GBRSScap/9GiBhostfloor preserve the learning math. Largest exact case completes104.405s/1.57GB; allseven reused case scores match prior partial scores exactly. Failed execution logs are development overhead; completed recipe work is reported separately.

Charge confidence acquisition, native proposals/keys/writes, alternative decodes, relationship evidence, every discovered/transported bank payload, optimizer, checkpoint copies and serving memory. Full stage time includes preprocessing/copies; source acquisition is paid once per eight-deployment cohort. Full denominator5120presentations is shared by every arm; training-loop and single-deployment costs remain in the decision. Key scoring currently reads full payloads; selected-value counters are not total traffic. No FLOP or energy estimate is inferred from them.

## Next theory and implementation decision

[Route similarity](../theory/ROUTE_SIMILARITY_AND_CREDIT_GENERALIZATION.md) defines functional equivalence through future causal response, with task-specific compact projections from rich native state. [Forward response sufficiency](../theory/FORWARD_RESPONSE_SUFFICIENCY_FOR_CREDIT.md) shows why calibrated pooled world prediction can miss credit-relevant information. [Shared producer noise](../theory/COHERENT_WORLD_CREDIT_AND_SHARED_PRODUCER_NOISE.md) shows why independent scalar credit improvements can break factual/route noise cancellation. These determine the diagnosis before a new model or fit is admitted.

Global learned destinations, calibrated general response geometry, economical discovery and full future-optimizer credit remain integration work. Hidden parent/gate oracle files are forbidden for training/selection. The old raw-credit TRANSFER.md gate stays closed. No new dense controller or external reference training is queued.

Saved affine selection reference is a different protocol: raw native credit0.613521 versus exact0.617196; fixedthree-actor/new-graph cohort beatszero/reset3/3 and exact2/3. Four route-loss shared-bank prototypes failed adoption. Their completed source-bound evidence is preserved in the report appendix; it is not directly compared with the nonlinear table above.
'''
        path.write_text(text)
    subprocess.run([python, 'scripts/update_curie_learning_status.py'], cwd=ROOT, check=True)
    subprocess.run([python, 'scripts/build_credit_report_parts.py', '--tag', a.tag], cwd=ROOT, check=True)


if __name__ == '__main__': main()
