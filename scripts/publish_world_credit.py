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
        path = ROOT/'experiments/causal_induction/NATIVE_SENSOR.md'; text = path.read_text()
        marker = 'Completed result files and [WORLD_CREDIT_DECISION.md](WORLD_CREDIT_DECISION.md) determine continuation; no pending score is a result.'
        assert marker in text
        text = text.replace(marker, 'Completed causal reuse: '+numbers+'. Development gate '+outcome+'. '+next_step+' Full per-system/half and causal counting diagnostics are in the immutable decision/result files. Generating depth differs from neural depth; this experiment changes source-choice teaching only.')
        path.write_text(text)
    subprocess.run([python, 'scripts/update_curie_learning_status.py'], cwd=ROOT, check=True)
    subprocess.run([python, 'scripts/build_credit_report_parts.py', '--tag', a.tag], cwd=ROOT, check=True)


if __name__ == '__main__': main()
