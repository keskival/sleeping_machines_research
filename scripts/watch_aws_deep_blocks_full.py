#!/usr/bin/env python3
"""Publish the completed three-seed block-credit adoption decision; no fitting."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
COORD = ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z/worker_recovery.status.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    result_path = ROOT/'experiments/results/credit'/f'{args.tag}.json'
    preview = ROOT/'.git/aws-deep-blocks-watch'/args.tag
    preview.mkdir(parents=True, exist_ok=True)
    with (preview/'watch.lock').open('a') as reservation:
        fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = preview/'state.json'
        if state_path.exists() and json.loads(state_path.read_text()).get('status') == 'completed': return
        state_path.write_text(json.dumps(dict(status='waiting_completed_publication', tag=args.tag))+'\n')
        deadline = time.monotonic()+10800
        while time.monotonic() < deadline:
            state = json.loads(COORD.read_text())
            errors = [e for e in state.get('errors', [])+state.get('blocked', []) if e['job'] == args.tag]
            if errors:
                state_path.write_text(json.dumps(dict(status='needs_review', errors=errors), indent=2)+'\n')
                raise RuntimeError(errors)
            if args.tag in state['completed']: break
            time.sleep(30)
        else: raise TimeoutError('Quality decision did not publish in three hours')
        r = json.loads(result_path.read_text())
        assert r['status'] == 'completed' and r['stage'] == 'quality_decision'
        for path, digest in r['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
        for path, digest in r['inputs'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
        with open('/tmp/aws-language-publication.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'main'
            verdict = 'passes' if r['quality_preservation_gate_pass'] else 'fails'
            text = (f"Three-seed 20-pass quality gate {verdict}: mean final DEV {r['mean_final_dev_ll']:.6f} "
                    f"versus saved dense traces {r['reference_mean_final_dev_ll']:.6f}, "
                    f"paired gains {r['paired_final_dev_gains']}. {r['decision']}. "
                    f"Mean whole fit {r['mean_whole_fit_wall_s']/60:.1f} minutes. No TEST or new leaderboard claim.")
            doc = ROOT/'experiments/credit/ONLINE_DEEP_BLOCKS.md'; heading = '\n## Current full-fit result\n'
            doc.write_text(doc.read_text().split(heading)[0]+heading+'\n'+text+'\n')
            science = ROOT/'report/I_SCIENCE.md'; marker = '**AWS block-credit adoption:**'
            lines = [line for line in science.read_text().splitlines() if not line.startswith(marker)]
            insertion = next(i for i, line in enumerate(lines) if line.startswith('**Credit development:**'))+1
            lines[insertion:insertion] = ['', marker+' '+text, '']
            science.write_text('\n'.join(lines)+'\n')
            handoff = ROOT/'experiments/HANDOFF.md'
            handoff.write_text(f"**AWS block-credit adoption decision — {time.strftime('%d %b %H:%M UTC', time.gmtime())}.** "
                f"`{args.tag}`: {text} Result/checkpoints published; current report refreshed. "
                'No automatic fit follows. Next: economical exact reference integration with learned-credit '
                'work, or stability diagnosis according to the gate. Original AWS controls remain admitted.\n\n'+handoff.read_text())
            for script in ('report/make_pdf.py', 'report/split_report.py'):
                subprocess.run([str(ROOT/'.venv-docker/bin/python'), script], cwd=ROOT, check=True)
            files = ['experiments/credit/ONLINE_DEEP_BLOCKS.md', 'experiments/HANDOFF.md', 'report/I_SCIENCE.md',
                     'REPORT.md', 'report/sleeping_machines_status.pdf', 'report/II_METHODS.md', 'report/III_RECORD.md',
                     'report/figures/aws_deep_credit_blocks.png', 'report/figures/aws_deep_credit_blocks.svg',
                     'report/figures/aws_deep_credit_blocks.inputs.json',
                     'report/figures/current_native_language_status.svg',
                     'report/figures/latest_native_language_fitting.svg',
                     'report/sleeping_machines_status.family.text.pdf']
            files += [str(p.relative_to(ROOT)) for p in (ROOT/'report').glob('I*_*.pdf')]
            files = sorted(set(files))
            subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
            subprocess.run(['git', 'commit', '--only', '-m', 'Publish three-seed block-credit adoption decision and current report',
                            '--', *files], cwd=ROOT, check=True)
            for attempt in range(6):
                subprocess.run(['git', 'pull', '--rebase'], cwd=ROOT, check=True)
                if subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT).returncode == 0: break
            else: raise RuntimeError('Report push retries exhausted')
            state_path.write_text(json.dumps(dict(status='completed', result_sha256=hashlib.sha256(
                result_path.read_bytes()).hexdigest(), tag=args.tag), indent=2)+'\n')


if __name__ == '__main__': main()
