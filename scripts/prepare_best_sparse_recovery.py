"""Reuse source-exact sparse validation for the best available completed fit.

Only stdlib bytes/metadata; preserve the actual producer's source aliases and
checkpoint. No runtime, training, rescore or host admission is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from sparse_validation_recovery import (dump, git_bytes, materialize, safe_name,
                                       sha, verify_record)


def prepare(template_name, parent_name, tag):
    if not re.fullmatch(r'aws_[A-Za-z0-9_]+', tag):
        raise ValueError('Fresh AWS-specific tag required')
    template_path = ROOT/safe_name(template_name)
    template = json.loads(template_path.read_text())
    verify_record(template)
    parent_path = ROOT/safe_name(parent_name)
    parent = json.loads(parent_path.read_text())
    if parent['status'] != 'completed' or parent['args'].get('max_windows', 0):
        raise ValueError('Completed quality fit required')
    parent_hash = sha(parent_path)
    checkpoint = safe_name(parent['final_weights'])
    weights_hash = sha(ROOT/checkpoint)
    with tempfile.TemporaryDirectory(prefix='best-sparse-prepare-', dir=ROOT/'.git') as folder:
        snapshot = materialize(template, Path(folder))
        additional = {}
        for name, digest in parent['source_sha256'].items():
            safe_name(name)
            destination = snapshot/name
            if destination.exists():
                if sha(destination) != digest:
                    raise ValueError('Canonical historical producer mismatch: ' + name)
            else:
                if sha(ROOT/name) != digest:
                    raise ValueError('Missing exact producer alias/provenance file: ' + name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT/name).read_bytes())
                additional[name] = digest
        # The recorded archived sources really are the canonical numerical
        # files the paired validator will load, rather than just similar names.
        for name, alias in parent.get('execution_source_aliases', {}).items():
            if sha(snapshot/safe_name(name)) != alias['actual_sha256']:
                raise ValueError('Execution alias does not match the restored canonical source')
            if sha(snapshot/safe_name(alias['actual_source'])) != alias['actual_sha256']:
                raise ValueError('Changed archived producer source')
        parent_copy = snapshot/parent_name
        parent_copy.parent.mkdir(parents=True, exist_ok=True)
        parent_copy.write_bytes(parent_path.read_bytes())
        copied_weights = snapshot/checkpoint
        copied_weights.parent.mkdir(parents=True, exist_ok=True)
        copied_weights.write_bytes((ROOT/checkpoint).read_bytes())
        builder = 'scripts/prepare_trained_sparse_validation.py'
        builder_copy = snapshot/builder
        builder_copy.parent.mkdir(parents=True, exist_ok=True)
        builder_copy.write_bytes(git_bytes(template['historical_revision'], builder))
        previous_path = list(sys.path)
        try:
            runpy.run_path(str(builder_copy))['prepare'](parent_name, tag+'_original')
        finally:
            sys.path[:] = previous_path
        original = f'experiments/queue/{tag}_original/manifest.json'
        ladder = json.loads((snapshot/original).read_text())
        directory = ROOT/'experiments/queue'/tag
        directory.mkdir(exist_ok=False)
        old_directory = ROOT/'experiments/queue'/(tag+'_original')
        old_directory.mkdir(exist_ok=False)
        # Publish the freshly frozen original protocol separately from the
        # recovery adapter. It is source-bound but not admitted on live code.
        for file in (snapshot/'experiments/queue'/(tag+'_original')).iterdir():
            with (old_directory/file.name).open('xb') as stream:
                stream.write(file.read_bytes())
        content = {name:(snapshot/name).read_bytes() for name in template['archived_sha256']
                   if name.endswith('.py')}
        content.update({name:(snapshot/name).read_bytes() for name in additional})
        content[builder] = builder_copy.read_bytes()
        content[parent_name] = parent_path.read_bytes()
        content[original] = (snapshot/original).read_bytes()
        pins = dict(parent['source_sha256'])
        jobs = []
        for index, old in enumerate(ladder['jobs']):
            spec = json.loads((snapshot/old['manifest']).read_text())
            for name, digest in spec['source_sha256'].items():
                if name in pins and pins[name] != digest:
                    raise ValueError('Incompatible canonical validation pins')
                pins[name] = digest
            for name in (old['manifest'], old['queue']):
                content[name] = (snapshot/name).read_bytes()
            stage = 'contracts' if index == 0 else f"{old['stage']}_{old['split']}_T{old['segment']}"
            name = tag+'_'+stage
            queue = directory/(name+'.txt')
            g = old['guards']
            queue.write_text('# PREPARED UNRUN; owner priorities and physical AWS reservation first.\n'
                f"# MEM_CAP_KB={g['address_space_kib']} MEM_CAP_RSS_KB={g['rss_kib']} "
                f"MIN_AVAIL_MB={g['min_available_mib']} JOB_TIMEOUT_S={g['timeout_s']}\n"
                f'{name} experiments/sparse_validation_recovery.py run --manifest '
                f'experiments/queue/{tag}/manifest.json --stage {stage}\n')
            jobs.append(dict(stage=stage, original=old, queue=str(queue.relative_to(ROOT)),
                queue_sha256=sha(queue), output=f'experiments/results/diagnostics/{name}.json', guards=g,
                required_stages=[] if index == 0 else ['contracts']+
                    ([f"prefix_dev_T{old['segment']}"] if old['stage']=='full' else [])))
        for name, digest in pins.items():
            if hashlib.sha256(content[name]).hexdigest() != digest:
                raise ValueError('Historical bundle differs from an actual saved source')
        archive = directory/'historical_sources.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
            for name, payload in sorted(content.items()):
                stream.writestr(safe_name(name), payload)
        record = dict(template, tag=tag, original_manifest=original,
            original_manifest_sha256=sha(ROOT/original), parent=parent_name,
            parent_sha256=parent_hash, checkpoint=checkpoint, checkpoint_sha256=weights_hash,
            archive=str(archive.relative_to(ROOT)), archive_sha256=sha(archive), jobs=jobs,
            archived_sha256={n:hashlib.sha256(b).hexdigest() for n,b in sorted(content.items())},
            frozen_source_pins=pins, source_files=sum(n.endswith('.py') for n in content),
            current_source_drift=[n for n,h in pins.items() if sha(ROOT/n)!=h],
            template_manifest_sha256=sha(template_path), additional_pinned_sources=additional,
            preparation_source_sha256=sha(Path(__file__)), saved_test_bpc_T256=parent['test_bpc_eval_segment'],
            scope='Best available completed 90M checkpoint, original source aliases retained. '
                  'Canonical execution is the historical exact producer/validator tree; additional '
                  'producer metadata sources are byte-bound to the completed result. '
                  'Seven stages remain unrun; no source, weights or score modified.')
        dump(directory/'manifest.json', record)
    if sha(parent_path)!=parent_hash or sha(ROOT/checkpoint)!=weights_hash:
        raise ValueError('Evidence changed during preparation')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    record = prepare(args.template, args.parent, args.tag)
    print(json.dumps(dict(status=record['status'], sources=record['source_files'],
        frozen_source_pins=len(record['frozen_source_pins']), saved_bpc=record['saved_test_bpc_T256'],
        jobs=len(record['jobs']))))
