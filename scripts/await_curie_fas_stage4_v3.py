"""B3 v3 admission: require selected native reconstruction/causality preflight.

Reuses the source-pinned v2 scoring construction. Adds the already admitted
native preflight to the dependency graph before committing fresh queues.
"""
import argparse
import json
import sys
from pathlib import Path

import await_curie_fas_stage4 as base


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--native-preflight-manifest',required=True)
    args,remaining=parser.parse_known_args()
    path=base.ROOT/args.native_preflight_manifest
    expected=base.sha(path)
    preflight=json.loads(path.read_text())['jobs'][0]
    original=base.prepare
    wrapper_path=Path(__file__).resolve()
    wrapper_digest=base.sha(wrapper_path)
    def prepare(tag,contract):
        if base.sha(path)!=expected or base.sha(wrapper_path)!=wrapper_digest:
            raise ValueError('Pinned native-preflight admission or watcher changed')
        prepared,missing=original(tag,contract)
        if prepared is None:return None,missing
        manifest_path,tracked=prepared
        manifest=base.ROOT/manifest_path
        plan=json.loads(manifest.read_text())
        plan['jobs'].insert(1,preflight)
        for job in plan['jobs'][2:]:
            job.setdefault('requires',[]).append(preflight['tag'])
            job['sources'][args.native_preflight_manifest]=expected
            job['sources'][str(wrapper_path.relative_to(base.ROOT))]=wrapper_digest
        plan['purpose']+='; selected native reconstruction and suffix-causality preflight required before scoring'
        base.atomic_json(manifest,plan)
        return (manifest_path,tracked),[]
    base.prepare=prepare
    sys.argv=[sys.argv[0],*remaining]
    base.main()


if __name__ == '__main__':main()
