"""Standard-library regression/admission checks for P0 comparison publication.

No numerical libraries, model evaluation, profiling or optimizer work.
"""
import argparse
import ast
import builtins
import copy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(manifest):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('numpy', 'torch', 'h5py', 'matplotlib'):
            raise AssertionError('Numerical import forbidden: '+name)
        return original(name, *args, **kwargs)
    builtins.__import__ = guarded
    try:
        ns = runpy.run_path(str(ROOT/'report/native_language_batched_appendix.py'))
        data = ns['load'](lambda p: json.loads((ns['RES']/p).read_text()))
        public = runpy.run_path(str(ROOT/'report/public_benchmarks_appendix.py'))['load']()
        score = runpy.run_path(str(ROOT/'report/scoreboard.py'))
        choose = score['best_within']; passed = []
        cheap = dict(label='cheap', test256=2., whole=100, sparse=5)
        better = dict(label='expensive', test256=1., whole=101, sparse=6)
        assert choose([cheap, better], 100, 'whole') == cheap
        assert choose([cheap, better], 5, 'sparse') == cheap
        assert choose([cheap], 99, 'whole') is None
        passed += ['training budget excludes better over-budget row', 'inference budget excludes better over-budget row', 'missing within-budget candidate stays pending']
        for change in (dict(test256=None), dict(test256=float('nan')), dict(test256=float('inf')),
                       dict(comparison_eligible=False), dict(whole=-1), dict(whole=0), dict(whole=float('inf'))):
            row = dict(cheap, **change)
            assert choose([row], 100, 'whole') is None
        passed.append('seven invalid metric/protocol/cost cases excluded')
        fixture = dict(data, native90=[better], controls90=[dict(label='ref', test=1.5, whole=100)])
        table = score['page'](fixture, public)[2][1][1]
        assert next(r[3] for r in table if r[0]=='90M vs ref') == 'Pending'
        passed.append('90M cannot publish over-budget win')
        good = json.loads((ROOT/data['native'][-1]['path']).read_text())
        assert ns['comparison_eligible'](good, 10_000_000)
        for edit in ('status', 'max_windows', 'fit', 'test', 'eval_segment', 'targets'):
            bad = copy.deepcopy(good)
            if edit == 'status': bad['status']='running'
            elif edit == 'max_windows': bad['args']['max_windows']=2
            elif edit == 'fit': bad['args']['fit']=90_000_000
            elif edit == 'test': bad['protocol']['test']=[95_000_001,96_000_001]
            elif edit == 'eval_segment': bad['eval_segment']=128
            else: bad['test_targets_eval_segment']=999999
            assert not ns['comparison_eligible'](bad, 10_000_000)
        passed.append('six incomplete/pilot/data/window/target mutations excluded')
        for n in (256, 512, 1024, 1_000_000):
            # Distinct scorer is imported without Torch: its plan must equal native's source expression.
            sys.path.insert(0, str(ROOT/'experiments'))
            rescore = runpy.run_path(str(ROOT/'experiments/reference_window_rescore.py'))
            if n==256:
                try: list(rescore['windows'](n))
                except ValueError: pass
                else: raise AssertionError('Too-short slice accepted')
                continue
            plan=list(rescore['windows'](n)); expected=list(range(0,n-256-1,128))
            assert [s for s, _, _ in plan]==expected
            done=1
            for s,b,e in plan:
                assert s+b+1==done;done=s+e+1
            assert sum(e-b for _,b,e in plan)==done-1
            if n==1_000_000: assert done-1==999936
        passed.append('exact contiguous native target geometry, including 63 omitted tail positions')
        record, raw = rescore['preflight'](manifest, enforce_host=False)
        assert record['targets']==999936 and len(raw)==1_000_000
        try: rescore['preflight'](manifest)
        except ValueError as exc: assert 'physical AWS' in str(exc)
        else: raise AssertionError('Review workspace admitted numerical run')
        passed += ['actual checkpoint/source/data admission', 'physical reservation rejects review workspace']
        with tempfile.TemporaryDirectory() as temp:
            bad=copy.deepcopy(record);bad['checkpoint_sha256']='0'*64
            path=Path(temp)/'bad.json';path.write_text(json.dumps(bad))
            try: rescore['preflight'](path, enforce_host=False)
            except ValueError: pass
            else: raise AssertionError('Changed weight identity accepted')
        passed.append('changed checkpoint identity rejected before numerical import')
        # Model definitions match the original checkpoint producer's Git source.
        historical=subprocess.check_output(['git','show','fdc886cb7f85f8e2e4b5df9af4bbdac442c84484:experiments/e64_lm_baselines.py'], cwd=ROOT, text=True)
        oldclasses=[n for n in ast.parse(historical).body if isinstance(n,ast.ClassDef) and n.name in ('LSTMLM','TfLM')]
        assert ast.dump(ast.Module(body=oldclasses,type_ignores=[]))==ast.dump(rescore['model_tree'](ROOT/'experiments/e64_lm_baselines.py'))
        passed.append('both current E64 model classes match historical definitions structurally')
        # Canonical MG collector, tested with duplicated and mismatched batches in isolated fixture roots.
        collector=runpy.run_path(str(ROOT/'experiments/public_benchmarks/collect_neurobench_mg.py'))['collect']
        original_root=collector.__globals__['ROOT']
        for mutation in ('duplicate', 'source', 'data', 'settings', 'primary'):
            with tempfile.TemporaryDirectory() as temp:
                fake=Path(temp);out=fake/'experiments/results/neurobench_mg';out.mkdir(parents=True)
                parents=public['mg']['parents'];assert len(parents)>=2
                for i,p in enumerate(parents[:2]):
                    row=json.loads((ROOT/p['path']).read_text())
                    if i==1:
                        if mutation=='duplicate': row['repeats'][0]['repeat']=0
                        elif mutation=='source': row['source_sha256']['mutated']='0'*64
                        elif mutation=='data': row['data_sha256']='0'*64
                        elif mutation=='settings': row['args']['payload']+=1
                        else: row['primary_mode']='argmax'
                    (out/Path(p['path']).name).write_text(json.dumps(row))
                collector.__globals__['ROOT']=fake
                try: collector()
                except AssertionError: pass
                else: raise AssertionError('Invalid official repeat batch accepted: '+mutation)
                finally: collector.__globals__['ROOT']=original_root
        passed.append('five official repeat identity/source/data/settings/mode mutations rejected')
        pages=score['page'](data,public); rows=pages[2][1][1]
        return dict(status='completed', checks=passed, checks_passed=len(passed),
                    wins_against_saved_references=sum(r[3]=='WIN' for r in rows), comparisons=rows,
                    official_mg_repeats=public['mg']['n'], manifest=str(Path(manifest).relative_to(ROOT)),
                    source_sha256={str(Path(p).relative_to(ROOT)):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in [__file__, ROOT/'report/scoreboard.py', ROOT/'report/native_language_batched_appendix.py',
                                             ROOT/'report/public_benchmarks_appendix.py', ROOT/'experiments/reference_window_rescore.py']},
                    native_forward_calls=0, backward_calls=0, optimizer_updates=0,
                    scope='Standard-library publication/protocol checks only; numerical reference rescore remains unrun.')
    finally: builtins.__import__=original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    output=ROOT/args.output
    if output.exists(): raise ValueError('Unique immutable check output required')
    result=check(ROOT/args.manifest)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_passed','wins_against_saved_references','official_mg_repeats')}))
