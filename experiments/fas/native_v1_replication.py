"""Run the recorded seed6 FAS driver, preserving its recipe and source identity."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'experiments/results/fas/curie_fas_v1_native_p32d4_linear_seg128_t1024_e2_20261004T231500Z.json'
FROZEN=ROOT/'experiments/fas/native_v1_seed6_frozen_20261005.py'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    control=json.loads(CONTROL.read_text())
    assert sha(FROZEN)==control['source_sha256']['experiments/fas/native.py']
    for name,digest in control['source_sha256'].items():
        if name!='experiments/fas/native.py':assert sha(ROOT/name)==digest,name
    manifest=json.loads((ROOT/'experiments/queue/curie_fas_v1_replication_20261005_v1.json').read_text())
    for name,digest in manifest['execution_source_sha256'].items():assert sha(ROOT/name)==digest,name
    tag=sys.argv[sys.argv.index('--tag')+1]
    seed=int(sys.argv[sys.argv.index('--seed')+1]);assert seed in (7,8)
    data=ROOT/'experiments/data/fas'/control['args']['data']/'manifest.json'
    expected=json.loads((ROOT/'experiments/results/fas/fas_v1_20261004_manifest.json').read_text())
    actual=json.loads(data.read_text())
    assert actual['args']==expected['args'] and actual['events']==expected['events']
    for split,info in expected['splits'].items():
        assert actual['splits'][split]['runs']==info['runs']
        assert actual['splits'][split]['events']==info['events']
        assert sha(data.parent/(split+'.npz'))==actual['splits'][split]['sha256']
    byte_identity=all(actual['splits'][k]['sha256']==v['sha256'] for k,v in expected['splits'].items())
    namespace={'__file__':str(FROZEN),'__name__':'fas_frozen_driver'}
    exec(compile(FROZEN.read_bytes(),str(FROZEN),'exec'),namespace)
    captured=[]; models=[]; original_scores=namespace['scores']
    def scores(*args,**kwargs):
        value=original_scores(*args,**kwargs)
        captured.append(value[0].copy())
        models[:] = [args[0]]
        return value
    namespace['scores']=scores
    namespace['main']()
    assert len(captured)>=2
    score_path=ROOT/f'experiments/results/fas/{tag}_scores.npz'
    if score_path.exists():raise FileExistsError(score_path)
    np=namespace['np']
    with np.load(data.parent/'test_faulty.npz',allow_pickle=False) as original:
        kinds=original['fault'][:len(captured[-1])]
        faulty_seeds=original['seed'][:len(captured[-1])]
    with np.load(data.parent/'test_clean.npz',allow_pickle=False) as original:
        clean_seeds=original['seed'][:len(captured[-2])]
    np.savez_compressed(score_path,test_clean=captured[-2],test_faulty=captured[-1],prefixes=namespace['PREFIXES'],fault_kind=kinds,test_clean_seed=clean_seeds,test_faulty_seed=faulty_seeds)
    weights=ROOT/f'experiments/results/fas/checkpoints/{tag}_selected.pt'
    weights.parent.mkdir(parents=True,exist_ok=True)
    if weights.exists():raise FileExistsError(weights)
    namespace['torch'].save(models[0].state_dict(),weights)
    output=ROOT/f'experiments/results/fas/{tag}.json'
    result=json.loads(output.read_text())
    for key,value in control['args'].items():
        if key not in ('tag','seed','max_windows','eval_runs','trace_windows'):assert result['args'][key]==value,key
    result['source_sha256']['experiments/fas/native.py']=sha(FROZEN)
    result['selected_weights']=str(weights.relative_to(ROOT))
    result['selected_weights_sha256']=sha(weights)
    result['replication']=dict(control=str(CONTROL.relative_to(ROOT)),control_sha256=sha(CONTROL),
        frozen_driver=str(FROZEN.relative_to(ROOT)),execution_source_sha256=manifest['execution_source_sha256'],
        scores_path=str(score_path.relative_to(ROOT)),scores_sha256=sha(score_path),historical_dataset_bytes_identical=byte_identity,
        scope='Exact recorded driver and recorded dependency hashes; additional current dependency hashes frozen. Historical unrecorded dependencies are not independently verified. FAS v1 replication, not sealed v2 confirmation. Original first-window extrapolated work convention retained.')
    output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
