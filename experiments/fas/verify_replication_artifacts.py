"""Read-only score/checkpoint receipt audit; no model import, fit or inference."""
import argparse
from bisect import bisect_left,bisect_right
import hashlib
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(clean,faulty):
    clean=sorted(float(v) for v in clean)
    return sum((bisect_left(clean,float(v))+bisect_right(clean,float(v)))/2 for v in faulty)/(len(clean)*len(faulty))


def verify(result_path):
    result_path=Path(result_path).resolve();r=json.loads(result_path.read_text());assert r['status']=='completed'
    replication=r['replication'];control_path=ROOT/replication['control'];control=json.loads(control_path.read_text())
    assert sha(control_path)==replication['control_sha256']
    for file,digest in replication['execution_source_sha256'].items():assert sha(ROOT/file)==digest,file
    for key,value in control['args'].items():
        if key not in ('tag','seed','max_windows','eval_runs','trace_windows'):assert r['args'][key]==value,key
    assert r['args']['max_windows']==0 and r['args']['eval_runs']==1000
    assert replication['historical_dataset_bytes_identical']
    assert r['selected_epoch']==min(r['curve'],key=lambda row:row['val_clean_nll'])['epoch']
    weights=ROOT/r['selected_weights'];scores=ROOT/replication['scores_path']
    assert sha(weights)==r['selected_weights_sha256'] and sha(scores)==replication['scores_sha256']
    recomputed={};max_error=0.
    with np.load(scores,allow_pickle=False) as arrays:
        clean,faulty=arrays['test_clean'],arrays['test_faulty'];prefixes=arrays['prefixes'];kinds=arrays['fault_kind']
        assert clean.shape==(2000,6) and faulty.shape==(2000,6)
        assert bool(np.isfinite(clean).all()) and bool(np.isfinite(faulty).all())
        assert arrays['test_clean_seed'].shape==(2000,) and arrays['test_faulty_seed'].shape==(2000,)
        assert list(map(int,prefixes))==[32,64,128,256,512,1024]
        names=set(kinds.tolist());assert names=={1,2},names
        for column,prefix in enumerate(prefixes):
            values={'all':auc(clean[:,column],faulty[:,column])}
            for number,name in [(1,'wear_and_tear'),(2,'retry_delay')]:values[name]=auc(clean[:,column],faulty[kinds==number,column])
            for name,value in values.items():
                error=abs(value-r['test_auroc'][str(prefix)][name]);max_error=max(max_error,error)
                assert error<1e-12,(prefix,name,error)
            recomputed[str(prefix)]=values
    return dict(status='completed',result=str(result_path.relative_to(ROOT)),result_sha256=sha(result_path),seed=r['args']['seed'],selected_epoch=r['selected_epoch'],score_shapes=dict(clean=[2000,6],faulty=[2000,6]),source_identity=True,data_byte_identity=True,recipe_identity=True,selected_weight_sha256=sha(weights),scores_sha256=sha(scores),auroc_recomputed=recomputed,auroc_max_error=max_error,scope='Artifact-only verification; no model fit/inference. Dataset seed IDs and fault kinds are audit/evaluation metadata, never model inputs. Exact recorded driver/current frozen dependency closure; historical unrecorded dependency identity is not independently verified. FAS v1 native replication, not sealed v2 confirmation.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    record=verify(a.result);record['producer_sha256']=sha(__file__);out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='auroc_recomputed'}))
