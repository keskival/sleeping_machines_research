"""R1/B1 readonly paired effects and finite-pool decision certificates."""
import argparse,hashlib,json,time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREFIX='curie_averaged_credit_v3_20261010T1720Z'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();rows=[];inputs={}
    for seed in (170,171,172):
        path=ROOT/'experiments/results/credit'/f'{PREFIX}_pilot_s{seed}.json';d=json.loads(path.read_text());assert d['status']=='completed';inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        y=np.array(d['metrics']['dev_branch_mean_advantages']);rng=np.random.default_rng(seed);idx=rng.integers(0,len(y),(2000,len(y)))
        for history in ('isolated','connected'):
            arms={v['labels']:v for v in d['metrics']['arms'] if v['history']==history};certs={};errors={};chosen={}
            for label,v in arms.items():
                pred=np.array(v['dev_predictions']);choice=np.argmin(pred,axis=1);actual=y[np.arange(len(y)),choice];regret=actual-y.min(axis=1);error=np.abs(pred-y).max(axis=1)
                assert np.all(regret<=2*error+1e-12)
                sorted_y=np.sort(y,axis=1);gap=sorted_y[:,1]-sorted_y[:,0];certified=gap>2*error
                assert np.all(regret[certified]<1e-12)
                certs[label]=dict(mean_pool_regret=float(regret.mean()),mean_maximum_value_error=float(error.mean()),certified_optimal_fraction=float(certified.mean()),observed_optimal_fraction=float((regret<1e-12).mean()))
                errors[label]=((pred-y)**2).mean(axis=1);chosen[label]=actual
            mse=errors['single']-errors['averaged'];utility=chosen['single']-chosen['averaged']
            rows.append(dict(seed=seed,history=history,averaging_mse_reduction=float(mse.mean()),averaging_mse_reduction_bootstrap95=np.quantile(mse[idx].mean(axis=1),[.025,.975]).tolist(),averaging_future_loss_reduction=float(utility.mean()),averaging_future_loss_reduction_bootstrap95=np.quantile(utility[idx].mean(axis=1),[.025,.975]).tolist(),certificates=certs))
    source='experiments/credit/audit_future_credit_decisions.py';result=dict(status='completed',tag=a.tag,battle='R1/B1',training=False,metrics=rows,input_sha256=inputs,source_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},wall_s=time.monotonic()-start,scope='Finite DEV continuation averages, shared contexts and teacher per seed. Certificates compare predictions to observed sample means, not population future utility. Bootstrap is conditional prefix uncertainty. No new scoring, teacher replay, fitting or test access.')
    output=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
