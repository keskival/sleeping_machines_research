"""R1 inspect learned consequence risk on preserved acquisition/scoring records."""
import argparse,hashlib,json,time
import numpy as np
import torch
from check_token_forward_credit import ROOT
from measure_token_forward_credit import verify
from fit_forward_credit_predictor import new_critic,predict
PARENT='curie_forward_credit_predictor_20261011_v1_fit'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    parent=verify(PARENT);cp=ROOT/parent['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==parent['checkpoint_sha256'];saved=torch.load(cp,weights_only=True);rows=[]
    with torch.no_grad():
        for arm,state in saved['critics'].items():
            model=new_critic(saved['train_records'][0]);model.load_state_dict(state)
            for split,records in [('train',saved['train_records']),('dev',saved['dev_records'])]:
                for i,record in enumerate(records):
                    h=predict(model,record);t=record['approximate'] if split=='train' else record['costs']-record['costs'][0]
                    rows.append(dict(arm=arm,split=split,prefix=i,mse=float((h-t).square().mean()),zero_mse=float(t.square().mean()),prediction_rms=float(h.square().mean().sqrt()),target_rms=float(t.square().mean().sqrt()),dot=float((h*t).mean()),cosine=float((h*t).sum()/(h.norm()*t.norm()).clamp_min(1e-30))))
    summary={f'{arm}_{split}':{key:float(np.mean([r[key] for r in rows if r['arm']==arm and r['split']==split])) for key in ['mse','zero_mse','prediction_rms','target_rms','dot','cosine']} for arm in saved['critics'] for split in ['train','dev']}
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,parent=PARENT,summary=summary,rows=rows,wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/diagnose_forward_credit_predictor.py']},scope='Reused training approximate labels and exact private development costs; no fitting/new future acquisition or confirmation.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
