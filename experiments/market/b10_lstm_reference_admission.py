"""Real TRAIN LSTM full-window pilot; preserve its state/work for budgeted continuation."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from b10_depth_tpp_v1 import TRAIN,windows
from b10_matched_models import MatchedRace
from b10_operator_ledger_v2 import Ledger
from b10_matched_pipeline import update
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();cpu=time.process_time()
    out=ROOT/f'experiments/results/market/{a.tag}.json';assert not out.exists()
    ws=[];hashes={}
    for day in TRAIN:
        p=ROOT/f'data/binance/npz/{day}.npz'
        deadline=time.monotonic()+300
        while not p.exists():
            if time.monotonic()>deadline:raise TimeoutError('Data preparation still pending: '+str(p))
            time.sleep(2)
        ws.extend(windows(day,20));hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(ws)==801
    gaps=np.concatenate([np.diff(t) for t,m in ws]);gaps=gaps[gaps>0]
    torch.manual_seed(0);model=MatchedRace(gaps,'lstm');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4,foreach=False)
    rng=np.random.default_rng(0);order=rng.permutation(len(ws));rows=[ws[i] for i in order[:16]]
    t=torch.tensor(np.stack([r[0] for r in rows]));m=torch.tensor(np.stack([r[1] for r in rows]))
    with Ledger() as ledger:loss=update(model,opt,t,m)
    weights=out.with_suffix('.pilot.pt');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,order=order,cursor=16,seed=0,lr=.001,train_sha256=hashes,ledger=ledger.record()),weights)
    names=['experiments/market/b10_lstm_reference_admission.py','experiments/market/b10_operator_ledger_v2.py','experiments/market/b10_matched_models.py','experiments/market/b10_matched_pipeline.py','experiments/market/b10_depth_tpp_v1.py','experiments/tpp/race_tpp_v5.py']
    source={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names}
    record=dict(status='completed',battle='B10',family='lstm',stage='real_train_full_window_pilot',targets=16*1023,loss=loss,ledger=ledger.record(),training_admitted=False,final_weights=str(weights.relative_to(ROOT)),source_sha256=source,train_sha256=hashes,wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,september_opened=False)
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
