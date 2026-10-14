"""Acquire checksum-verified August-only B10 data; September is never opened."""
import datetime,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
from prepare import convert,RAW,OUT
from b10_depth_tpp_v1 import TRAIN,VAL,windows
ROOT=Path(__file__).resolve().parents[2]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    sources={}; identities={}; sets={}
    for split,days,stride,want in [('train',TRAIN,20,801),('dev',VAL,10,756)]:
        sets[split]=[];identities[split]=[]
        for day in days:
            name=f'BTCUSDT-aggTrades-{day}.zip'
            for suffix in ('.CHECKSUM',''):
                path=RAW/(name+suffix)
                if not path.exists():
                    url='https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/'+name+suffix
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temporary=path.with_name(path.name+'.downloading')
                    urllib.request.urlretrieve(url,temporary);temporary.replace(path)
            checksum=(RAW/(name+'.CHECKSUM')).read_text().split()[0]
            assert digest(RAW/name)==checksum,name
            if not (OUT/f'{day}.npz').exists():convert(day)
            z=np.load(OUT/f'{day}.npz');t=z['t'];m=z['mark']
            assert np.all(np.diff(t)>=0) and np.all((m>=0)&(m<6))
            sources[str((RAW/name).relative_to(ROOT))]=checksum
            sources[str((OUT/f'{day}.npz').relative_to(ROOT))]=digest(OUT/f'{day}.npz')
            ws=windows(day,stride);sets[split].extend(ws)
            for index,(tt,mm) in enumerate(ws):
                w=index*stride
                identities[split].append(dict(day=str(day),window=w,start_event=w*1024,
                    source_window_sha256=hashlib.sha256(t[w*1024:(w+1)*1024].tobytes()+m[w*1024:(w+1)*1024].tobytes()).hexdigest()))
        assert len(sets[split])==want,(split,len(sets[split]),want)
    record=dict(status='completed',battle='B10',source_sha256=sources,identities=identities,
                train_windows=801,dev_windows=756,dev_targets=677376,september_opened=False)
    out=ROOT/'experiments/results/market/aws_b10_matched_data_20261014_v1.json'
    if out.exists():assert json.loads(out.read_text())==record
    else:out.write_text(json.dumps(record,indent=2)+'\n')
    return sets,record
