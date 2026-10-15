"""Checksum-verified October data, acquired only after confirmation preregistration."""
import argparse,hashlib,json,shutil,urllib.request,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];DAY='2026-10-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();protocol=ROOT/'experiments/market/B10_OCTOBER_CONFIRMATION.json';assert protocol.exists();out=ROOT/f'experiments/results/market/{a.tag}.json';assert not out.exists();raw=ROOT/'data/binance';raw.mkdir(parents=True,exist_ok=True);name=f'BTCUSDT-aggTrades-{DAY}.zip';url='https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/'+name
 for suffix in ('.CHECKSUM',''):
  path=raw/(name+suffix)
  if not path.exists():
   temporary=path.with_suffix(path.suffix+'.download')
   with urllib.request.urlopen(url+suffix,timeout=60) as response,temporary.open('wb') as sink:shutil.copyfileobj(response,sink)
   temporary.replace(path)
 expected=(raw/(name+'.CHECKSUM')).read_text().split()[0];assert sha(raw/name)==expected
 with zipfile.ZipFile(raw/name) as archive:
  member=archive.namelist()[0]
  with archive.open(member) as stream:
   first=stream.readline();header=b'price' in first.lower()
   def lines():
    if not header:yield first.replace(b'True',b'1').replace(b'False',b'0')
    for line in stream:yield line.replace(b'True',b'1').replace(b'False',b'0')
   values=np.loadtxt(lines(),delimiter=',',usecols=(1,5,6),dtype=np.float64)
 assert np.isfinite(values).all() and (values[:,0]>0).all();assert np.isin(values[:,2],[0,1]).all();times=values[:,1].astype(np.int64)
 if times[0]<10**14:times*=1000
 order=np.argsort(times,kind='stable');times=times[order];price=values[order,0];side=values[order,2].astype(np.int64);delta=np.diff(price,prepend=price[0]);marks=(3*side+np.where(delta>0,0,np.where(delta<0,2,1))).astype(np.int8);assert len(times)>=65536+128 and (np.diff(times)>=0).all();path=ROOT/f'data/binance/npz/{DAY}.npz';assert not path.exists(),'Do not overwrite existing data';np.savez(path,t=times,mark=marks)
 record=dict(status='completed',battle='B10',day=DAY,events=len(times),source_url=url,archive_sha256=expected,npz_sha256=sha(path),preregistration_sha256=sha(protocol),source_sha256={'experiments/market/b10_shift_data.py':sha(Path(__file__))},scope='First October acquisition after fixed confirmation preregistration; full anonymous public trade day, stable timestamp order, canonical six marks. Price only derives observed mark; models receive no price/quantity features. September unopened.')
 with out.open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
