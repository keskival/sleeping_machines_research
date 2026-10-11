"""B4 multi-position causal prefix equivalence for portable one-core bundles."""
import argparse,json,time,resource,random
import torch
from check_single_readout_serving import adapter,ROOT,digest,restore
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    proof=ROOT/'experiments/results/tpp_b4/aws_single_readout_serving_20261011_v1.json';r=json.loads(proof.read_text());assert r['status']=='completed';dev,K=adapter.load('github_filtered',1,'val');inputs={str(proof.relative_to(ROOT)):digest(proof)};rows=[]
    for item in r['rows']:
        path=ROOT/item['bundle'];assert digest(path)==item['bundle_sha256'];inputs[item['bundle']]=digest(path);model=restore(torch.load(path,weights_only=True));error=0.;checks=0
        for seq in dev:
            t,m,mask=next(adapter.base.batches([seq],1,False,random.Random(0)));lt,lp=model.event_distributions(t,m,mask);n=t.shape[1]
            for end in sorted(set([2,max(2,n//3),max(2,2*n//3),n])):
                pt,pp=model.event_distributions(t[:,:end],m[:,:end],mask[:,:end]);error=max(error,float((pt-lt[:,:end-1]).abs().max()),float((pp-lp[:,:end-1]).abs().max()));checks+=1
        assert error<1e-11;rows.append(dict(seed=item['seed'],prefix_checks=checks,maximum_prefix_error=error))
    sources=['experiments/tpp/check_single_readout_prefixes.py','experiments/tpp/check_single_readout_serving.py','experiments/tpp/single_mark_readout.py','experiments/tpp/nonlinear_single_mark_readout.py'];out=dict(status='completed',tag=a.tag,battle='B4',rows=rows,source_sha256={s:digest(ROOT/s) for s in sources},input_sha256=inputs,wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Same Github DEV portable bundles, multiple prefix lengths across all sequences/three seeds, no fitting or TEST.')
    with (ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))
if __name__=='__main__':main()
