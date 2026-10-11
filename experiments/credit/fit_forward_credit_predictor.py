"""R1 connected prefix-only future route credit, learned from forward returns."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from measure_token_forward_credit import verify
from token_issued_credit_geometry import issue_geometry
from token_issued_route_features import RouteController
from token_learner_state_features import learner_features
from token_forward_credit_vectorized import packet
from token_optimizer_transport_carried import replay_transport
from token_persistent_credit import joint_replay_carry
from token_exact_world_readouts import replay
from connected_credit_risk import corrected_credit_risk
from token_flop_accounting import TokenFlopCounter
SCALE=.001

def inputs(m,tokens):
    issued=issue_geometry(m,tokens[:64],tokens[:64]);context=torch.cat([issued['context'],learner_features(None,issued['context'],m.embed.embedding_dim)])
    return issued,context

def new_critic(record):
    torch.manual_seed(99510);critic=RouteController(len(record['context']),record['descriptors'].shape[1]);torch.nn.init.zeros_(critic.output.weight);torch.nn.init.zeros_(critic.output.bias);return critic

def predict(critic,record):
    raw=critic(record['context'],record['descriptors']);return SCALE*(raw-raw[0])

def bootstrap(values):
    v=np.array(values);rng=np.random.default_rng(99512);b=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
    return dict(mean_gain=float(v.mean()),paired_prefix_bootstrap95=np.quantile(b,[.025,.975]).tolist(),per_prefix=v.tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);started=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    assert verify('curie_connected_credit_risk_20261011_v1_contract')['gate'];assert verify('curie_token_optimizer_transport_carried_20261011_v1_contract')['gate']
    if a.mode=='fit':assert verify('curie_forward_credit_predictor_20261011_v1_smoke')['gate']
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];cp=ROOT/source['checkpoint']
    m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(cp,weights_only=True));m.requires_grad_(False);m.eval()
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');records=[];counts=dict(actual_worlds=0,actual_updates=0,transported_updates=0);stages={}
    ntrain,ndev=(2,1) if a.mode=='smoke' else (32,8)
    begin=time.monotonic()
    with TokenFlopCounter(display=False) as work:
        for i in range(ntrain):
            offset=c['train_tokens']+393216+128*i;tokens=torch.tensor(np.array(data[offset:offset+112],dtype=np.int64));issued,context=inputs(m,tokens)
            count=len(issued['receivers']);q=torch.full((count,),1/count);A=int(torch.multinomial(q,1,generator=torch.Generator().manual_seed(99520+i)))
            p=packet(m,tokens,index=63);teacher=replay_transport(p,issued['receivers'],issued['actions'],.01,'diagonal')
            exact=joint_replay_carry(p,issued['receivers'],issued['actions'],A,0,.01)
            assert float((teacher['reference_scores']-exact['actual_scores'][1]).abs().max())<1e-10
            t=teacher['scores'].mean(1);t=t-t[0];cost=(exact['actual_scores'][0]-exact['actual_scores'][1]).mean()
            records.append(dict(tokens=tokens,context=context,descriptors=issued['descriptors'],approximate=t,action=A,exact_cost=cost,probabilities=q))
            counts['actual_worlds']+=teacher['actual_worlds']+exact['actual_worlds'];counts['actual_updates']+=teacher['reference_updates']+exact['readout_updates'];counts['transported_updates']+=teacher['transported_updates']
            if (i+1)%8==0:print(json.dumps(dict(stage='acquisition',prefixes=i+1)),flush=True)
    stages['acquisition']=dict(wall_s=time.monotonic()-begin,supported_flops=work.get_total_flops(),**counts)
    critics={};training=[];begin=time.monotonic()
    with TokenFlopCounter(display=False) as work:
        for arm in ['uncorrected','corrected']:
            critic=new_critic(records[0]);opt=torch.optim.Adam(critic.parameters(),lr=.001);losses=[]
            for epoch in range(10):
                order=torch.randperm(ntrain,generator=torch.Generator().manual_seed(99511+epoch)).tolist()
                for i in order:
                    record=records[i];h=predict(critic,record)
                    if arm=='corrected':loss=corrected_credit_risk(h,record['approximate'],record['action'],record['exact_cost'],record['probabilities'])/SCALE**2
                    else:loss=(h-record['approximate']).square().mean()/SCALE**2
                    assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            critics[arm]=critic;training.append(dict(arm=arm,updates=10*ntrain,first_loss=losses[0],last_loss=losses[-1],parameters=sum(v.numel() for v in critic.parameters())))
    stages['fitting']=dict(wall_s=time.monotonic()-begin,supported_flops=work.get_total_flops(),arms=training)
    rows=[];dev=[];begin=time.monotonic();worlds=updates=0
    with TokenFlopCounter(display=False) as work:
        for i in range(ndev):
            offset=c['train_tokens']+(462848 if a.mode=='smoke' else 458752)+128*i;tokens=torch.tensor(np.array(data[offset:offset+112],dtype=np.int64));issued,context=inputs(m,tokens);p=packet(m,tokens,index=63)
            truth=replay(p,issued['receivers'],issued['actions'],.01);costs=truth['scores'].mean(1);contrasts=costs-costs[0];record=dict(context=context,descriptors=issued['descriptors'])
            with torch.no_grad():predictions={arm:predict(critic,record) for arm,critic in critics.items()}
            predictions['zero']=torch.zeros_like(contrasts)
            mse={arm:float((h-contrasts).square().mean()) for arm,h in predictions.items()}
            rows.append(dict(prefix=i,mse=mse,contrast_rms=float(contrasts.square().mean().sqrt()),greedy_regret={arm:float(costs[h.argmin()]-costs.min()) for arm,h in predictions.items()},prediction_rms={arm:float(h.square().mean().sqrt()) for arm,h in predictions.items()},routes=len(costs)))
            dev.append(dict(tokens=tokens,context=context,descriptors=issued['descriptors'],costs=costs,predictions=predictions));worlds+=truth['actual_worlds'];updates+=truth['readout_updates']
    stages['scoring']=dict(wall_s=time.monotonic()-begin,supported_flops=work.get_total_flops(),actual_worlds=worlds,actual_updates=updates)
    comparisons={f'{b}_over_{r}':bootstrap([row['mse'][r]-row['mse'][b] for row in rows]) for b,r in [('corrected','zero'),('corrected','uncorrected'),('uncorrected','zero')]}
    checkpoint=out.with_suffix('.pt');torch.save(dict(train_records=records,dev_records=dev,critics={arm:v.state_dict() for arm,v in critics.items()}),checkpoint)
    paths=['experiments/credit/fit_forward_credit_predictor.py','experiments/credit/token_learner_state_features.py','experiments/credit/connected_credit_risk.py','experiments/credit/FORWARD_CREDIT_PREDICTOR_PROTOCOL.md','experiments/credit/token_issued_credit_geometry.py','experiments/credit/token_optimizer_transport_carried.py','experiments/credit/token_exact_world_readouts.py']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,gate=True,predictive_gate=comparisons['corrected_over_zero']['paired_prefix_bootstrap95'][0]>0,config=dict(train_prefixes=ntrain,scoring_prefixes=ndev,context=64,horizon=48,fit_passes=10,lr=.001,output_scale=SCALE,train_offset=c['train_tokens']+393216,scoring_offset=c['train_tokens']+(462848 if a.mode=='smoke' else 458752)),stages=stages,rows=rows,comparisons=comparisons,wall_s=time.monotonic()-started,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Connected prefix-only48target route-consequence predictor, one development initialization. Cold lexical state, fixed native2layer pretrained model/proper50257GPT2vocabulary. New private TRAIN-tail acquisition/scoring cohorts, no public DEV/TEST or actor/fullnative learning. Approximate transported labels and exact sampled/no-op correction paid; all exact scoring/discovery/copies paid. Source pretraining/contracts/smokes additional; unsupported arithmetic included in wall but omitted from supported FLOPs. Final10pass checkpoint, no scoring selection. Adam feature interface present but constant during this cold-state fit; its utility not measured.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(comparisons=comparisons,predictive_gate=result['predictive_gate'],wall_s=result['wall_s'])),flush=True)
if __name__=='__main__':main()
