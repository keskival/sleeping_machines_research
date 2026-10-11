"""Exact conditional gradient variance in the real connected route controller."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM
from measure_token_forward_credit import verify
from token_issued_credit_geometry import issue_geometry
from token_issued_route_features import RouteController


def artifact(result):
    path=ROOT/result['checkpoint'];assert hashlib.sha256(path.read_bytes()).hexdigest()==result['checkpoint_sha256']
    return torch.load(path,weights_only=True)

def flat(gradients):return torch.cat([g.flatten() for g in gradients])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    acquisition=verify('curie_long_token_credit_20261011_v1_measure');saved=artifact(acquisition)['prefixes']
    transport=verify('curie_token_optimizer_transport_20261011_v1_measure');predictions=artifact(transport)['predictions']
    controller=verify('curie_continuous_token_route_dev_20261011_v2_fit');state=artifact(controller)['actors']['reference']
    config=json.loads((ROOT/SOURCE).read_text());c=config['args'];m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(ROOT/config['checkpoint'],weights_only=True));m.requires_grad_(False);m.eval();rows=[]
    for i,s in enumerate(saved):
        issued=issue_geometry(m,s['tokens'][:64],s['tokens'][:64]);actor=RouteController(len(issued['context']),issued['descriptors'].shape[1]);actor.load_state_dict(state)
        assert issued['receivers']==s['receivers']
        logits=actor(issued['context'],issued['descriptors']);pi=.9*logits.softmax(0)+.1/len(logits);params=list(actor.parameters())
        cost=s['true_scores'].mean(1);d=cost-cost[0]
        truth=flat(torch.autograd.grad((pi*d).sum(),params,retain_graph=True))
        score=torch.stack([flat(torch.autograd.grad(pi[a].log(),params,retain_graph=True)) for a in range(len(pi))])
        diagonal=next(p['scores'] for p in predictions if p['prefix']==i and p['curvature']=='diagonal').mean(1)
        static=s['static_scores'].mean(1);reference=s['projected_scores'].mean(1)
        for name,h in [('zero',torch.zeros_like(d)),('static',static-static[0]),('reference',reference-reference[0]),('transport_diagonal',diagonal-diagonal[0])]:
            model=flat(torch.autograd.grad((pi*h).sum(),params,retain_graph=True))
            gradients=model[None]+(d-h)[:,None]*score
            mean=(pi.detach()[:,None]*gradients).sum(0)
            error=float((mean-truth).abs().max());assert error<1e-12
            variance=float((pi.detach()[:,None]*(gradients-truth[None]).square()).sum())
            rows.append(dict(prefix=i,arm=name,conditional_parameter_gradient_variance=variance,mean_gradient_error=error))
    summary=[dict(arm=name,mean_conditional_parameter_gradient_variance=float(np.mean([r['conditional_parameter_gradient_variance'] for r in rows if r['arm']==name]))) for name in ['zero','static','reference','transport_diagonal']]
    paths=['experiments/credit/measure_token_controller_transport_variance.py','experiments/credit/token_issued_credit_geometry.py','experiments/credit/token_issued_route_features.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,rows=summary,per_prefix=rows,controller_parameters=sum(p.numel() for p in actor.parameters()),source_controller_result=controller['tag'],source_labels_sha256=acquisition['checkpoint_sha256'],source_transport_sha256=transport['checkpoint_sha256'],whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='One fixed trained connected37,377parameter controller, actual native50257word prefixes, four paid48target all-route world cost arrays. Exact enumeration of issued0.9softmax+0.1uniform mean and conditional controller-parameter covariance trace, full factual residual. No actor update/future-distribution variance/native-carrier gradient or learning-utility claim; controller fitting/source acquisition/pretraining additional.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
