"""R1 fixed-version mean of issued future-loss derivatives."""
import argparse,hashlib,json,time
import torch
from check_token_forward_credit import ROOT
from online_geometric_update_accumulated import UpdateFunction,OnlineUpdateLearner

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(99715)
    function=UpdateFunction(42);learner=OnlineUpdateLearner(function,8,True,feedback_batch=3);h=torch.eye(8);hidden=torch.randn(8);g=torch.randn(8);learner.feedback(g,h,hidden,2.,0)
    gradients=[];captured=[];original=learner.optimizer.step
    def step():
        captured.append([p.grad.clone() for p in function.parameters()]);return original()
    learner.optimizer.step=step
    for i in range(3):
        future=torch.randn(8);gradients.append(torch.autograd.grad(learner.pending,tuple(function.parameters()),future,retain_graph=True))
        learner.feedback(future,h,hidden,2.,i+1)
        assert learner.version==(1 if i==2 else 0)
    error=max(float((seen-torch.stack([row[j] for row in gradients]).mean(0)).abs().max()) for j,seen in enumerate(captured[0]));assert error<1e-12
    assert learner.meta_updates==1 and learner.meta_feedbacks==3
    learner.feedback(g,h,hidden,2.,4);assert learner.accumulated==1;learner.reset();assert learner.pending is None and learner.accumulated==1
    learner.finish();assert learner.meta_updates==2 and learner.accumulated==0 and learner.pending is None
    learner.feedback(g,h,hidden,2.,5);learner.version+=1
    try:learner.feedback(g,h,hidden,2.,6)
    except ValueError:pass
    else:raise AssertionError('Stale graph accepted')
    paths=['experiments/credit/check_online_geometric_update_accumulated.py','experiments/credit/online_geometric_update_accumulated.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,mean_gradient_max_error=error,wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Exact fixed-version average of local issued-update derivatives, delayed commit, reset buffers, partial flush, stale version; synthetic contract, no task quality claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
