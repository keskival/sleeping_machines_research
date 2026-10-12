"""B1 check unchanged query/update objective under batched depth decoding."""
import argparse,hashlib,json,resource,time
import torch
from copy import deepcopy
import run_native_batched_event_optimizer as run
from event_optimizer_credit import EventOptimizerCredit as Reference

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=run.ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);run.extension();run.RUN_WIDTH=32;data=[s for s in run.native.load_split('taxi','train') if len(s[1])>=32];K=1+max(int(m.max()) for _,m in data);actor=run.initial(99910,8,K,1.,[-1.,0.,1.]);run.install_event_hooks(actor);actor.collect_events=True;actor.cut=True;t,m,mask=run.one_block(data[1328]);loss=actor(t,m,mask);c=torch.autograd.grad(loss,actor.terminal)[0];p=run.context(actor,c,loss,t);named=dict(actor.named_parameters());core={n:x for n,x in named.items() if run.core_name(n)};reference=Reference(32,core);batched=run.new_credit(actor,8)
 with torch.no_grad():
  for decoder in reference.decoders.values():decoder.weight.normal_(0,.02);decoder.bias.normal_(0,.01)
 batched.load_state_dict(reference.state_dict());forced=[16,20,24,28];errors={};times={};outputs=[];gradients=[]
 for kind,field in [('reference',reference),('batched',batched)]:
  tick=time.monotonic();direction=field(p,forced=forced);issued,_=field.propose(named,direction);objective=sum(v.square().sum() for v in issued.values())+.1*field.last_logprob;gradients.append(torch.autograd.grad(objective,list(field.parameters()),allow_unused=True));outputs.append(direction.detach());times[kind]=time.monotonic()-tick
 errors['direction']=float((outputs[0]-outputs[1]).abs().max());errors['parameter_gradient']=max(float((x-y).abs().max()) for x,y in zip(*gradients) if x is not None and y is not None);gate=max(errors.values())<1e-10;assert gate,errors
 sources=['experiments/credit/check_batched_event_credit.py','experiments/credit/batched_event_parameter_credit.py','experiments/credit/event_parameter_credit.py','experiments/credit/event_optimizer_credit.py','experiments/credit/run_native_batched_event_optimizer.py','experiments/credit/NATIVE_BATCHED_EVENT_OPTIMIZER_PROTOCOL.md'];record=dict(status='completed',battle='B1',tag=a.tag,gate=gate,errors=errors,unreplicated_kernel_wall_s=times,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={n:hashlib.sha256((run.ROOT/n).read_bytes()).hexdigest() for n in sources},scope='Unchanged width32depth8 event credit: forced query output and full proposal-plus-policy parameter gradient parity. One timing is diagnostic only; complete native fits decide efficiency.');out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
