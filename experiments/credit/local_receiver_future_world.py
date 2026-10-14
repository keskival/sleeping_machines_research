"""R1 replay of the integrated allocated sparse teacher and receiver-local AdamW.
No live learner, allocator, numeric state or RNG is advanced by cloning.
"""
import copy,hashlib
from native_token_teacher_world import native,ROOT
import local_clip_receiver_token_language as integrated
from sleeping_machines.depth_credit_allocator import DepthCreditAllocator
from sleeping_machines.learning.clone_local_receiver_learner import clone_local_receiver_learner as clone_receiver_learner
DRIVER=ROOT/'experiments/local_clip_receiver_token_language.py'
EXPECTED_SHA256='2e2407f154b0897aa539343cce89579914b0c3ffac80d38b5690f5603202d414'

def kernel():
    if hashlib.sha256(DRIVER.read_bytes()).hexdigest()!=EXPECTED_SHA256:
        raise ValueError('Integrated driver changed; review/rebind future policy')
    source=DRIVER.read_text();begin=source.index('        use_future = bool(');end=source.index('        receiver_grad = {}',begin)
    body='\n'.join(line[4:] for line in source[begin:end].splitlines())
    function='def step(model,opt,state,gen,teacher_gen,allocator,a,step,buf):\n'
    function+='    n=buf.shape[1]-1\n    opt.zero_grad(set_to_none=True)\n    teacher_calls=teacher_replay_tokens=teacher_scored_targets=0\n'
    function+=body+'\n    opt.step()\n    return token_nll.detach().clone(),detach(state),future_audit,(teacher_calls,teacher_replay_tokens,teacher_scored_targets)\n'
    namespace=dict(integrated.__dict__);exec(compile(function,str(DRIVER)+'::future-step','exec'),namespace)
    return namespace['step']

class LocalReceiverFutureWorld:
    def __init__(self,model,optimizer,state,generator,args,*,teacher_generator=None,next_step=1,allocator=None):
        self.model,self.optimizer=clone_receiver_learner(model,optimizer)
        self.state=integrated.clone_numeric_state(state);self.args=copy.deepcopy(args)
        self.generator=native.torch.Generator().set_state(generator.get_state().clone())
        self.teacher_generator=native.torch.Generator()
        if teacher_generator is None:self.teacher_generator.manual_seed(args.seed+6)
        else:self.teacher_generator.set_state(teacher_generator.get_state().clone())
        self.allocator=copy.deepcopy(allocator) if allocator is not None else DepthCreditAllocator(args.depth,args.heads,args.credit_window)
        if self.allocator.last is not None:raise ValueError('Clone only after allocation feedback has matured')
        self.next_step=next_step;self.teacher_work=[0,0,0];self.updates=0
        self.scored_targets=self.own_gradient_targets=0;self.last_audit=None;self.kernel=kernel()
    def advance(self,inputs,targets,*,force_site=None):
        if inputs.shape!=targets.shape or not native.torch.equal(inputs[:,1:],targets[:,:-1]):raise ValueError('Contiguous shifted token window required')
        if inputs.shape[0]!=self.args.lanes:raise ValueError('Wrong lane count')
        self.model.train();self.model.force_site=force_site
        try:
            losses,state,audit,work=self.kernel(self.model,self.optimizer,self.state,self.generator,self.teacher_generator,self.allocator,self.args,self.next_step,native.torch.cat((inputs,targets[:,-1:]),1))
            self.state=state;self.last_audit=audit;self.teacher_work=[a+b for a,b in zip(self.teacher_work,work)]
            self.next_step+=1;self.updates+=1;self.scored_targets+=targets.numel();self.own_gradient_targets+=targets.numel()
            return losses.reshape_as(targets)
        finally:
            self.model.force_site=self.model.suppress_site=None;self.model.shadow_mode=False;self.model.replay_window=None
