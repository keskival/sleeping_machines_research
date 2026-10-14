"""R1 future learner using the source-bound production Bernoulli teaching step.

The kernel is extracted verbatim from the frozen driver, avoiding a second
hand-maintained policy. Corpus boundaries/evaluation remain caller-owned.
"""
import copy
import hashlib
from native_token_future_world import NativeTokenFutureWorld, native, ROOT

DRIVER = ROOT/'experiments/bernoulli_teacher_token_language.py'
EXPECTED_SHA256 = '85b82441775b74573db2ad86459443fca9d0fc0e3e40ac205a2c5fd652db8b45'


def production_kernel():
    source = DRIVER.read_text()
    if hashlib.sha256(DRIVER.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Production driver changed; review and rebind replay policy')
    start = source.index('        use_future = bool(')
    end = source.index('        receiver_grad = {}', start)
    body = source[start:end]
    function = 'def kernel(model,opt,state,gen,teacher_gen,a,step,buf):\n'
    function += '    n=buf.shape[1]-1\n    opt.zero_grad(set_to_none=True)\n'
    function += '    teacher_calls=teacher_replay_tokens=teacher_scored_targets=0\n'
    function += '\n'.join(line[4:] for line in body.splitlines())+'\n'
    function += '    opt.step()\n'
    function += '    return token_nll.detach().clone(),native_detach(state),future_audit,(teacher_calls,teacher_replay_tokens,teacher_scored_targets)\n'
    namespace = dict(native.__dict__, native_detach=native.detach)
    exec(compile(function, str(DRIVER)+'::future-step', 'exec'), namespace)
    return namespace['kernel']


class NativeTokenTeacherWorld(NativeTokenFutureWorld):
    def __init__(self, model, optimizer, state, generator, args, *, teacher_generator=None, next_step=1):
        super().__init__(model, optimizer, state, generator, args)
        self.teacher_generator=native.torch.Generator()
        if teacher_generator is None:self.teacher_generator.manual_seed(args.seed+6)
        else:self.teacher_generator.set_state(teacher_generator.get_state().clone())
        self.next_step=next_step
        self.kernel=production_kernel()
        self.teacher_work=[0,0,0]
        self.last_audit=None

    def advance(self, inputs, targets, *, force_site=None):
        if inputs.shape != targets.shape or not native.torch.equal(inputs[:,1:],targets[:,:-1]):
            raise ValueError('Production replay requires a contiguous shifted token window')
        if inputs.shape[0] != self.args.lanes:
            raise ValueError('Replay lane count differs from declared policy')
        buf=native.torch.cat((inputs,targets[:,-1:]),dim=1)
        self.model.train();self.model.force_site=force_site
        try:
            losses,state,audit,work=self.kernel(self.model,self.optimizer,self.state,
                self.generator,self.teacher_generator,self.args,self.next_step,buf)
            self.state=state;self.last_audit=audit
            self.teacher_work=[a+b for a,b in zip(self.teacher_work,work)]
            self.next_step+=1;self.updates+=1
            self.scored_targets+=targets.numel();self.own_gradient_targets+=targets.numel()
            return losses.reshape_as(targets)
        finally:
            self.model.force_site=self.model.suppress_site=None
            self.model.shadow_mode=False;self.model.replay_window=None
