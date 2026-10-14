"""Trajectory counters for raw teacher demand and actual gradient clipping."""
import math


class CreditUpdateDiagnostics:
    def __init__(self):
        self.steps=0;self.clipped_steps=0;self.teacher_steps=0
        self.gradient_norm_sum=0.;self.teacher_score_norm_sum=0.
        self.teacher_score_norm_max=0.

    def observe(self, gradient_norm, teacher_score_norm=None):
        if not math.isfinite(gradient_norm) or gradient_norm<0:
            raise ValueError('Finite nonnegative preclip gradient norm required')
        if teacher_score_norm is not None and (not math.isfinite(teacher_score_norm) or teacher_score_norm<0):
            raise ValueError('Finite nonnegative raw teacher score norm required')
        self.steps+=1;self.clipped_steps+=int(gradient_norm>1.)
        self.gradient_norm_sum+=gradient_norm
        if teacher_score_norm is not None:
            self.teacher_steps+=1;self.teacher_score_norm_sum+=teacher_score_norm
            self.teacher_score_norm_max=max(self.teacher_score_norm_max,teacher_score_norm)

    def state_dict(self):
        return dict(vars(self))

    def load_state_dict(self, state):
        if set(state)!=set(vars(self)):
            raise ValueError('Credit diagnostic checkpoint fields changed')
        if not 0<=state['clipped_steps']<=state['steps'] or not 0<=state['teacher_steps']<=state['steps']:
            raise ValueError('Invalid trajectory counters')
        if any(not math.isfinite(v) or v<0 for v in state.values()):
            raise ValueError('Invalid trajectory diagnostic value')
        self.__dict__.update(state)

    def packet(self):
        return dict(**self.state_dict(),
            clipping_fraction=self.clipped_steps/max(1,self.steps),
            mean_preclip_full_gradient_norm=self.gradient_norm_sum/max(1,self.steps),
            mean_raw_teacher_score_gradient_norm=self.teacher_score_norm_sum/max(1,self.teacher_steps),
            scope='Teacher magnitude is the raw route-logit-space surrogate norm, not full parameter or Adam-step norm. Full gradient clipping frequency is measured before clip1; direction, bias and variance are not inferred from these summaries.')
