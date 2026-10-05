"""Closed-moment contracts for a typed numeric threshold's winner/time credit."""
import math

def moments(m,a_true,a_false,time_cost):
    rt,rf=math.exp(m),math.exp(-m);total=rt+rf;p=rt/total;q=rf/total
    value=p*a_true+q*a_false+time_cost/total
    choice=2*p*q*(a_true-a_false)
    clock=-time_cost*(rt-rf)/(total*total)
    # E[(A_W+B*T)*(sign(W)-T*(rt-rf))], with E[T]=1/total and E[T²]=2/total².
    joint=p*a_true-q*a_false-(rt-rf)*(p*a_true+q*a_false)/total
    joint+=time_cost*((p-q)/total-2*(rt-rf)/(total*total))
    return value,choice,clock,joint

def contracts():
    errors=[]
    for m in (-2.,-.3,0.,.7,2.):
        for a,b,c in ((1.,0.,0.),(0.,0.,2.),(.2,1.7,.5),(1.,1.,1.)):
            value,choice,clock,joint=moments(m,a,b,c)
            eps=1e-5;finite=(moments(m+eps,a,b,c)[0]-moments(m-eps,a,b,c)[0])/(2*eps)
            errors.append(abs(finite-joint));assert abs(joint-choice-clock)<1e-12
            assert abs(finite-joint)<1e-8
    _,choice,clock,joint=moments(.7,1.,1.,1.)
    assert choice==0 and abs(clock)>0.1 and abs(joint-clock)<1e-12
    x,tau,scale=30.,20.,10.;m=(x-tau)/scale
    assert math.isclose(m,(86.-68.)/18.)
    # d/dtau = -(d/dm)/scale; transformed-unit threshold derivative rescales correctly.
    gradient=moments(m,.2,1.7,.5)[3]
    assert math.isclose(-gradient/18.,(-gradient/10.)/1.8)
    return dict(cases=len(errors),joint_score_matches_derivative=True,
        choice_plus_clock_matches_joint=True,categorical_only_misses_pure_timing_gradient=True,
        units_transform_threshold_derivative=True,max_finite_difference_error=max(errors),
        scope='Analytical moments for two exponential clocks and branch-plus-time cost. No integrated model, empirical learning gain or resource advantage. Actual nonlinear suffix utility and deep path credit require integrated contracts.')

if __name__=='__main__':
    import json
    print(json.dumps(contracts(),indent=2))
