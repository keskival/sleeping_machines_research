"""Non-fitting finite-state contracts for pairwise posterior credit and MH."""
import math


def transition(rho, proposal):
    n=len(rho); out=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            out[i][j]=proposal[j]*min(1.,rho[j]*proposal[i]/(rho[i]*proposal[j]))
        out[i][i]=1-sum(out[i])
    return out


def advance(distribution, kernel):
    return [sum(distribution[i]*kernel[i][j] for i in range(len(distribution))) for j in range(len(distribution))]


def pair_gradient(rho, logits, pair_weights):
    """Expected pair BCE gradient, frozen positive pair weights."""
    g=[0.0]*len(rho)
    for i,j,w in pair_weights:
        pred=1/(1+math.exp(logits[i]-logits[j])); target=rho[j]/(rho[i]+rho[j])
        g[j]+=w*(pred-target);g[i]-=w*(pred-target)
    return g


def audit():
    rows=[];max_balance=0.;max_stationary=0.
    for rho,q in [([.8,.2],[.2,.8]),([.6,.3,.1],[.05,.15,.8]),([.6,.3,.1],[.6,.3,.1])]:
        kernel=transition(rho,q)
        for i in range(len(rho)):
            assert abs(sum(kernel[i])-1)<1e-14
            for j in range(len(rho)):
                max_balance=max(max_balance,abs(rho[i]*kernel[i][j]-rho[j]*kernel[j][i]))
        stationary=advance(rho,kernel)
        max_stationary=max(max_stationary,max(abs(a-b) for a,b in zip(stationary,rho)))
        dist=q[:];trajectory=[]
        for step in range(33):
            if step in (0,1,2,4,8,16,32):trajectory.append(dict(steps=step,tv=0.5*sum(abs(a-b) for a,b in zip(dist,rho))))
            dist=advance(dist,kernel)
        pairs=[(i,j,1.) for i in range(len(rho)) for j in range(i+1,len(rho))]
        assert max(abs(v) for v in pair_gradient(rho,[math.log(r) for r in rho],pairs))<1e-14
        g=pair_gradient(rho,[math.log(r) for r in q],pairs)
        rows.append(dict(posterior=rho,proposal=q,kernel=kernel,mixing=trajectory,pair_logit_gradient=g))
    assert max_balance<1e-14 and max_stationary<1e-14
    # Two-cause example: one pair supplies nonzero verdict learning even though
    # one self-normalized proposal supplies zero expected q-wake gradient.
    assert abs(rows[0]['pair_logit_gradient'][0]+.6)<1e-14
    assert rows[0]['mixing'][1]['tv']>0.4 # one MH step is NOT an exact posterior sample
    return dict(status='completed',scope='finite-state mathematical contracts, no fitting',
                max_detailed_balance_error=max_balance,max_stationarity_error=max_stationary,examples=rows,
                depth2_h8_leading_training_macs=4064,dense_leading_training_macs=17280,
                work_scope='3 router + 5 expert + 3 credit forward-equivalent linear MACs; optimizer/nonlinear work excluded')


if __name__=='__main__':
    import json
    print(json.dumps(audit(),indent=2))
