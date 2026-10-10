"""R1/B1 credit generalization prerequisite: relabeling equivariance.

The task and forward parameters must be transformed together. A hard-wired
coordinate-sharing symmetry is a correctness contract, not learned grokking.
"""
import argparse
import hashlib
import json
import time

import torch
from torch.func import functional_call
from reciprocal_native_v1 import (ROOT, DT, Credit, Update, batch,
                                  flat, native_pair, parameters)

from reciprocal_native_v4 import actor_evidence


def relabel(values, permutation, K=8, d=8, dv=2, clocks=4):
    inv=torch.argsort(permutation);out=dict(values)
    for name,value in values.items():
        if name=='model.embed.weight':out[name]=value[inv]
        elif name=='model.clock.weight':
            slots=value[:,d:].reshape(value.shape[0],K,dv)[:,inv].reshape(value.shape[0],K*dv)
            out[name]=torch.cat((value[:,:d],slots),-1)
        elif name.startswith('model.mark_ctx.'):
            out[name]=value.reshape(clocks,K,*value.shape[1:])[:,inv].reshape(value.shape)
    return out


def contracts():
    rows=[]
    permutation=torch.tensor([2,0,3,1,6,4,7,5])
    for depth in (2,4,8):
        full,local=native_pair(depth,166);p=parameters(full);q=relabel(p,permutation)
        t,m=batch(166);changed=(t,permutation[m])
        original_loss,gf,gl,packet=actor_evidence(full,local,p,(t,m))
        transformed_loss,gfq,glq,packetq=actor_evidence(full,local,q,changed)
        loss_error=abs(original_loss-transformed_loss)
        gradient_error=float((flat(relabel(gf,permutation))-flat(gfq)).abs().max())
        local_error=float((flat(relabel(gl,permutation))-flat(glq)).abs().max())
        assert max(loss_error,gradient_error,local_error)<1e-10
        credit=Credit()
        residual={k:gf[k]-gl[k] for k in p};residualq={k:gfq[k]-glq[k] for k in q}
        state=credit.observe({},residual,packet,.4)
        stateq=credit.observe({},residualq,packetq,.4)
        predicted=credit.predict(p,gl,packet,state,.6)
        predictedq=credit.predict(q,glq,packetq,stateq,.6)
        credit_error=float((flat(relabel(predicted,permutation))-flat(predictedq)).abs().max())
        update=Update()
        updated,ostate=update.step(p,predicted,{},1.)
        updatedq,ostateq=update.step(q,predictedq,{},1.)
        update_error=float((flat(relabel(updated,permutation))-flat(updatedq)).abs().max())
        assert max(credit_error,update_error)<1e-10
        # Removing available information can make exact credit unidentifiable.
        # H is constant, target +/-1 equiprobable: best squared-error risk is 1.
        # Adding the observed bit makes the target identifiable and risk zero.
        rows.append(dict(depth=depth,forward_loss_equivariance_error=loss_error,
                         full_gradient_equivariance_error=gradient_error,local_gradient_equivariance_error=local_error,
                         learned_credit_function_equivariance_error=credit_error,
                         learned_optimizer_update_equivariance_error=update_error))
    return dict(native_relabeling=rows,information_witness=dict(without_cause_bit_minimum_mse=1.,with_cause_bit_minimum_mse=0.),
                scope='Task-preserving key/value relabeling with corresponding parameter action; exact structure contract, no trained transfer or grokking claim')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=ROOT/'experiments/results/credit'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.set_default_dtype(DT);torch.manual_seed(166);start=time.monotonic()
    result=contracts()
    paths=['experiments/credit/check_credit_equivariance_v2.py','experiments/credit/reciprocal_native_v1.py',
           'experiments/credit/reciprocal_native_v4.py',
           'experiments/tpp/recall_tpp_v4.py','experiments/tpp/race_tpp_v5.py']
    record=dict(status='completed',tag=a.tag,battle='R1/B1',training=False,metrics=result,wall_s=time.monotonic()-start,
                decision='Whether task symmetries are respected before fitting a transferable credit learner',
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out)
    print('RESULT',json.dumps(record),flush=True)


if __name__=='__main__':main()
