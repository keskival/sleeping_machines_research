"""B1/R1 component gate: native forward packets teach versioned transport."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import torch
from reciprocal_probe_contract import ROOT, TemporalMemoryLayer
from reciprocal_transport import DirectionalPacket, ReciprocalTransport


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--tag',required=True); a=p.parse_args()
    gate=ROOT/'experiments/results/credit/aws_reciprocal_probe_contract_20261010T2149Z.json'
    g=json.loads(gate.read_text()); assert g['status']=='completed'
    for s,h in g['source_sha256'].items(): assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(212);start=time.monotonic()
    layer=TemporalMemoryLayer(8,4,0.0).eval();x=torch.randn(16);dt=torch.tensor([[0.0,.37]])
    def f(z): return layer(z.reshape(1,2,8),dt).flatten()
    context='native-frozen-two-event-prefix'; basis=torch.eye(16)*4;packets=[]
    for v in basis:
        _,jv=torch.func.jvp(f,(x,),(v,));packets.append(DirectionalPacket.capture(context,0,v,jv))
    target=torch.stack([packet.response/4 for packet in packets],1).T
    transport=ReciprocalTransport(torch.zeros(16,16),context=context,version=0)
    for packet in packets:
        for cotangent in basis: transport.teach(packet,cotangent,step=1/256)
    error=float((transport.matrix-target).abs().max());assert error<1e-12
    stale_rejected=wrong_context_rejected=False
    try: transport.predict(torch.ones(16),context=context,version=1)
    except ValueError: stale_rejected=True
    try: transport.teach(DirectionalPacket.capture('other-prefix',0,basis[0],packets[0].response),basis[0],step=1/256)
    except ValueError: wrong_context_rejected=True
    assert stale_rejected and wrong_context_rejected
    probe=torch.randn(16);parity=float((transport.predict(probe,context=context,version=0)-target@probe).abs().max());assert parity<1e-12
    sources=['experiments/credit/reciprocal_transport_contract.py','experiments/credit/reciprocal_transport.py',
             'experiments/credit/reciprocal_probe_contract.py','experiments/tpp/race_tpp_v19.py']
    result=dict(status='completed',tag=a.tag,battle='B1/R1',stage='reciprocal_transport_component',
                max_matrix_error=error,max_prediction_error=parity,stale_version_rejected=stale_rejected,
                wrong_context_rejected=wrong_context_rejected,forward_packets=len(packets),local_teaching_updates=transport.updates,
                matrix_elements=transport.matrix.numel(),retained_packet_elements=sum(q.direction.numel()+q.response.numel() for q in packets),
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                gate=dict(path=str(gate.relative_to(ROOT)),sha256=hashlib.sha256(gate.read_bytes()).hexdigest()),
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                scope='Frozen native temporal layer; 16 forward directional packets reused for 256 local outer-product teaching updates. '
                      'Dense matrix/packet prototype, no third autograd teacher for local update. No changing actor, task utility, TEST or efficiency claim.',
                decision='Use versioned local primitive for proposed reciprocal integration; context-conditioned sharing and moving-target tracking still require evidence.')
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json'
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
