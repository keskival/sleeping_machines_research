"""Publish completed native-work ledger and local update bank contracts."""
import fcntl,json
from continue_window_replay import ROOT,sha,wait_result
from continue_unique_language import command


def main():
    packets=[]
    for tag in ('aws_native_work_accounting_20261013_v2','aws_local_credit_banks_contract_20261013_v1','aws_receiver_credit_banks_contract_20261013_v1'):
        d=wait_result(ROOT/f'experiments/results/diagnostics/{tag}.json')
        for name,digest in {**d['source_sha256'],**d.get('input_sha256',{})}.items():assert sha(ROOT/name)==digest,name
        packets.append(d)
    work,leaf,receiver=packets;assert work['exact_instrumented_learning_curve'] and leaf['parameter_error']==receiver['parameter_error']==0
    a=work['accounting'];s=a['optimizer_storage'];native=receiver['storage']
    text=(f"8192-target exact-learning instrumentation records{a['registered_fit_flops']/1e9:.6f}GFLOPs as registered fitting-phase subtotal; "
        f"full Torch operation/payload ledger retains unsupported and unclassified work. Parameter bytes{s['parameter_bytes']}, "
        f"gradient bytes{s['gradient_bytes']}, optimizer bytes{s['optimizer_tensor_bytes']};16global clipping/backward/Adam boundaries. "
        f"Rank diagnostic rounding{work['diagnostic_rank_max_error']:.3g}, learner fingerprints exact. "
        f"Independent leaf and receiver-bank update contracts match three un-clipped ordinary-Adam updates exactly. "
        f"Private/tied/shared mapping{native['banks_by_placement']}; independent versions, stale rejection and isolated receiver update pass. "
        "Credit generation remains factual window BP; these are accounting/update-placement contracts, not learned-credit replacement or measured ASIC energy. "
        "Tensor operands are logical payload exposures, semantic deliveries exclude placement/fanout/handshakes; no complete-FLOP comparison to BPTT is claimed.")
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS native work/update contracts — 13 October.** '),('report/I_SCIENCE.md','Current AWS native training work. ')]:
            p=ROOT/name;parts=[v for v in p.read_text().split('\n\n') if not v.startswith(('**AWS native work/update contracts','Current AWS native training work'))]
            if name.startswith('experiments/'):p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts))
            else:p.write_text('\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record completed native accounting and local receiver update contracts')
        command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish native work accounting and bank-placement evidence')
        command('git','pull','--rebase');command('git','push','origin','main')


if __name__=='__main__':main()
