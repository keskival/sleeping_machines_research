"""Export measured development points; never fit or extrapolate a scaling law."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def build(prefix):
    provenance={}
    def read(name):
        p=ROOT/name;blob=p.read_bytes();provenance[name]=hashlib.sha256(blob).hexdigest();r=json.loads(blob)
        assert r['status']=='completed';return r
    rows=[]
    for tokens,label,widths in ((8192,'8k',(16,)),(65536,'64k',(16,24,32)),(262144,'256k',(24,))):
        for width in widths:
            tag=f'curie_fixed_batch_tokens_8k_b64_c16_s6_20261005_v1' if label=='8k' else f'curie_data_growth_tokens_{label}_b64_c16_p{width}_s6_20261005_v1'
            r=read(f'experiments/results/token_language/{tag}.json')
            selected=read(f'experiments/results/token_language/{tag}.selection.json')['selected']
            a=r['args'];assert a['train_tokens']==tokens and a['payload']==width and a['credit_window']==16 and a['seed']==6
            assert a['dev_offset']==20971520 and a['dev_tokens']==2048 and a['eval_lanes']==8
            assert r['presentations_total']==2*(tokens-8)
            assert selected['dev_nll']==min(x['dev_nll'] for x in r['curve'])
            rows.append(dict(tokens=tokens,width=width,parameters=r['parameters'],initial=r['curve'][0]['dev_nll'],selected=selected['dev_nll'],final=r['curve'][-1]['dev_nll']))
    repeat=read('experiments/results/token_language/curie_data_growth_tokens_64k_b64_c16_p24_s7_20261005_v1.selection.json')['selected']['dev_nll']
    work=[]
    for path,tokens,width in [('experiments/results/diagnostics/curie_fixed_batch_tokens_8k_c16_work_20261005_v1.json',8192,16),('experiments/results/diagnostics/curie_data_growth_tokens_64k_p24_work_20261005_v1.json',65536,24)]:
        r=read(path);assert r['curve_parity_max_error']<2e-6 and r['fitting']['formula_coverage_complete']
        row=next(x for x in rows if (x['tokens'],x['width'])==(tokens,width))
        assert abs(r['selected_dev_nll']-row['selected'])<2e-6
        work.append(dict(tokens=tokens,width=width,gflops=r['fitting']['arithmetic_flops']/1e9,nll=row['selected']))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    colors={16:'#3178a8',24:'#b84b36',32:'#4e8761'}
    for width in (16,24,32):
        chosen=[r for r in rows if r['width']==width]
        axes[0].scatter([r['tokens'] for r in chosen],[r['selected'] for r in chosen],color=colors[width],label=f'P{width}',s=45)
    axes[0].scatter([65536],[repeat],marker='D',facecolors='none',edgecolors=colors[24],label='P24 seed7',s=50)
    axes[0].set_xscale('log',base=2);axes[0].set_xticks([8192,65536,262144],['8K','64K','256K'])
    axes[0].set(xlabel='Distinct admitted TRAIN tokens',ylabel='Selected development NLL',title='Data growth: measured points')
    axes[0].legend(fontsize=8)
    central=[r for r in rows if r['tokens']==65536]
    axes[1].scatter([r['parameters']/1e6 for r in central],[r['selected'] for r in central],color=[colors[r['width']] for r in central],s=45)
    for r in central:axes[1].annotate(f"P{r['width']}",(r['parameters']/1e6,r['selected']),xytext=(5,5),textcoords='offset points')
    axes[1].set(xlabel='All parameters (millions)',ylabel='Selected development NLL',title='64K capacity: seed6')
    axes[2].scatter([r['gflops'] for r in work],[r['nll'] for r in work],color=[colors[r['width']] for r in work],s=45)
    for r in work:axes[2].annotate(f"{r['tokens']//1024}K / P{r['width']}",(r['gflops'],r['nll']),xytext=(5,5),textcoords='offset points')
    axes[2].set_xscale('log');axes[2].set(xlabel='Executed whole-fit arithmetic GFLOPs',ylabel='Selected development NLL',title='Audited fits: different widths')
    for ax in axes:ax.grid(alpha=.2);ax.margins(.2)
    fig.suptitle('Integrated tokenized language: completed measurements, no fitted scaling law',fontsize=13)
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.svg','.json'):
        if prefix.with_suffix(suffix).exists():raise FileExistsError(prefix.with_suffix(suffix))
    fig.savefig(prefix.with_suffix('.png'),dpi=200);fig.savefig(prefix.with_suffix('.svg'));plt.close(fig)
    record=dict(status='completed_measured_visualization',rows=rows,seed7_p24_repeat=repeat,work=work,input_sha256=provenance,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),curve_fitted=False,
        scope='Same2040-target DEV population and two-pass fits, initial-inclusive selection; TRAIN frequency prior and checkpoint cadence depend on budget. Seed6 primary; one P24seed7 repeat. Width changes core/input/readout together. No raw points connected or extrapolated. Audited cost points differ in data and width; no iso-quality/iso-FLOP or exponent claim. Arithmetic fit boundary includes discovery/replay/backward/optimizer; special functions separate, random-sampling work unquantified. 256K work absent until an actual audit.')
    prefix.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Exported PNG/SVG and source-bound measurement receipt')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-prefix',required=True);a=p.parse_args();build(a.output_prefix)
