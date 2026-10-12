"""Recover the larger R1 stage after packet-name collision; bounded publish."""
import json
import statistics
import time
from continue_unique_language import FOLDER, publish, submit_larger, wait


def main():
    cohort=[json.loads((FOLDER/f'aws_integrated_p24_unique_stream_decision_20261012_s{s}_v1.json').read_text()) for s in (6,7,8)]
    assert all(d['status']=='completed' and d['quality_gain']>0 and d['rows'][1]['context_gain']>0 for d in cohort)
    assert statistics.mean(d['learning_gain_change'] for d in cohort)>0
    assert statistics.mean(d['rows'][0]['training_wall_s']/d['rows'][1]['training_wall_s'] for d in cohort)>=1/1.05
    print(submit_larger('v2'),flush=True)
    d=wait('aws_integrated_p24_unique_512k_decision_20261012_s6_v2',time.monotonic()+2*3600)
    r,c=d['rows']
    publish(f"Larger same-target stage completed: repeated DEV{r['selected_dev_nll']:.6f}, distinct DEV{c['selected_dev_nll']:.6f}, gain{d['quality_gain']:+.6f}; training{r['training_wall_s']:.3f}/{c['training_wall_s']:.3f}s. Same524272fitting targets/1024updates;262144two-pass versus524280one-pass tokens. Initial-prior gain{d['initial_prior_gain']:+.6f}; learning-gain change{d['learning_gain_change']:+.6f}; context-gain change{d['context_gain_change']:+.6f}. Smaller fixed-target improvement confirmed on three seeds. Integrated P24D2H2U4/credit16 retained; public validation and4M remain reserved. One larger development seed, no Transformer or nonstationarity claim. Packet collision repaired with atomic exclusive names and completed-result prerequisites. Next decision uses this larger quality/work evidence for natural-stream learning and useful capacity, with Curie compact-credit ownership preserved.")


if __name__=='__main__':main()
