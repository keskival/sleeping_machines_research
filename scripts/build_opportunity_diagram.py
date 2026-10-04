"""Render the proposed codec boundary as native SVG; standard library only."""
from pathlib import Path
from build_architecture_atlas import Diagram, BLUE, GREEN, GRAY, ORANGE

ROOT = Path(__file__).resolve().parents[1]


def diagram():
    d = Diagram('Opportunity: learned asynchronous communication and memory', 670)
    d.label(34,78,'Proposed composition, not a trained codec or a measured bitrate advantage.',17,GRAY)
    d.box(30,115,245,145,'Causal observations',[
        'Irregular events or frames', 'Typed content / address / time', 'Only available source history'])
    d.box(320,115,245,145,'Sender state + policy',[
        'What to retain and retrieve', 'What / where / when to send', 'Fast state; slow learned rules'],GREEN)
    d.box(610,115,245,145,'Quantized wire format',[
        'Codes, addresses and times', 'Entropy / framing / reset bits', 'Count actual serialized bytes'],ORANGE)
    d.box(900,115,270,145,'Receiver state',[
        'Update from received history', 'Predict between messages', 'No access to unsent facts'],GREEN)
    for left,right in ((275,320),(565,610),(855,900)):
        d.arrow([(left,185),(right-6,185)])
    d.box(30,315,535,120,'Variable activity is a design choice',[
        'Wait when receiver predicts; send useful changes.',
        'Cadence, message count, bits and work are separate.'],BLUE)
    d.box(610,315,560,120,'Fixed reconstruction or task queries',[
        'Same query cutoffs, coverage, deadlines and declared fallback.',
        'A task-sufficient feature stream may not reconstruct the source.'],BLUE)
    d.arrow([(1035,260),(1035,309)])
    d.box(30,495,1140,120,'Learn from future usefulness and complete resource cost',[
        'Objective: query distortion/task loss + transmitted bits + sender/receiver work + lateness.',
        'Continuous credit plus useful alternative send/wait, write, retention and timing consequences.',
        'Meta-learning analogy: fast evidence/parameters; slow rules that learn how to update and communicate.'],GREEN)
    d.arrow([(890,435),(890,489)],GREEN)
    d.arrow([(442,495),(442,441),(585,441),(585,96),(442,96),(442,109)],GREEN,True)
    d.label(34,651,'Encoder/decoder agreement, side information and training work belong in the contract.',16,GRAY)
    return d.finish()


if __name__ == '__main__':
    (ROOT/'report/figures/model_family_codec.svg').write_text(diagram())
