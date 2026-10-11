"""Build the current reader-facing learned-credit explanation as vector art."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/figures'
INK = '#172b43'
BLUE = '#2563a6'
TEAL = '#087f80'
ORANGE = '#c76b23'
MUTED = '#546579'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'font.size': 11})


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor('#fbfcfe')
    ax.set(xlim=(0, 18), ylim=(0, 12))
    ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, top=.97, bottom=.025)
    ax.text(.6, 11.5, title, fontsize=23, weight='bold', color=INK, va='top')
    ax.text(.6, 10.8, subtitle, fontsize=11, color=MUTED, va='top')
    return fig, ax


def box(ax, x, y, w, h, title, detail='', color=BLUE, fill='#edf4fc', size=13):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle='round,pad=0.08,rounding_size=0.15',
                               linewidth=1.4, edgecolor=color, facecolor=fill, zorder=2))
    ax.text(x, y + (.17 if detail else 0), title, ha='center', va='center',
            color=INK, fontsize=size, weight='bold', zorder=3)
    if detail:
        ax.text(x, y-.26, detail, ha='center', va='center', color=MUTED,
                fontsize=10, linespacing=1.35, zorder=3)


def arrow(ax, points, color=BLUE, dashed=False, lw=1.8):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                linestyle='--' if dashed else '-', zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                 mutation_scale=14, lw=lw, color=color,
                                 linestyle='--' if dashed else '-', zorder=1))


def comparison():
    fig, ax = canvas('Same goal. A different way to assign learning credit.',
                     'Ordinary synchronous Transformer training vs the proposed temporal, sparse, learned-credit substrate\nCredit tells each component how to change to reduce the prediction loss.')
    ax.text(4.55, 10.15, 'STANDARD TRANSFORMER', ha='center', color=BLUE, weight='bold', fontsize=13)
    ax.text(13.15, 10.15, 'OUR LEARNED-CREDIT CYCLE', ha='center', color=TEAL, weight='bold', fontsize=13)
    ax.plot([8.8, 8.8], [1.4, 10.25], color='#dbe3ee', lw=1)
    box(ax, 4.55, 9.25, 5.4, .8, 'A token batch / context')
    box(ax, 4.55, 7.65, 5.4, 1.45, 'Forward through dense layers',
        'Synchronous layer stages\nKeep activations needed for backward')
    box(ax, 4.55, 5.85, 5.4, 1.0, 'Prediction → observed loss')
    box(ax, 4.55, 4.05, 5.4, 1.35, 'Backpropagate through the graph',
        'Compute exact gradients for this loss', color=ORANGE, fill='#fff4e9')
    box(ax, 4.55, 2.25, 5.4, 1.0, 'Optimizer updates model weights',
        'Then process the next batch', color=ORANGE, fill='#fff4e9')
    for a,b in [(8.83,8.42),(6.88,6.4),(5.3,4.77),(3.33,2.8)]:
        arrow(ax, [(4.55,a),(4.55,b)], BLUE if a>5.3 else ORANGE)
    box(ax, 13.15, 9.25, 5.9, .8, 'A new event + persistent memory', color=TEAL, fill='#e9f7f4')
    box(ax, 13.15, 7.65, 5.9, 1.45, 'Timed, addressed forward computation',
        'Selected state / value work\nProduce reusable evidence for learning', color=TEAL, fill='#e9f7f4')
    box(ax, 13.15, 5.85, 5.9, 1., 'Prediction → observed loss', color=TEAL, fill='#e9f7f4')
    box(ax, 13.15, 4.05, 5.9, 1.35, 'Learned estimator predicts credit',
        'Forward evidence teaches the estimator\nSampled exact residuals anchor correction', color=ORANGE, fill='#fff4e9')
    box(ax, 13.15, 2.25, 5.9, 1., 'Update model + credit estimator',
        'Carry declared state into the next event', color=ORANGE, fill='#fff4e9')
    for a,b in [(8.83,8.42),(6.88,6.4),(5.3,4.77),(3.33,2.8)]:
        arrow(ax, [(13.15,a),(13.15,b)], TEAL if a>5.3 else ORANGE)
    arrow(ax, [(16.35,2.25),(17.2,2.25),(17.2,9.25),(16.2,9.25)], TEAL, True)
    ax.text(.65, 1.25, 'The intended saving: reuse learned credit instead of recomputing all learning work at every event.',
            color=INK, fontsize=12, weight='bold')
    ax.text(.65, .75, 'Evidence scope: local/deep numerical contracts and measured component savings; the complete sparse, persistent loop is in development.',
            color=MUTED, fontsize=10)
    ax.text(.65, .37, 'Charge discovery, counterfactuals, teaching, audits, state transport and optimizer work. No general Transformer advantage is claimed here.',
            color=MUTED, fontsize=10)
    return fig


def closure():
    fig, ax = canvas('How the forward pass teaches the credit estimator',
                     'One local credit boundary; measured responses and observed outcomes close the training loop\nv is an input direction; Jv its forward response; c the observed loss’s local output credit.')
    box(ax, 3.2, 9.15, 4.5, 1.0, 'Forward model θ', 'Produces the next prediction', color=TEAL, fill='#e9f7f4')
    box(ax, 8.5, 9.15, 3.6, 1., 'Prediction', 'Before observing the outcome', color=TEAL, fill='#e9f7f4', size=12)
    box(ax, 14, 9.15, 4.7, 1., 'Observe outcome → loss', 'Get local output credit c', size=12)
    arrow(ax, [(5.55,9.15),(6.62,9.15)], TEAL)
    arrow(ax, [(10.38,9.15),(11.57,9.15)], TEAL)
    box(ax, 3.2, 6.9, 4.5, 1.2, 'Reusable forward packet', 'Context + tangent response Jv', color=TEAL, fill='#e9f7f4', size=12)
    box(ax, 8.5, 6.9, 3.6, 1.2, 'Observed local credit c', 'From the prediction loss', size=11)
    box(ax, 14, 6.9, 4.7, 1.2, 'Credit estimator ψ', 'Predicts input credit ĝ', color=ORANGE, fill='#fff4e9')
    arrow(ax, [(3.2,8.58),(3.2,7.56)], TEAL)
    arrow(ax, [(14,8.58),(14,8.1),(8.5,8.1),(8.5,7.56)], BLUE)
    arrow(ax, [(10.38,6.9),(11.57,6.9)], BLUE)
    arrow(ax, [(5.55,6.9),(6.1,6.9),(6.1,7.9),(11,7.9),(11,7.2),(11.57,7.2)], TEAL)
    ax.text(8.4, 7.9, 'conditioning context', fontsize=9, color=TEAL, ha='center', va='center',
            bbox={'facecolor':'#fbfcfe','edgecolor':'none','pad':2})
    box(ax, 3.2, 4.6, 4.5, 1.2, 'Measured teaching target', 'cᵀ(Jv): loss response to a probe', color=TEAL, fill='#e9f7f4', size=12)
    box(ax, 14, 4.6, 4.7, 1.2, 'Predicted response', 'ĝᵀv: what the credit model expects', color=ORANGE, fill='#fff4e9', size=12)
    arrow(ax, [(3.2,6.24),(3.2,5.26)], TEAL)
    arrow(ax, [(8.5,6.24),(8.5,5.65),(3.8,5.65),(3.8,5.26)], BLUE)
    arrow(ax, [(14,6.24),(14,5.26)], ORANGE)
    box(ax, 8.5, 2.7, 6.1, 1.4, 'Teaching error: (ĝᵀv − cᵀJv)²',
        'A local gradient updates ψ\nNo new learned teacher is needed', color=ORANGE, fill='#fff4e9', size=13)
    arrow(ax, [(3.2,3.94),(3.2,2.7),(5.37,2.7)], TEAL)
    arrow(ax, [(14,3.94),(14,2.7),(11.63,2.7)], ORANGE)
    arrow(ax, [(11.63,2.35),(16.8,2.35),(16.8,6.6),(16.43,6.6)], ORANGE, True)
    ax.text(16.98, 4.5, 'train ψ', rotation=90, color=ORANGE, fontsize=10, va='center')
    arrow(ax, [(16.43,7.2),(17.35,7.2),(17.35,10.05),(3.2,10.05),(3.2,9.72)], ORANGE, True)
    ax.text(8.5,10.15,'Predicted credit + sampled exact residual → next model update',
            fontsize=10, color=ORANGE, ha='center')
    ax.text(.65, 1.55, 'Why it closes: the target is a response measured from the forward computation, anchored on an observed loss.',
            color=INK, fontsize=11, weight='bold')
    ax.text(.65, 1.02, 'Residual audit: ĝ + I/π · (g − ĝ), with positive audit probability π, preserves the conditional raw-gradient mean.',
            color=MUTED, fontsize=10)
    ax.text(.65, .60, 'For content + time, use joint probes. Shared producer probes retain their cross terms. Versions and causal horizons must match.',
            color=MUTED, fontsize=10)
    ax.text(.65, .20, 'This panel explains continuous credit. Hard-route interventions use future-loss advantages; their acquisition and audits are additional work.',
            color=MUTED, fontsize=10)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    with PdfPages(ROOT / 'report/learned_credit_explained.pdf',
                  metadata={'Title':'Learned credit: comparison and closed teaching loop',
                            'Author':'Sleeping Machines Research'}) as pdf:
        for name, builder in [('learned_credit_comparison', comparison),
                              ('learned_credit_teaching_loop', closure)]:
            fig = builder()
            fig.savefig(OUT / f'{name}.svg')
            fig.savefig(OUT / f'{name}.png', dpi=140)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    main()
