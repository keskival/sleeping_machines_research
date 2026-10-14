"""Bounded descriptive logit spectra without materializing contexts x vocabulary."""
import torch
import time


@torch.no_grad()
def centered_logit_spectrum(features, weights):
    """Singular values of context- and class-centered affine logits.

    Rows are contexts/classes; columns share a feature space. Biases and the
    per-context softmax log-normalizer cancel under double centering. This is
    a descriptive prediction spectrum, not an optimization response spectrum.
    """
    if features.ndim!=2 or weights.ndim!=2 or features.shape[1]!=weights.shape[1]:
        raise ValueError('Matched context and class feature matrices required')
    if features.shape[0]<2 or weights.shape[0]<2:
        raise ValueError('At least two contexts and classes required')
    x=features.double();w=weights.double()
    if not bool(torch.isfinite(x).all() and torch.isfinite(w).all()):
        raise ValueError('Finite features and weights required')
    x=x-x.mean(0,keepdim=True);w=w-w.mean(0,keepdim=True)
    rx=torch.linalg.qr(x,mode='r').R
    rw=torch.linalg.qr(w,mode='r').R
    values=torch.linalg.svdvals(rx@rw.T)
    energy=values.square();total=float(energy.sum())
    if total>0:
        fractions=energy/total
        entropy=-torch.where(fractions>0,fractions*fractions.clamp_min(1e-300).log(),0.).sum()
        effective_rank=float(entropy.exp())
        top_fraction=float(fractions[0])
    else:
        effective_rank=0.;top_fraction=0.
    return dict(contexts=features.shape[0],classes=weights.shape[0],feature_width=features.shape[1],
                rank_upper_bound=min(features.shape[0]-1,weights.shape[0]-1,features.shape[1]),
                singular_values=values.tolist(),squared_logit_energy=total,
                energy_effective_rank=effective_rank,top_direction_energy_fraction=top_fraction)


@torch.no_grad()
def adaptive_readout_spectra(readout, features):
    start=time.monotonic()
    x=features.reshape(-1,features.shape[-1])
    if not readout.adaptive:
        branches={'all_tokens':centered_logit_spectrum(x,readout.output.weight)}
    else:
        output=readout.output;top=output.cutoffs[0]
        branches={'frequent_tokens':centered_logit_spectrum(x,output.head.weight[:top])}
        for tail,low,high in zip(output.tail,output.cutoffs[:-1],output.cutoffs[1:]):
            branches[f'ranks_{low}_{high}']=centered_logit_spectrum(tail[0](x),tail[1].weight)
    return dict(branches=branches,diagnostic_wall_s=time.monotonic()-start,
        class_rows_scanned=sum(branch['classes'] for branch in branches.values()),
        output_weight_coordinates_scanned=sum(branch['classes']*branch['feature_width'] for branch in branches.values()),
        scope='Frozen TRAIN contexts; conditional within-branch centered affine logits. QR factors preserve nonzero singular values without a contexts-by-vocabulary matrix. Context rank/energy is not conditional information, useful-depth causality, learning rate, attainable quality or a frontier scaling law.')
