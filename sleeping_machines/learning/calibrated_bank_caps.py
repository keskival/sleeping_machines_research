"""Offline TRAIN calibration of fixed bank-local radial caps.

A global reduction is paid only during calibration, never during local updates.
The cap budget is a declared engineering target, not an optimality claim.
"""
import math
import torch

def calibrate_bank_caps(norm_samples,global_scales,*,quantile=.9,budget=1.25):
    if not norm_samples or not 0<quantile<1 or not math.isfinite(budget) or budget<=0:
        raise ValueError('Positive finite budget, supported quantile and banks required')
    scales=torch.as_tensor(global_scales,dtype=torch.float64)
    if scales.ndim!=1 or not scales.numel() or not bool(torch.isfinite(scales).all()) or not bool(((scales>0)&(scales<=1)).all()):
        raise ValueError('Finite global attenuation samples required')
    rows={}
    for bank,values in norm_samples.items():
        v=torch.as_tensor(values,dtype=torch.float64)
        if v.shape!=scales.shape or not bool(torch.isfinite(v).all()) or bool((v<0).any()):raise ValueError('Matched raw norm samples required')
        rows[bank]=v
    observed=torch.cat(list(rows.values()));positive=observed[observed>0]
    if not positive.numel():raise ValueError('No nonzero calibration credit')
    # Cold banks retain a declared conservative positive fallback; later exposure
    # must be audited rather than treating absent calibration credit as zero cost.
    fallback=float(torch.quantile(positive,quantile))*float(scales.median())
    caps={bank:max(float(torch.quantile(v*scales,quantile)),fallback if not bool((v>0).any()) else torch.finfo(torch.float64).eps) for bank,v in rows.items()}
    radius=math.sqrt(sum(c*c for c in caps.values()))
    multiplier=budget/radius
    caps={bank:c*multiplier for bank,c in caps.items()}
    assert all(math.isfinite(c) and c>0 for c in caps.values())
    rates={bank:float((v>caps[bank]).double().mean()) for bank,v in rows.items()}
    return dict(caps=caps,quantile=quantile,budget=budget,cap_squared_sum=sum(c*c for c in caps.values()),clip_rates=rates,cold_banks=[b for b,v in rows.items() if not bool((v>0).any())],scope='Fixed caps from same-cut TRAIN raw norms attenuated by global-reference scales; aggregate cap budget may change quantile clipping rates. No Adam-step, quality or optimality claim.')
