#!/usr/bin/env python3
"""Native mechanism coverage gate for R1 queue files (founder, 17 Oct: "You switched off all our unfair advantages and started tuning a
version of ReZero Transformers"; "Why did you not proactively notice that we were drifting towards classic models?").

A queue file whose header says `# Role: flagship` must run the full native member: every implemented native mechanism on. Files
without a Role line, or `# Role: diagnostic`, are reported with their missing mechanisms so a diagnostic can never pass silently as
the flagship. Designed mechanisms that are not yet integrated into the driver are listed as declared gaps on every check.
Usage: check_native_coverage.py <queue file> [...]   (exit 1 when a flagship file misses a mechanism)."""
import re, shlex, sys

NATIVE = [  # (mechanism, flag, test on the flag's value)
    ('sparse addressed slot level 1', '--level1', lambda v: v == 'slots'),
    ('product-key slot addressing (capacity beyond activity)', '--slot-pk', lambda v: v == '1'),
    ('sparse level reads', '--read-topk', lambda v: int(v) > 0),
    ('read probes (counterfactual read credit)', '--read-explore', lambda v: int(v) > 0),
    ('slot probes (counterfactual slot credit)', '--slot-explore', lambda v: int(v) > 0),
    ('asynchronous firing levels', '--fire', lambda v: v == '1'),
    ('composing event spine between levels', '--events', lambda v: v == '1'),
    ('write-address race', '--write-race', lambda v: v == '1'),
    ('write probes (counterfactual write credit)', '--write-explore', lambda v: int(v) > 0),
    ('normalised write race (full route-tree limit)', '--write-norm', lambda v: v == '1'),
    ('per-target old-slot corrections', '--old-per-target', lambda v: v == '1'),
    ('per-correction clocks (learnable lifetime per stored value)', '--old-clocks', lambda v: int(v) > 0),
    ('persistent memory across lanes', '--carry', lambda v: int(v) > 0),
    ('matrix-only weight decay (clocks and dormant capacity kept)', '--wd-groups', lambda v: v == '1'),
    ('large concept bank with product-key wake (capacity beyond activity)', '--product-keys', lambda v: v == '1'),
    ('concept symmetry breaking', '--unit-init', lambda v: float(v) > 0),
    ('concept load balance', '--balance', lambda v: float(v) > 0),
    ('calibrated bank-local clipping', '--clip', lambda v: v == 'local'),
]
BUNDLES = {  # synergistic mechanisms are attributed together: an ablation removes one whole bundle from the full native member
    'sparse_addressing': ['sparse addressed slot level 1', 'product-key slot addressing (capacity beyond activity)', 'sparse level reads', 'read probes (counterfactual read credit)', 'slot probes (counterfactual slot credit)'],
    'async_spine': ['asynchronous firing levels', 'composing event spine between levels'],
    'persistence': ['persistent memory across lanes'],
    'capacity': ['large concept bank with product-key wake (capacity beyond activity)', 'concept symmetry breaking', 'concept load balance'],
    'writes': ['write-address race', 'write probes (counterfactual write credit)', 'normalised write race (full route-tree limit)', 'per-target old-slot corrections'],
    'local_learning': ['calibrated bank-local clipping'],
}
DECLARED_GAPS = [
    'threshold (price) read races (--read-price), the TD critic over persistent memory (--carry-critic) and credit-guided write probes (--write-guided): implemented (contracts v27/v28), paired AWS arms pending',
    'credit across lane boundaries for persistent memory (theory 160): implemented as --carry-credit (CarryCredit, exact, contracts v21); paired arm aws_r1_rmr_nativeev_cc1 pending, required for the flagship once it reports',
    'forward-trained credit and future-oriented credit (theory 164, REVEALED_FUTURE_CREDIT, NATIVE_RECIPROCAL_FUTURE_INTEGRATION): training is exact backprop within the lane',
    'event-local asynchronous learning as the training route (AGENTS.md 16 Oct): parameters update synchronously from the whole-lane gradient',
    'homeostatic thresholds as prices (THEORY P5) for concept load: an auxiliary balance loss is used',
    'causal online weight adaptation in FineWeb scoring: implemented for carried runs (online column, contracts v22); not yet a training-time route',
]


def check(path):
    text = open(path).read(); role = re.search(r'^# Role: (\w+)', text, re.M); role = role.group(1) if role else 'undeclared'
    cmd = [l for l in text.splitlines() if l and not l.startswith('#')][-1]; args = shlex.split(cmd); vals = {}
    for i, t in enumerate(args):
        if t.startswith('--') and i + 1 < len(args): vals[t] = args[i + 1]
    missing = [m for m, f, ok in NATIVE if not (f in vals and ok(vals[f]))]
    return role, missing


def main():
    bad = False
    for path in sys.argv[1:]:
        role, missing = check(path); text = open(path).read(); ab = re.search(r'^# Role: ablation (\w+)', text, re.M)
        if ab:                                                                                     # an ablation misses exactly its bundle
            role = f'ablation {ab.group(1)}'; want = set(BUNDLES[ab.group(1)]); status = 'PASS' if set(missing) == want else 'FAIL'
            missing = sorted(set(missing) ^ want)
        else: status = 'PASS' if not missing else ('FAIL' if role == 'flagship' else 'DIAGNOSTIC')
        bad |= status == 'FAIL'
        print(f'{status} [{role}] {path}' + ''.join(f'\n    missing: {m}' for m in missing))
    print('declared gaps (designed, not yet integrated in the R1 driver):' + ''.join(f'\n    - {g}' for g in DECLARED_GAPS))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
