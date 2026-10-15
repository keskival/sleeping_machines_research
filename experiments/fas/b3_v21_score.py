"""FAS v2.1 sealed confirmation scoring (experiments/FAS_V21_CONFIRMATION.md). Scores every arm once on the fresh test
draw and applies the frozen B3 Stage 5 decision rule. Refuses to run unless the preregistration is committed and
unmodified and the test draw was generated after it."""
import argparse, datetime, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'experiments/fas')); sys.path.insert(0, str(ROOT/'experiments/tpp'))
import torch
from native import load
import race_delay_binding as R
import addressed_evidence as A
import classical_scores_v2 as C
from baselines import runs as raw_runs

PREREG = 'experiments/FAS_V21_CONFIRMATION.md'; SPEC = 'experiments/fas/FAS_V21_CONFIRMATION.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
N_STAR = 1024


def committed(path):
    ok = subprocess.run(['git', 'ls-files', '--error-unmatch', path], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', path], cwd=ROOT).returncode == 0
    when = subprocess.check_output(['git', 'log', '-1', '--format=%ct', '--', path], cwd=ROOT, text=True).strip()
    return ok and clean, int(when) if when else None


def auc(neg, pos):
    s = np.r_[neg, pos]; r = s.argsort().argsort().astype(float)
    # midranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); csum = np.cumsum(cnt); r = (csum - (cnt - 1)/2.0)[inv]
    return float((r[len(neg):].sum() - len(pos)*(len(pos) + 1)/2)/(len(neg)*len(pos)))


def ranks(neg, pos):
    s = np.r_[neg, pos]; _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); csum = np.cumsum(cnt)
    r = (csum - (cnt - 1)/2.0)[inv]/len(s); return r[:len(neg)], r[len(neg):]


def seed_avg_ranks(pairs):
    neg = np.mean([ranks(c, f)[0] for c, f in pairs], 0); pos = np.mean([ranks(c, f)[1] for c, f in pairs], 0); return neg, pos


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args(); torch.set_num_threads(1); began = time.time()
    spec = json.loads((ROOT/SPEC).read_text()); out = ROOT/f'experiments/results/fas/{a.tag}.json'; assert not out.exists()
    for p in (PREREG, SPEC):
        ok, when = committed(p); assert ok, p
    test_dir = ROOT/'experiments/data/fas'/spec['test_data']; manifest = json.loads((test_dir/'manifest.json').read_text())
    assert manifest['args']['seed_base'] == spec['test_seed_base'] and manifest['args']['K'] == 2 and manifest['args']['drop'] == 0.02 and manifest['args']['delta'] == 0.0
    assert manifest['started'] > committed(PREREG)[1], 'test draw must postdate the committed preregistration'
    tc, _ = load(test_dir/'test_clean.npz', 1100); tf, tk = load(test_dir/'test_faulty.npz', 1100); tk = np.asarray(tk)
    keep_c = np.array([len(r[0]) >= N_STAR for r in tc]); keep_f = np.array([len(r[0]) >= N_STAR for r in tf])
    arms = {}
    # Native delay-race binding: three disjoint TRAIN-slice clock fits, identical calibration and rule.
    train_all, _ = load(ROOT/'experiments/data/fas'/R.MAIN/'train_clean.npz', 1100)
    cal_runs, _ = load(ROOT/'experiments/data/fas'/R.CALIB/'val_clean.npz', 1100)
    cfg = spec['native']; native = []
    for s0, s1 in cfg['train_slices']:
        clocks = R.learn_clocks(train_all[s0:s1], cfg['window'], cfg['top']); cal = R.calibrate(cal_runs, clocks, cfg['window'], bool(cfg['consume']), cfg['clip'])
        native.append((R.score(tc, clocks, cal, N_STAR, cfg['kappa'], cfg['window'], bool(cfg['consume']), cfg['clip']),
                       R.score(tf, clocks, cal, N_STAR, cfg['kappa'], cfg['window'], bool(cfg['consume']), cfg['clip'])))
    del train_all
    arms['native_delay_race'] = dict(eligible=False, native=True, seeds=native)
    # Learned references: family statistic selected on v2 validation, then one test scoring per seed.
    vc, _ = load(ROOT/'experiments/data/fas'/A.MAIN/'val_clean.npz', 1100); vf, _ = load(ROOT/'experiments/data/fas'/A.MAIN/'val_faulty.npz', 1100)
    for family, results in spec['references'].items():
        val_auc = {}; seeds = {}
        for rj in results:
            pack = {}
            for name, rr in (('calib', cal_runs), ('vc', vc), ('vf', vf), ('tc', tc), ('tf', tf)):
                PT, PG, _ = A.per_event_dense(ROOT/rj, rr, lanes=4); pack[name] = (PT, PG, A.targets(rr, PT.shape[1]))
            for channel in ('total', 'gap'):
                pick = (lambda T, G: T + G) if channel == 'total' else (lambda T, G: G)
                mu, sd = A.calibrate(pick(*pack['calib'][:2]), pack['calib'][2])
                stv = {k: (A.statistics(pick(*pack['vc'][:2]), pack['vc'][2], mu, sd, N_STAR)[k], A.statistics(pick(*pack['vf'][:2]), pack['vf'][2], mu, sd, N_STAR)[k]) for k in ('total', 'cusum_k0', 'cusum_k0.5', 'cusum_k1', 'excess')}
                stt = {k: (A.statistics(pick(*pack['tc'][:2]), pack['tc'][2], mu, sd, N_STAR)[k], A.statistics(pick(*pack['tf'][:2]), pack['tf'][2], mu, sd, N_STAR)[k]) for k in stv}
                for k in stv:
                    val_auc.setdefault(f'{channel}:{k}', []).append(auc(*stv[k])); seeds.setdefault(f'{channel}:{k}', []).append(stt[k])
        chosen = max(val_auc, key=lambda k: np.mean(val_auc[k]))
        for stat in {'total:total', chosen}:
            arms[f'{family}[{stat}]'] = dict(eligible=True, seeds=seeds[stat], val_auroc=float(np.mean(val_auc[stat])), selected_on_val=stat == chosen)
    # Classical information-matched detectors, fitted on 2,000 clean TRAIN runs (raw runs with TICK, as in v2).
    train_raw, _ = raw_runs(ROOT/'experiments/data/fas'/R.MAIN/'train_clean.npz'); fns = C.fit(train_raw[:2000]); del train_raw
    rc, _ = raw_runs(test_dir/'test_clean.npz'); rf, _ = raw_runs(test_dir/'test_faulty.npz'); j = C.PREFIXES.index(N_STAR)
    sc, sf = C.score_runs(fns, rc), C.score_runs(fns, rf)
    for name in C.DETECTORS: arms[f'classical:{name}'] = dict(eligible=True, seeds=[(sc[name][:, j], sf[name][:, j])])
    # Eligible samples: runs reaching the primary prefix for every arm.
    for k, v in arms.items():
        v['seeds'] = [(np.asarray(c)[keep_c], np.asarray(f)[keep_f]) for c, f in v['seeds']]
        v['seed_aucs'] = [auc(c[np.isfinite(c)], f[np.isfinite(f)]) for c, f in v['seeds']]; v['auroc'] = float(np.mean(v['seed_aucs']))
    eligible = {k: v for k, v in arms.items() if v['eligible']}; strongest = max(eligible, key=lambda k: eligible[k]['auroc'])
    nat = arms['native_delay_race']; diff = nat['auroc'] - eligible[strongest]['auroc']
    rng = np.random.default_rng(20261015); nn_, npos = seed_avg_ranks(nat['seeds']); bn, bp = seed_avg_ranks(eligible[strongest]['seeds']); boots = []
    for _ in range(10000):
        ic = rng.integers(0, len(nn_), len(nn_)); ifa = rng.integers(0, len(npos), len(npos))
        boots.append(auc(nn_[ic], npos[ifa]) - auc(bn[ic], bp[ifa]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    criteria = dict(margin_ge_0_02=bool(diff >= 0.02), bootstrap_lower_gt_0=bool(lo > 0), every_seed_above=bool(all(x > eligible[strongest]['auroc'] for x in nat['seed_aucs'])))
    verdict = 'win' if all(criteria.values()) else ('loss' if diff <= -0.02 and hi < 0 else 'tie')
    per_kind = {}
    for kind in np.unique(tk[keep_f]):
        sel = tk[keep_f] == kind; per_kind[str(int(kind))] = {k: float(np.mean([auc(c, f[sel]) for c, f in v['seeds']])) for k, v in arms.items()}
    record = dict(status='completed', battle='B3', tag=a.tag, preregistration=PREREG, preregistration_sha256=sha(ROOT/PREREG), spec_sha256=sha(ROOT/SPEC),
                  test_data=spec['test_data'], test_manifest_sha256=sha(test_dir/'manifest.json'), eligible_samples=dict(clean=int(keep_c.sum()), faulty=int(keep_f.sum())),
                  native_seed_aucs=nat['seed_aucs'], native_mean=nat['auroc'], strongest_baseline=strongest, strongest_auroc=eligible[strongest]['auroc'],
                  difference=diff, bootstrap_95=[float(lo), float(hi)], criteria=criteria, verdict=verdict, per_fault_kind=per_kind,
                  arms={k: dict(auroc=v['auroc'], seed_aucs=v['seed_aucs'], eligible=v['eligible'], **({'val_auroc': v['val_auroc'], 'selected_on_val': v['selected_on_val']} if 'val_auroc' in v else {})) for k, v in arms.items()},
                  wall_s=time.time() - began, source_sha256={n: sha(ROOT/n) for n in ['experiments/fas/b3_v21_score.py', 'experiments/fas/race_delay_binding.py', 'experiments/fas/addressed_evidence.py',
                  'experiments/fas/classical_scores_v2.py', 'experiments/fas/baselines.py', 'experiments/fas/native.py', 'experiments/fas/dense.py']})
    out.write_text(json.dumps(record, indent=1) + '\n')
    with (ROOT/'experiments/results/fas/fas_v2_test_ledger.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(tag=a.tag, kind='v2.1 sealed confirmation, all arms once', data=spec['test_data'], reason='FAS_V21_CONFIRMATION preregistered fresh draw',
                                 utc=datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'))) + '\n')
    print(json.dumps(dict(verdict=verdict, native=nat['auroc'], strongest=strongest, strongest_auroc=eligible[strongest]['auroc'], ci=[lo, hi])))


if __name__ == '__main__':
    main()
