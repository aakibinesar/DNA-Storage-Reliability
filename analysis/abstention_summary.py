"""
analysis/abstention_summary.py
==============================
Aggregate the abstention analysis: combine the validation OFR of each candidate
reallocation budget (results/deployable_abstention/, from
analysis/deployable_abstention.py) with the test-set OFR arrays already stored in
results/allocation/ to evaluate three deployable policies against uniform allocation:

  forced      the primary experiment's policy -- budget tau chosen on validation from
              {0.05, 0.10, 0.20, 0.30}, so it can never decline to reallocate
  abstain     same grid plus tau = 0 (uniform allocation); ties resolved toward 0
  selected    validation OFR chooses among {uniform, risk model @ its best non-zero tau,
              benefit-aware model @ its best non-zero tau}; ties resolved toward
              uniform, then the risk model (delta in {2, 3, 4} only)

No new channel simulation is needed. Candidates and seeds are identical to the primary
experiment, so an abstention-enabled policy that picks a non-zero tau picks the same one
as the forced policy (its test OFR is the stored array), and one that picks tau = 0 IS
uniform allocation (test OFR = ofr_uniform). A hard assertion checks that the re-scored
validation reproduces every tier fraction saved by allocation/experiment.py.

Statistics: paired Wilcoxon signed-rank against uniform per (config, delta) with
Benjamini-Hochberg FDR within each policy's family of tests (112 for the risk model,
84 for the benefit-aware and selected policies); direction of a significant result
comes from the mean paired OFR difference. Comparisons with identical OFR in all runs
(e.g. an abstaining policy vs. uniform) have no defined test and count as not significant.

Outputs (in --out, default results/deployable_abstention/):
  abstention_summary.csv   one row per (key, delta, model in {risk, benefit})
  abstention_selected.csv  one row per (key, delta in {2, 3, 4})

Usage:
    python analysis/abstention_summary.py
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    o = np.argsort(pv)
    r = np.minimum.accumulate((pv[o] * m / (np.arange(m) + 1))[::-1])[::-1]
    adj = np.empty(m)
    adj[o] = np.clip(r, 0, 1)
    out[ok] = adj
    return out


def ptest(cond, uni):
    if np.all(cond == uni):
        return np.nan
    try:
        return float(wilcoxon(cond, uni, zero_method='wilcox').pvalue)
    except ValueError:
        return np.nan


def direction(p_adj, gain):
    return np.where(~(p_adj < 0.05), 'ns', np.where(gain > 0, 'better', 'worse'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allocation-dir', default='results/allocation/')
    parser.add_argument('--abstention-dir', default='results/deployable_abstention/')
    parser.add_argument('--out', default='results/deployable_abstention/')
    args = parser.parse_args()

    abst = pd.concat([pd.read_csv(f) for f in
                      sorted(glob.glob(os.path.join(args.abstention_dir, '*_abstention.csv')))])
    assert abst.key.nunique() == 28, f'expected 28 configs, found {abst.key.nunique()}'

    npz_files = sorted(glob.glob(os.path.join(args.allocation_dir, '*_delta*.npz')))
    rows, crow = [], []
    for f in npz_files:
        m = re.match(r'^(?P<key>.+)_delta(?P<delta>\d+)\.npz$', os.path.basename(f))
        key, delta = m['key'], int(m['delta'])
        d = np.load(f)
        uni = d['ofr_uniform']
        for model, ofr_name, tf_name in [('risk', 'ofr_xgb_cal_deployable', 'tier_fraction_deployable'),
                                         ('benefit', 'ofr_benefit_model_deployable', 'tier_fraction_benefit')]:
            if ofr_name not in d.files:
                continue
            g = abst[(abst.key == key) & (abst.delta == delta) & (abst.model == model)]
            gnz = g[g.tier_fraction > 0]
            best_nz = float(gnz.loc[gnz.val_ofr.idxmin(), 'tier_fraction'])
            tf_forced = float(d[tf_name][0])
            assert abs(best_nz - tf_forced) < 1e-9, \
                f'reproduction mismatch {key} delta={delta} {model}: {best_nz} vs {tf_forced}'
            vmin = g.val_ofr.min()
            tf_abst = float(g[g.val_ofr <= vmin + 1e-12].tier_fraction.min())
            forced = d[ofr_name]
            chosen = uni if tf_abst == 0 else forced
            val0 = float(g[g.tier_fraction == 0].val_ofr.iloc[0])
            rows.append(dict(
                key=key, delta=delta, model=model, tf_forced=tf_forced, tf_abstain=tf_abst,
                abstained=bool(tf_abst == 0), val_gain_pp=(val0 - float(gnz.val_ofr.min())) * 100,
                gain_forced_pp=(uni.mean() - forced.mean()) * 100,
                gain_abstain_pp=(uni.mean() - chosen.mean()) * 100,
                p_forced=ptest(forced, uni), p_abstain=ptest(chosen, uni)))

        if 'ofr_benefit_model_deployable' in d.files:
            gr = abst[(abst.key == key) & (abst.delta == delta) & (abst.model == 'risk')]
            gb = abst[(abst.key == key) & (abst.delta == delta) & (abst.model == 'benefit')]
            v0 = float(gr[gr.tier_fraction == 0].val_ofr.iloc[0])
            vr = float(gr[gr.tier_fraction > 0].val_ofr.min())
            vb = float(gb[gb.tier_fraction > 0].val_ofr.min())
            choices = [('uniform', v0, uni), ('risk', vr, d['ofr_xgb_cal_deployable']),
                       ('benefit', vb, d['ofr_benefit_model_deployable'])]
            best = min(choices, key=lambda c: c[1])  # ties -> first (uniform, then risk)
            crow.append(dict(key=key, delta=delta, choice=best[0],
                             gain_pp=(uni.mean() - best[2].mean()) * 100, p=ptest(best[2], uni)))

    res = pd.DataFrame(rows)
    for model in res.model.unique():
        mk = res.model == model
        for v in ('forced', 'abstain'):
            res.loc[mk, f'p_adj_{v}'] = bh(res.loc[mk, f'p_{v}'].to_numpy())
    for v in ('forced', 'abstain'):
        res[f'direction_{v}'] = direction(res[f'p_adj_{v}'], res[f'gain_{v}_pp'])
    sel = pd.DataFrame(crow)
    sel['p_adj'] = bh(sel.p.to_numpy())
    sel['direction'] = direction(sel.p_adj, sel.gain_pp)

    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, 'abstention_summary.csv'), index=False, float_format='%.6g')
    sel.to_csv(os.path.join(args.out, 'abstention_selected.csv'), index=False, float_format='%.6g')

    def ms(x):
        return f'{x.mean():+.2f} +- {x.std(ddof=1) / np.sqrt(len(x)):.2f}'

    print('Policy outcomes vs uniform allocation; mean OFR reduction (pp, mean +- s.e. over configs) and '
          'significantly better/worse counts')
    for (model, delta), g in res.groupby(['model', 'delta']):
        for v in ('forced', 'abstain'):
            nb, nw = int((g[f'direction_{v}'] == 'better').sum()), int((g[f'direction_{v}'] == 'worse').sum())
            extra = f'  (abstains in {int(g.abstained.sum())}/{len(g)})' if v == 'abstain' else ''
            print(f'  {model:8s} delta={delta} {v:8s} {ms(g[f"gain_{v}_pp"]):>15s}  better/worse {nb}/{nw}{extra}')
    for delta, g in sel.groupby('delta'):
        nb, nw = int((g.direction == 'better').sum()), int((g.direction == 'worse').sum())
        print(f'  selected delta={delta} {ms(g.gain_pp):>15s}  better/worse {nb}/{nw}  '
              f'choices {g.choice.value_counts().to_dict()}')
    for model, g in res.groupby('model'):
        print(f'{model}: forced better/worse {int((g.direction_forced == "better").sum())}/'
              f'{int((g.direction_forced == "worse").sum())} of {len(g)}; abstain '
              f'{int((g.direction_abstain == "better").sum())}/{int((g.direction_abstain == "worse").sum())}')
    print(f'selected: {int((sel.direction == "better").sum())}/{int((sel.direction == "worse").sum())} of {len(sel)}')


if __name__ == '__main__':
    main()
