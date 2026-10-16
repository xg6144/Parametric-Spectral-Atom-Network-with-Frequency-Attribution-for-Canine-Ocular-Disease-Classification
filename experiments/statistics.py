"""Paired, condition-controlled comparisons with explicit follow-up families."""
import json

import numpy as np
import pandas as pd

from .analysis import read_new
from .common import ROOT, sha256, write_json, status

EFFECTS = ['atom_main_effect_gaussian_minus_gabor', 'dropout_main_effect',
           'interaction_difference_of_differences']
SIMPLE = ['atom_at_ad0', 'atom_at_ad02', 'dropout_gabor', 'dropout_gaussian']


def aligned(a, b):
    fa, pa, _, _ = read_new(a); fb, pb, _, _ = read_new(b)
    if not np.array_equal(fa.path, fb.path) or not np.array_equal(fa.label, fb.label):
        raise ValueError('Different paths/labels in paired comparison')
    if not np.array_equal(fa.group_id, fb.group_id):
        raise ValueError('Different bootstrap groups')
    return fa, pa, pb


def weighted_macro_f1(y, pred, weights, classes=11):
    """Exact all-class F1 from weighted confusion counts, including absent classes."""
    counts = np.bincount(y * classes + pred, weights=weights, minlength=classes**2).reshape(classes, classes)
    tp = np.diag(counts)
    denominator = counts.sum(0) + counts.sum(1)
    return float(np.divide(2 * tp, denominator, out=np.zeros(classes), where=denominator > 0).mean())


def tail_summary(observed, distribution, family):
    p = min(1., 2 * min((np.count_nonzero(distribution <= 0) + 1) / (len(distribution) + 1),
                       (np.count_nonzero(distribution >= 0) + 1) / (len(distribution) + 1)))
    return dict(estimate=float(observed), ci_low=float(np.quantile(distribution, .025)),
                ci_high=float(np.quantile(distribution, .975)), p_uncorrected=p,
                p_bonferroni=min(1., p * family), planned_family=family)


def pair_summary(frame, pa, pb, args, family):
    y = frame.label.to_numpy(dtype=int); pred_a = pa.argmax(1); pred_b = pb.argmax(1)
    _, codes = np.unique(frame.group_id, return_inverse=True); groups = int(codes.max() + 1)
    rng = np.random.default_rng(args.seed)
    observed = weighted_macro_f1(y, pred_a, np.ones(len(y))) - weighted_macro_f1(y, pred_b, np.ones(len(y)))
    distribution = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        weights = np.bincount(rng.integers(0, groups, groups), minlength=groups)[codes]
        distribution[i] = weighted_macro_f1(y, pred_a, weights) - weighted_macro_f1(y, pred_b, weights)
    value = tail_summary(observed, distribution, family)
    value['delta_macro_f1'] = value.pop('estimate')
    return value


def prediction_path(summary):
    return (ROOT / summary['checkpoint']).with_name('test_predictions.csv')


def factorial_scores(values):
    g0, g2, q0, q2 = np.moveaxis(np.asarray(values), -1, 0)
    return np.stack([(q0-g0), (q2-g2), (g2-g0), (q2-q0),
                     (q0+q2-g0-g2)/2, (g2+q2-g0-q0)/2, (q2-q0)-(g2-g0)], axis=-1)


def run(args):
    protocol_path = ROOT / 'configs/round2_statistics.json'
    plan = json.loads(protocol_path.read_text())
    runs = [json.loads(p.read_text()) for p in (ROOT/'results/training').glob('*/summary.json')]
    canonical = {}
    for s in runs:
        if s['dataset'] != 'canine': continue
        if s['model'] == 'psan' and not s['tag'].startswith('canine_psan_gabor_ad0p2_M16_morlet_s'): continue
        canonical[(s['model'], s['seed'])] = s
    primary = []
    for (model, seed), s in sorted(canonical.items()):
        if model == 'psan' or ('psan', seed) not in canonical: continue
        if not ((model in ['gfnet', 'afno'] and seed in [42,43,44,45,46]) or
                (model == 'gfnet_matched' and seed in [42,43,44]) or
                (model in ['deit','convnext','fastvit_t8','swin','shvit_s1'] and seed == 42)): continue
        f, pa, pb = aligned(prediction_path(canonical[('psan', seed)]), prediction_path(s))
        primary.append(dict(seed=seed, comparison='PSAN minus '+model,
                            **pair_summary(f, pa, pb, args, plan['primary_family'])))
    pd.DataFrame(primary).to_csv(ROOT/'results/tables/main_paired_comparisons.csv', index=False)
    baseline = next((s for s in runs if s['tag'] == 'canine_psan_gabor_ad0p0_M16_morlet_s42'), None)
    ofat = []
    if baseline:
        for s in runs:
            if s['dataset']!='canine' or s['model']!='psan' or s['seed']!=42 or s['tag']==baseline['tag']: continue
            config = json.loads((ROOT/s['checkpoint']).with_name('config.json').read_text())
            changed = sum([config['atoms']!=16, config['atom']!='gabor', config['init']!='morlet',
                           bool(config['axis_aligned']), bool(config['fixed_phase']), config['atom_dropout']!=0])
            if changed != 1 or s['atom_dropout'] not in [0., .1, .2]: continue
            f, pa, pb = aligned(prediction_path(s), prediction_path(baseline))
            ofat.append(dict(variant=s['tag'], reference=baseline['tag'],
                **pair_summary(f, pa, pb, args, plan['ofat_family']),
                initial_nonfilter_control=config.get('initialization_protocol',
                    'same M16 architecture/seed; identical initial non-filter tensors')))
    pd.DataFrame(ofat).to_csv(ROOT/'results/tables/ofat_controlled_comparisons.csv', index=False)
    cells = {}
    for s in runs:
        if s['dataset']=='canine' and s['model']=='psan' and s['seed'] in [42,43,44]:
            config = json.loads((ROOT/s['checkpoint']).with_name('config.json').read_text())
            if (config['atoms']==16 and config['init']=='morlet' and not config['axis_aligned']
                and not config['fixed_phase'] and s['atom_dropout'] in [0., .2]):
                cells[(s['seed'], s['atom'], s['atom_dropout'])] = s
    order = [('gabor',0.), ('gabor',.2), ('gaussian',0.), ('gaussian',.2)]
    complete_seeds = [seed for seed in [42,43,44] if all((seed,*cell) in cells for cell in order)]
    simple_rows, joint_rows, across_rows, descriptive = [], [], [], []
    if complete_seeds:
        reference = None; predictions = []
        for seed in complete_seeds:
            seed_predictions = []
            for atom, ad in order:
                f, p, _, _ = read_new(prediction_path(cells[(seed,atom,ad)]))
                if reference is None: reference = f
                if not (np.array_equal(reference.path, f.path) and np.array_equal(reference.label, f.label)
                        and np.array_equal(reference.group_id, f.group_id)):
                    raise ValueError('Unpaired factorial cell')
                seed_predictions.append(p.argmax(1))
            predictions.append(seed_predictions)
        y = reference.label.to_numpy(dtype=int)
        predictions = np.asarray(predictions)
        _, codes = np.unique(reference.group_id, return_inverse=True); groups = int(codes.max()+1)
        def estimate(weights):
            return factorial_scores([[weighted_macro_f1(y,p,weights) for p in seed_preds] for seed_preds in predictions])
        observed = estimate(np.ones(len(y)))
        distribution = np.empty((args.bootstrap,len(complete_seeds),7))
        crossed = np.empty((args.bootstrap,3)); rng = np.random.default_rng(args.seed)
        seed_rng = np.random.default_rng(args.seed+10000)
        for i in range(args.bootstrap):
            weights = np.bincount(rng.integers(0,groups,groups),minlength=groups)[codes]
            distribution[i] = estimate(weights)
            selected_seeds = seed_rng.integers(0,len(complete_seeds),len(complete_seeds))
            crossed[i] = distribution[i,selected_seeds,4:].mean(0)
        for j, seed in enumerate(complete_seeds):
            for k, name in enumerate(SIMPLE):
                simple_rows.append(dict(seed=seed, contrast=name,
                    **{('delta_macro_f1' if key=='estimate' else key):value for key,value in
                       tail_summary(observed[j,k],distribution[:,j,k],plan['factorial_family']).items()}))
            for k, name in enumerate(EFFECTS):
                joint_rows.append(dict(seed=seed, effect=name,
                    **tail_summary(observed[j,k+4],distribution[:,j,k+4],plan['factorial_family']),
                    uncertainty='test visual groups conditional on this seed'))
            descriptive.append(dict(seed=seed,**dict(zip(EFFECTS,observed[j,4:]))))
        for k,name in enumerate(EFFECTS):
            value = tail_summary(observed[:,k+4].mean(),crossed[:,k],plan['factorial_family'])
            across_rows.append(dict(effect=name,seeds=','.join(map(str,complete_seeds)),runs=len(complete_seeds),
                **value, seed_sd=float(observed[:,k+4].std(ddof=1)) if len(complete_seeds)>1 else None,
                test_only_ci_low=float(np.quantile(distribution[:,:,k+4].mean(1),.025)),
                test_only_ci_high=float(np.quantile(distribution[:,:,k+4].mean(1),.975)),
                uncertainty='crossed empirical seed/visual-group bootstrap; few seeds, not a population guarantee'))
    for name,rows in [('factorial_contrasts',simple_rows),('factorial_joint_effects',joint_rows),
                      ('factorial_across_seeds',across_rows),('factorial_effects',descriptive)]:
        pd.DataFrame(rows).to_csv(ROOT/'results/tables'/(name+'.csv'),index=False)
    tuned_rows = []
    tuned = {p.stem:json.loads(p.read_text()) for p in (ROOT/'results/tuning/final_selected').glob('*.json')}
    if 'psan' in tuned:
        for model, s in sorted(tuned.items()):
            if model == 'psan': continue
            f,pa,pb = aligned(prediction_path(tuned['psan']),prediction_path(s))
            tuned_rows.append(dict(seed=42,comparison='Tuned PSAN minus '+model,
                                  **pair_summary(f,pa,pb,args,plan['tuned_family'])))
    if tuned_rows: pd.DataFrame(tuned_rows).to_csv(ROOT/'results/tables/tuned_paired_comparisons.csv',index=False)
    write_json(ROOT/'results/statistical_protocol.json', dict(plan, iterations=args.bootstrap, seed=args.seed,
        protocol_sha256=sha256(protocol_path), primary_family=plan['primary_family'],
        ci='paired visual-group percentile bootstrap conditional on trained models; crossed factorial intervals separate',
        patient_dependence_not_covered=True, completed_primary=len(primary),completed_ofat=len(ofat),
        completed_joint=len(joint_rows),complete_factorial_seeds=complete_seeds,completed_tuned=len(tuned_rows)))
    complete = len(primary)==18 and len(complete_seeds)==3 and len(tuned_rows)==7
    status('statistics','COMPLETED' if complete else 'IN PROGRESS',primary=len(primary),ofat=len(ofat),
           joint_effects=len(joint_rows),factorial_seeds=complete_seeds,tuned=len(tuned_rows))


if __name__=='__main__':
    from types import SimpleNamespace
    run(SimpleNamespace(bootstrap=1000,seed=42))
