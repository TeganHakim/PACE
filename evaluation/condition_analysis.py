#!/usr/bin/env python3
"""Discover RGB/IR input conditions under which PACE beats 50/50 Addition.

Reads outputs from evaluation/eval_metrics.py; performs NO model inference.

Example (run from PACE repository root):
    python evaluation/condition_analysis.py \
        --run evaluation/results/2026-11-17_12-00-00 \
        --min-leaf 20 --confidence 0.5 --examples 8

Requires: numpy, pandas, opencv-python, matplotlib, scikit-learn.

Primary endpoint: matched-box detection F1 (IoU >= 0.5), with false-positive
counts reported independently. Rules are discovered on a train split and
assessed on a disjoint holdout; no result is asserted if evidence is weak.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, _tree, plot_tree

EPS = 1e-9
SIZE = (256, 256)


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='Completed evaluation/eval_metrics.py output folder')
    p.add_argument('--dataset-root', type=Path, help='Override degraded dataset path (usually read from run/config.json)')
    p.add_argument('--output', type=Path, help='Default: <run>/condition_analysis')
    p.add_argument('--confidence', type=float, default=None, help='Default: recorded evaluation confidence, otherwise 0.5')
    p.add_argument('--iou', type=float, default=.5)
    p.add_argument('--min-leaf', type=int, default=20, help='Minimum discovery samples per tree leaf')
    p.add_argument('--max-depth', type=int, default=3)
    p.add_argument('--test-size', type=float, default=.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bootstrap', type=int, default=1000, help='Number of paired image-level bootstrap draws')
    p.add_argument('--examples', type=int, default=8, help='Number of representative result panels')
    a = p.parse_args()
    if not (0 <= a.confidence <= 1 if a.confidence is not None else True):
        p.error('Confidence must be in [0,1]')
    if not 0 < a.iou <= 1 or not 0 < a.test_size < 1 or a.min_leaf < 2 or a.max_depth < 1 or a.bootstrap < 0 or a.examples < 0:
        p.error('Invalid threshold, split, tree, bootstrap, or example count')
    return a


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing evaluation artifact: {path}')
    return json.loads(path.read_text())


def records_by_id(path):
    records = read_json(path)
    if isinstance(records, dict):
        records = records.get('records', records.get('samples', []))
    out = {}
    for row in records:
        sid = str(row['sample_id'])
        if sid in out:
            raise ValueError(f'Duplicate sample ID in {path}: {sid}')
        out[sid] = row
    return out


def bounded_box(b):
    x1, y1, x2, y2 = map(float, b)
    return x1, y1, x2, y2


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = bounded_box(a)
    bx1, by1, bx2, by2 = bounded_box(b)
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    ar_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ar_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = ar_a + ar_b - inter
    return inter / union if union > 0 else 0.


def ground_truth_boxes(record):
    boxes = []
    for gt in record.get('ground_truth', []):
        cx, cy, w, h = [float(gt[k]) for k in ('cx', 'cy', 'width', 'height')]
        boxes.append((int(gt['class_id']), [(cx - w / 2)*256, (cy-h/2)*256,
                                            (cx + w / 2)*256, (cy+h/2)*256]))
    return boxes


def evaluate_boxes(record, confidence, iou_cutoff):
    """Class-aware, one-to-one confidence-greedy matching. Image-level TP/FP/FN."""
    targets = ground_truth_boxes(record)
    used = set()
    preds = sorted((p for p in record.get('predictions', [])
                    if float(p.get('score', 0)) >= confidence),
                   key=lambda p: float(p['score']), reverse=True)
    tp = 0
    for p in preds:
        cls = int(p['class_id'])
        best_id, best_iou = None, iou_cutoff
        for idx, (tcls, tbox) in enumerate(targets):
            if idx in used or cls != tcls:
                continue
            overlap = box_iou(p['box_xyxy'], tbox)
            if overlap >= best_iou:
                best_id, best_iou = idx, overlap
        if best_id is not None:
            used.add(best_id)
            tp += 1
    fp = len(preds) - tp
    fn = len(targets) - tp
    precision = tp / (tp + fp) if tp + fp else (1. if not targets else 0.)
    recall = tp / (tp + fn) if tp + fn else (1. if not preds else 0.)
    # No detections + no GT is an uninformative image, not a perfect F1 win.
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) else float('nan')
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, gt=len(targets))


def load_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), mode)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)


def image_features(image, prefix):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.Canny(np.uint8(gray*255), 75, 150)
    return {
        prefix+'_brightness':float(gray.mean()),
        prefix+'_contrast':float(gray.std()),
        prefix+'_sharpness':float(lap.var()),
        prefix+'_gradient':float(np.hypot(gx,gy).mean()),
        prefix+'_edge_density':float((edges>0).mean()),
        prefix+'_dark_fraction':float((gray<.15).mean()),
        prefix+'_bright_fraction':float((gray>.85).mean()),
    }


def degradation_features(path, clean, mask):
    diff = path.astype(np.float32)/255. - clean.astype(np.float32)/255.
    m = mask
    if not np.any(m):
        return 0., 0.
    if diff.ndim == 3:
        d = np.abs(diff).mean(axis=2)
    else:
        d = np.abs(diff)
    return float(d[m].mean()), float(np.sqrt(np.mean(d[m]**2)))



def reference_structure(fused, rgb_clean, ir_clean, tolerance=2):
    """Edge precision/recall/F1 against the UNION of both clean source edge maps.

    This is a supporting structural proxy, NOT a ground-truth fused-image score.
    Boundary unsupported rate measures uncorroborated edges near 8x8 grid seams.
    """
    gray = lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) if x.ndim==3 else x
    edge = lambda x: cv2.Canny(gray(x),75,150)>0
    ref = edge(rgb_clean) | edge(ir_clean)
    pred = edge(fused)
    kernel = np.ones((2*tolerance+1,2*tolerance+1),np.uint8)
    close_ref = cv2.dilate(ref.astype(np.uint8),kernel).astype(bool)
    close_pred = cv2.dilate(pred.astype(np.uint8),kernel).astype(bool)
    precision=float((pred & close_ref).sum()/pred.sum()) if pred.any() else float('nan')
    recall=float((ref & close_pred).sum()/ref.sum()) if ref.any() else float('nan')
    f1=2*precision*recall/(precision+recall) if np.isfinite(precision) and np.isfinite(recall) and precision+recall else float('nan')
    unsupported=pred & ~close_ref
    h,w=pred.shape
    boundary=np.zeros_like(pred)
    for k in range(1,8):
        y=int(round(k*h/8));x=int(round(k*w/8))
        boundary[max(0,y-2):min(h,y+3),:]=True
        boundary[:,max(0,x-2):min(w,x+3)]=True
    seam=pred & boundary
    return {'reference_edge_f1':f1,'reference_edge_precision':precision,
            'reference_edge_recall':recall,
            'unsupported_edge_fraction':float(unsupported.sum()/pred.sum()) if pred.any() else float('nan'),
            'seam_unsupported_fraction':float((unsupported & boundary).sum()/seam.sum()) if seam.any() else float('nan')}

def bootstrap_ci(values, count, seed):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a) or count == 0:
        return (float('nan'),float('nan'))
    gen = np.random.default_rng(seed)
    means = np.empty(count)
    for i in range(count):
        means[i] = gen.choice(a, size=len(a), replace=True).mean()
    return tuple(map(float,np.quantile(means, [.025,.975])))


def group_stats(group, bootstrap, seed):
    valid = group['delta_f1'].dropna()
    ci_lo, ci_hi = bootstrap_ci(valid, bootstrap, seed)
    return {
        'n_images':len(group), 'n_scored':len(valid),
        'pace_f1_mean':float(group['pace_f1'].mean()),
        'addition_f1_mean':float(group['addition_f1'].mean()),
        'delta_f1_mean':float(valid.mean()) if len(valid) else float('nan'),
        'delta_f1_ci95_low':ci_lo,'delta_f1_ci95_high':ci_hi,
        'pace_win_rate':float((valid>0).mean()) if len(valid) else float('nan'),
        'tie_rate':float((valid==0).mean()) if len(valid) else float('nan'),
        'delta_recall_mean':float(group['delta_recall'].mean()),
        'delta_fp_mean':float(group['delta_fp'].mean()),
        'pace_fp_per_image':float(group['pace_fp'].mean()),
        'addition_fp_per_image':float(group['addition_fp'].mean()),
        'pace_tp_total':int(group['pace_tp'].sum()),
        'addition_tp_total':int(group['addition_tp'].sum()),
    }


def leaf_rules(tree, names):
    children_left, children_right = tree.children_left, tree.children_right
    thresholds, features = tree.threshold, tree.feature
    out = {}
    def go(node, clauses):
        if features[node] == _tree.TREE_UNDEFINED:
            out[node] = ' AND '.join(clauses) if clauses else 'All images'
            return
        feature = names[features[node]]
        cut = thresholds[node]
        go(children_left[node], clauses+[f'{feature} <= {cut:.4g}'])
        go(children_right[node], clauses+[f'{feature} > {cut:.4g}'])
    go(0, [])
    return out


def create_panel(row, dataset, run, out):
    sid = row['sample_id']
    sources = [
        ('RGB (degraded input)', dataset/'test'/'rgb'/f'{sid}.jpg'),
        ('IR (degraded input)', dataset/'test'/'ir'/f'{sid}.jpg'),
        ('PACE fused', run/'pace'/'degraded'/'fused'/f'{sid}.png'),
        ('Addition fused', run/'addition'/'degraded'/'fused'/f'{sid}.png'),
    ]
    fig, axs = plt.subplots(1,4,figsize=(14,4))
    for ax,(title,path) in zip(axs,sources):
        img = load_image(path, grayscale=True)
        ax.imshow(img,cmap='gray',vmin=0,vmax=255)
        ax.set_title(title,fontsize=10)
        ax.axis('off')
    fig.suptitle(f"{sid} | {row['degraded_modality']} {row['degradation_type']} | "
                 f"Δ detection F1={row['delta_f1']:+.3f}, ΔFP={int(row['delta_fp']):+d}",fontsize=11)
    fig.tight_layout()
    fig.savefig(out/f'{sid}.png',dpi=150,bbox_inches='tight')
    plt.close(fig)


def main():
    a = arguments()
    run = a.run.resolve()
    config = read_json(run/'config.json')
    dataset = (a.dataset_root or Path(config['dataset'])).resolve()
    out = (a.output or run/'condition_analysis').resolve()
    out.mkdir(parents=True,exist_ok=True)
    confidence = a.confidence if a.confidence is not None else float(config.get('confidence', .5))
    p_preds = records_by_id(run/'pace'/'degraded'/'predictions.json')
    a_preds = records_by_id(run/'addition'/'degraded'/'predictions.json')
    if set(p_preds) != set(a_preds):
        raise ValueError('PACE and Addition predictions use different image IDs; cannot compare paired outcomes')
    metadata = pd.read_csv(dataset/'metadata.csv',dtype={'pair_id':str}).fillna('')
    metadata['sample_id'] = metadata['pair_id'].map(lambda x:Path(str(x)).stem)
    if metadata['sample_id'].duplicated().any():
        metadata = metadata[metadata.get('split','test').astype(str).eq('test')]
    meta = metadata.set_index('sample_id')
    clean_root = run/'_dataset_clean'/'test'
    rows=[]
    for i,sid in enumerate(sorted(p_preds)):
        if sid not in meta.index:
            raise KeyError(f'Missing metadata for {sid}')
        m = meta.loc[sid]
        if isinstance(m,pd.DataFrame):
            raise ValueError(f'Ambiguous duplicate metadata for {sid}')
        rgb_path = dataset/'test'/'rgb'/f'{sid}.jpg'
        ir_path = dataset/'test'/'ir'/f'{sid}.jpg'
        rgb,ir = load_image(rgb_path),load_image(ir_path)
        pf = evaluate_boxes(p_preds[sid],confidence,a.iou)
        af = evaluate_boxes(a_preds[sid],confidence,a.iou)
        if ground_truth_boxes(p_preds[sid]) != ground_truth_boxes(a_preds[sid]):
            raise ValueError(f'Ground truth differs between methods on {sid}')
        mask_path = dataset/'test'/'degradation_masks'/f'{sid}.png'
        mask = load_image(mask_path,True)>127 if mask_path.exists() else np.zeros((256,256),dtype=bool)
        modality = str(m.get('modality','')).strip().lower()
        degradation = str(m.get('degradation',m.get('degradation_type','')))
        row={'sample_id':sid,'degraded_modality':modality,'degradation_type':degradation,
             'degradation_coverage':float(mask.mean())}
        row.update(image_features(rgb,'rgb'))
        row.update(image_features(ir,'ir'))
        for field in ('brightness','contrast','sharpness','gradient','edge_density'):
            row[f'rgb_minus_ir_{field}']=row[f'rgb_{field}']-row[f'ir_{field}']
        if clean_root.is_dir():
            for md,img in [('rgb',rgb),('ir',ir)]:
                cp=clean_root/md/f'{sid}.jpg'
                if cp.exists():
                    mean,rmse=degradation_features(img,load_image(cp),mask)
                    row[md+'_corruption_mae']=mean
                    row[md+'_corruption_rmse']=rmse
        for name,metrics in [('pace',pf),('addition',af)]:
            row.update({name+'_'+k:v for k,v in metrics.items()})
        row['delta_f1']=pf['f1']-af['f1']
        row['delta_recall']=pf['recall']-af['recall']
        row['delta_fp']=pf['fp']-af['fp']
        row['delta_tp']=pf['tp']-af['tp']
        if clean_root.is_dir():
            rgb_c=clean_root/'rgb'/f'{sid}.jpg'
            ir_c=clean_root/'ir'/f'{sid}.jpg'
            p_fused=run/'pace'/'degraded'/'fused'/f'{sid}.png'
            a_fused=run/'addition'/'degraded'/'fused'/f'{sid}.png'
            if rgb_c.exists() and ir_c.exists() and p_fused.exists() and a_fused.exists():
                sources=[load_image(rgb_c),load_image(ir_c)]
                for method,path in [('pace',p_fused),('addition',a_fused)]:
                    result=reference_structure(load_image(path,True),*sources)
                    row.update({method+'_'+k:v for k,v in result.items()})
                for metric in ('reference_edge_f1','reference_edge_precision','reference_edge_recall',
                               'unsupported_edge_fraction','seam_unsupported_fraction'):
                    row['delta_'+metric]=row['pace_'+metric]-row['addition_'+metric]
        rows.append(row)
        if (i+1)%100==0: print(f'Analyzed {i+1}/{len(p_preds)} paired images',flush=True)
    data=pd.DataFrame(rows).sort_values('sample_id')
    data.to_csv(out/'per_image_comparison.csv',index=False)
    summary=[]
    for key, group in [('all_images',data),*[(f'modality={k}',v) for k,v in data.groupby('degraded_modality')],
                        *[(f'degradation={k}',v) for k,v in data.groupby('degradation_type')],
                        *[(f'modality={k[0]}; degradation={k[1]}',v) for k,v in data.groupby(['degraded_modality','degradation_type'])]]:
        summary.append({'condition':key,**group_stats(group,a.bootstrap,a.seed)})
    # Add supporting structural outcomes when clean references are available.
    if 'delta_reference_edge_f1' in data.columns:
        for idx,(_,group) in enumerate([('all_images',data),
                         *[(f'modality={k}',v) for k,v in data.groupby('degraded_modality')],
                         *[(f'degradation={k}',v) for k,v in data.groupby('degradation_type')],
                         *[(f'modality={k[0]}; degradation={k[1]}',v) for k,v in data.groupby(['degraded_modality','degradation_type'])]]):
            for key in ('delta_reference_edge_f1','delta_unsupported_edge_fraction','delta_seam_unsupported_fraction'):
                summary[idx][key+'_mean']=float(group[key].mean())
    pd.DataFrame(summary).to_csv(out/'condition_summary.csv',index=False)
    valid=data.dropna(subset=['delta_f1']).copy()
    features=[x for x in data.columns if x.startswith(('rgb_','ir_','degradation_coverage'))
              and pd.api.types.is_numeric_dtype(data[x]) and data[x].notna().all()]
    # Important: do NOT use evaluation outcomes or ground truth as tree inputs.
    # Degradation metadata is summarized independently; tree discovers continuous input conditions.
    feature_exclusions={'rgb_corruption_mae','rgb_corruption_rmse','ir_corruption_mae','ir_corruption_rmse'}
    # Corruption magnitude may be a valid predictor, but depends on access to clean references;
    # leave it out of the main input-only tree for reproducible inference-time rules.
    features=[f for f in features if f not in feature_exclusions]
    if len(valid)>=max(2*a.min_leaf+4,15) and valid['delta_f1'].nunique()>1:
        indices=np.arange(len(valid))
        train_ids,test_ids=train_test_split(indices,test_size=a.test_size,random_state=a.seed)
        train,test=valid.iloc[train_ids].copy(),valid.iloc[test_ids].copy()
        tree=DecisionTreeRegressor(max_depth=a.max_depth,min_samples_leaf=a.min_leaf,random_state=a.seed)
        tree.fit(train[features],train['delta_f1'])
        rules=leaf_rules(tree.tree_,features)
        train['leaf']=tree.apply(train[features]);test['leaf']=tree.apply(test[features])
        discovered=[]
        for leaf,rule in rules.items():
            tr=train[train.leaf==leaf]
            te=test[test.leaf==leaf]
            if len(te)==0:continue
            discovered.append({'leaf':int(leaf),'rule':rule,
                               **{'train_'+k:v for k,v in group_stats(tr,a.bootstrap,a.seed+leaf).items()},
                               **{'test_'+k:v for k,v in group_stats(te,a.bootstrap,a.seed+100+leaf).items()},
                               'holdout_positive_ci':bool(group_stats(te,a.bootstrap,a.seed+100+leaf)['delta_f1_ci95_low']>0)})
        pd.DataFrame(discovered).sort_values('test_delta_f1_mean',ascending=False).to_csv(out/'discovered_rules.csv',index=False)
        valid['split']='train';valid.loc[test.index,'split']='holdout'
        valid['rule_leaf']=tree.apply(valid[features]);valid[['sample_id','split','rule_leaf']].to_csv(out/'rule_assignments.csv',index=False)
        fig,ax=plt.subplots(figsize=(max(12,2.6*a.max_depth+9),6))
        plot_tree(tree,feature_names=features,filled=False,rounded=True,precision=3,fontsize=8,ax=ax)
        ax.set_title('Conditions discovered on training images only (outcome = Δ detection F1)')
        fig.tight_layout();fig.savefig(out/'condition_tree.png',dpi=180,bbox_inches='tight');plt.close(fig)
    else:
        print(f'Insufficient informative samples ({len(valid)}) for an honest tree; writing descriptive summaries only.')
    # Summarize input modality / degradation conditions without selecting on evaluation outcome.
    fig,ax=plt.subplots(figsize=(10,5))
    c=data.groupby(['degraded_modality','degradation_type'],dropna=False)['delta_f1'].agg(['mean','size']).reset_index()
    labels=[f'{r.degraded_modality} / {r.degradation_type}\n(n={r.size})' for r in c.itertuples(index=False)]
    ax.bar(range(len(c)),c['mean'],color='0.45');ax.axhline(0,color='black',lw=1)
    ax.set_xticks(range(len(c)),labels,rotation=15,ha='right');ax.set_ylabel('Mean PACE − Addition detection F1')
    ax.set_title('Detection performance by known degradation condition')
    fig.tight_layout();fig.savefig(out/'degradation_groups.png',dpi=180);plt.close(fig)
    # Show cases with positive detection improvement and no additional false positives.
    candidates=data[(data['delta_f1']>0)&(data['delta_fp']<=0)].sort_values('delta_f1',ascending=False)
    panels=out/'examples';panels.mkdir(exist_ok=True)
    for _,row in candidates.head(a.examples).iterrows():
        create_panel(row,dataset,run,panels)
    report={'run':str(run),'dataset':str(dataset),'output':str(out),'images':len(data),
            'images_with_defined_f1':len(valid),'confidence':confidence,'iou':a.iou,
            'selected_example_count':min(a.examples,len(candidates)),
            'primary_metric':'per-image class-aware detection F1 at fixed confidence and IoU',
            'supporting_metrics':['recall','false positives per image','TP count','reference edge F1 when clean sources are available','unsupported edges near patch seams'],
            'note':'Reference edge metrics are proxies measured against the union of clean RGB and IR edges, not a ground-truth fused image. Rules are exploratory. Only holdout metrics support generalization; these are not causal findings. An undefined F1 for empty-GT/empty-pred images is excluded from tree fitting.'}
    (out/'analysis_config.json').write_text(json.dumps(report,indent=2))
    print(f"Done: {len(data)} images; {len(valid)} with defined detection F1. Output: {out}")
    print('Overall ΔF1:',summary[0]['delta_f1_mean'],'| ΔFP:',summary[0]['delta_fp_mean'])
    print('Representative panels:',min(a.examples,len(candidates)))


if __name__=='__main__':
    main()
