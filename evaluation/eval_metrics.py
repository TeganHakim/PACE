"""One-command evaluation of existing PACE, CrossFuse, Addition pipelines.

Run from repo root: python evaluation/eval_metrics.py --max-samples 5 --device cpu
Full Jetson:       python evaluation/eval_metrics.py --max-samples 0 --device cuda

Pass --original-root to score robustness against clean original DroneVehicle
images; without this argument, full reference-based fusion scores are omitted.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from common import ROOT, EVAL, METHODS, command, load_json, read_csv, write_csv, write_json, selected_ids, check_run
from fusion_metrics import calculate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=ROOT/'processed_dataset_realistic_degraded_v2')
    p.add_argument('--original-root', type=Path, help='VisDrone-DroneVehicle folder containing test/testimg and test/testimgr')
    p.add_argument('--max-samples', type=int, default=5, help='0 means all samples')
    p.add_argument('--device', choices=('auto','cuda','mps','cpu'), default='auto')
    p.add_argument('--visualizations', type=int, default=12)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--confidence', type=float, default=.5)
    p.add_argument('--nms-iou', type=float, default=.7)
    p.add_argument('--output', type=Path)
    p.add_argument('--skip-clean', action='store_true', help='Skip clean inference and reference metrics')
    p.add_argument('--no-figures', action='store_true')
    return p.parse_args()


def prepare_clean(dataset: Path, original: Path, output: Path, ids: list[str]):
    """Create an evaluation-only clean image cache; original dataset untouched.

    Same 100-pixel crop and JPEG quality 97 as build_realistic_degraded_test.py.
    Annotations copied unchanged from degraded dataset (already cropped).
    """
    for modality, source_folder in (('rgb','testimg'), ('ir','testimgr')):
        dst = output/'test'/modality
        dst.mkdir(parents=True, exist_ok=True)
        for stem in ids:
            path = original/'test'/source_folder/f'{stem}.jpg'
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(f'Cannot read original clean image {path}')
            if min(raw.shape[:2]) <= 200:
                raise ValueError(f'Original image too small for 100px crop: {path}')
            cropped = raw[100:-100,100:-100]
            if not cv2.imwrite(str(dst/f'{stem}.jpg'), cropped, [cv2.IMWRITE_JPEG_QUALITY,97]):
                raise IOError('Failed to write clean image')
    for folder in ('rgb_annotations','ir_annotations'):
        dst = output/'test'/folder
        dst.mkdir(parents=True, exist_ok=True)
        src = dataset/'test'/folder
        if not src.is_dir():
            raise FileNotFoundError(src)
        for stem in ids:
            xml = src/f'{stem}.xml'
            if not xml.is_file():
                raise FileNotFoundError(xml)
            shutil.copy2(xml, dst/xml.name)
    # Pre-existing evaluators use the old column `degraded` for subset grouping.
    with (output/'metadata.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair_id','split','degraded'])
        w.writeheader()
        w.writerows({'pair_id': sid, 'split':'test','degraded':'False'} for sid in ids)


def adapt_metadata(dataset: Path, destination: Path, ids: list[str]):
    """Add legacy metadata aliases without changing image or annotation contents."""
    destination.mkdir(parents=True, exist_ok=True)
    src = dataset/'metadata.csv'
    if not src.is_file():
        raise FileNotFoundError(src)
    rows = read_csv(src)
    by_id = {Path(row['pair_id']).stem:row for row in rows}
    if any(sid not in by_id for sid in ids):
        raise ValueError('Missing realistic degradation metadata for selected samples')
    with (destination/'metadata.csv').open('w', newline='') as f:
        columns = list(dict.fromkeys([*rows[0].keys(),'split','degraded','degradation_type']))
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for sid in ids:
            r = dict(by_id[sid]);r['split']='test';r['degraded']='True'
            r['degradation_type'] = r.get('degradation',r.get('degradation_type',''))
            w.writerow(r)
    for folder in ('rgb','ir','rgb_annotations','ir_annotations','degradation_masks'):
        source = dataset/'test'/folder
        dest = destination/'test'/folder
        if not source.is_dir():
            raise FileNotFoundError(source)
        dest.parent.mkdir(parents=True,exist_ok=True)
        # Keep the large dataset in place; link folders, not individual files.
        if not dest.exists():
            try:
                dest.symlink_to(source.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(source,dest)


def run_method(name: str, dataset: Path, out: Path, a):
    shared = ('--dataset-root',dataset,'--split','test','--max-samples',a.max_samples,
              '--output',out,'--device',a.device,'--visualizations',a.visualizations,
              '--warmup',a.warmup,'--seed',a.seed,'--confidence',a.confidence,
              '--nms-iou',a.nms_iou)
    if name == 'pace':
        command(EVAL/'pace_e2e.py', *shared)
    elif name == 'crossfuse':
        command(EVAL/'crossfuse_e2e.py', *shared)
    else:
        command(EVAL/'addition_e2e.py','--dataset-root',dataset,'--max-samples',a.max_samples,
                '--output',out,'--device',a.device,'--visualizations',a.visualizations,
                '--warmup',a.warmup,'--seed',a.seed,'--confidence',a.confidence,
                '--nms-iou',a.nms_iou)


def number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):
        return None


def matched_confidence(prediction_records, threshold=.5):
    """Class-aware 1:1 score-greedy IoU>=.5 matching to ground truth.

    Returns mean score of matched true positives (not all predicted boxes),
    fraction of GT objects matched, number of GT objects.
    """
    scores = []
    matched, n_gt = 0,0
    for record in prediction_records:
        targets = record.get('ground_truth',[])
        n_gt += len(targets)
        used = set()
        predictions = sorted(record.get('predictions',[]), key=lambda p:p.get('score',0),reverse=True)
        for pred in predictions:
            if pred.get('score',0) < threshold:
                continue
            x1,y1,x2,y2 = pred['box_xyxy']
            area_p = max(0,x2-x1)*max(0,y2-y1)
            best,best_iou=None,0.
            for i,tgt in enumerate(targets):
                if i in used or int(tgt['class_id'])!=int(pred['class_id']):
                    continue
                # Existing evaluators save GT in normalized cxcywh; predictions are at 256px.
                cx,cy,w,h = [float(tgt[k])*256. for k in ('cx','cy','width','height')]
                tx1,ty1,tx2,ty2 = cx-w/2,cy-h/2,cx+w/2,cy+h/2
                ix=max(0,min(x2,tx2)-max(x1,tx1))
                iy=max(0,min(y2,ty2)-max(y1,ty1))
                inter=ix*iy
                union=area_p+w*h-inter
                iou=inter/union if union>0 else 0.
                if iou>best_iou:
                    best,best_iou=i,iou
            if best is not None and best_iou>=.5:
                used.add(best);matched+=1;scores.append(float(pred['score']))
    return {'detection_confidence_matched':float(np.mean(scores)) if scores else None,
            'confidence_matched_count':matched,'confidence_ground_truth_count':n_gt,
            'matched_object_fraction':matched/n_gt if n_gt else None}


def metric_aggregate(values):
    arr=np.asarray([v for v in values if v is not None and math.isfinite(float(v))],dtype=float)
    return {'mean':float(arr.mean()) if arr.size else None,'std':float(arr.std(ddof=1)) if arr.size>1 else None,'n':int(arr.size)}


def summarize(folder: Path, ids: list[str], annotated: dict, degraded: Path,
              clean_data: Path|None, has_clean: bool):
    results=[]
    metrics=[]
    for method in METHODS:
        run=folder/method/'degraded'
        summary=load_json(run/'summary.json')
        latency=load_json(run/'latency.json')
        preds=load_json(run/'predictions.json')
        detection=summary.get('detection',{})
        c=matched_confidence(preds)
        for k in ('map50','recall'):
            pass
        row={'method':method,'map50':number(detection.get('map50',detection.get('map_50'))),
             'recall':number(detection.get('recall')),
             **c}
        # Confidence from the existing JSON might be stored alongside detection metrics.
        if row['recall'] is None:
            row['recall']=c['matched_object_fraction']
        def lat_val(key):
            v=latency.get(key)
            return number(v.get('mean') if isinstance(v,dict) else v)
        row['latency_ms']=lat_val('end_to_end_ms')
        row['peak_gpu_mb']=lat_val('cuda_peak_allocated_mb')
        for sid in ids:
            m=annotated[sid]
            sample={'method':method,'sample_id':sid,'modality':m.get('modality',''),
                    'degradation_type':m.get('degradation_type','')}
            if has_clean and clean_data is not None:
                sample.update(calculate(degraded,clean_data,run,folder/method/'clean',sid,m['modality']))
            metrics.append(sample)
        for name in ('degradation_transfer_ratio','edge_f1','qabf'):
            row[name]=metric_aggregate([r.get(name) for r in metrics if r['method']==method])['mean']
        results.append(row)
    out=folder/'metrics'
    write_csv(out/'comparison.csv',results)
    write_csv(out/'fusion_per_sample.csv',metrics)
    write_csv(out/'system.csv',[{'method':r['method'],'latency_ms':r['latency_ms'],'peak_gpu_mb':r['peak_gpu_mb']} for r in results])
    write_csv(out/'detection.csv',[{'method':r['method'],'map50':r['map50'],'recall':r['recall'],
                                   'detection_confidence_matched':r['detection_confidence_matched'],
                                   'confidence_matched_count':r['confidence_matched_count'],
                                   'confidence_ground_truth_count':r['confidence_ground_truth_count']} for r in results])
    write_csv(out/'fusion_quality.csv',[{'method':r['method'],**{k:r[k] for k in ('degradation_transfer_ratio','edge_f1','qabf')}} for r in results])
    return results


def main():
    a=parse_args()
    if a.max_samples<0 or a.warmup<0 or a.visualizations<0:
        raise ValueError('Counts must be nonnegative')
    source=a.dataset_root.resolve()
    ids=selected_ids(source,a.max_samples)
    if not a.skip_clean and not a.original_root:
        raise ValueError('Provide --original-root for all eight metrics, or --skip-clean for detection/system only')
    timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder=(a.output or EVAL/'results'/timestamp).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output folder: {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    indexed={Path(x['pair_id']).stem:x for x in read_csv(source/'metadata.csv')}
    annotated={sid:{**indexed[sid],'degradation_type':indexed[sid].get('degradation','')} for sid in ids}
    staged=folder/'_dataset_degraded'
    adapt_metadata(source,staged,ids)
    clean_data=None
    if not a.skip_clean:
        clean_data=folder/'_dataset_clean'
        prepare_clean(source,a.original_root.resolve(),clean_data,ids)
    config={'timestamp':timestamp,'repository':'TeganHakim/PACE',
            'dataset':str(source),'original_root':str(a.original_root) if a.original_root else None,
            'sample_count':len(ids),'first_id':ids[0],'last_id':ids[-1],
            'device':a.device,'seed':a.seed,'warmup':a.warmup,'confidence':a.confidence,
            'nms_iou':a.nms_iou,'platform':platform.platform(),'python':sys.version}
    try:
        config['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        config['git_commit']='unavailable'
    write_json(folder/'config.json',config)
    for name in METHODS:
        for label, data in [('degraded',staged),('clean',clean_data)]:
            if data is None:
                continue
            target=folder/name/label
            print(f'\n=== {name}: {label} ({len(ids)} images) ===',flush=True)
            run_method(name,data,target,a)
            check_run(target,ids)
    rows=summarize(folder,ids,annotated,staged,clean_data,clean_data is not None)
    if not a.no_figures:
        command(EVAL/'visualize_results.py','--run',folder,'--visualizations',a.visualizations)
    print('\nEvaluation complete:',folder)
    for row in rows:
        print(row['method'], 'mAP@50:', row['map50'], 'recall:',row['recall'],
              'transfer:',row['degradation_transfer_ratio'])


if __name__=='__main__':
    main()
