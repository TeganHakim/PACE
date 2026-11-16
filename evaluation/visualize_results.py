"""Preserve the seven original PACE/CrossFuse figures and add 3-way comparisons."""
from __future__ import annotations
import argparse
import math
import shutil
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import EVAL, METHODS, command, read_csv, load_json

COLORS = {'pace':'#D17842','crossfuse':'#375A7F','addition':'#669C82'}
NAMES = {'pace':'PACE','crossfuse':'CrossFuse','addition':'Addition'}
METRICS = [
    ('map50','01_map50.png','mAP@50','higher'),
    ('recall','02_recall.png','Detection recall','higher'),
    ('detection_confidence_matched','03_confidence.png','Matched detection confidence','higher'),
    ('latency_ms','04_latency.png','End-to-end latency (ms)','lower'),
    ('peak_gpu_mb','05_memory.png','Peak allocated CUDA memory (MB)','lower'),
    ('degradation_transfer_ratio','06_degradation_transfer.png','Degradation transfer ratio','lower'),
    ('edge_f1','07_edge_f1.png','Edge stability F1','higher'),
    ('qabf','08_qabf.png','Q_AB/F (clean sources)','higher'),
]


def setup_style():
    plt.rcParams.update({'font.family':'sans-serif','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.spines.left':False,
                         'figure.facecolor':'white','axes.facecolor':'white',
                         'savefig.facecolor':'white','axes.titleweight':'bold'})


def scalar(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):
        return None


def bar_metric(rows, key, filename, title, direction, dest):
    observed=[(r['method'],scalar(r.get(key))) for r in rows]
    observed=[(name,val) for name,val in observed if val is not None]
    if not observed:
        return
    fig,ax=plt.subplots(figsize=(7.8,4.0),layout='constrained')
    x=np.arange(len(observed))
    vals=[v for _,v in observed]
    ax.bar(x,vals,width=.55,color=[COLORS[name] for name,_ in observed])
    ax.set_xticks(x,[NAMES[name] for name,_ in observed]);ax.set_ylabel(title)
    ax.set_title(f'{title}  |  {direction} is better',loc='left',pad=12)
    ax.grid(axis='y',color='#E9EDF1');ax.set_axisbelow(True)
    span=max(max(abs(v) for v in vals),.01)
    lo=min(0.,min(vals)*1.15);hi=max(vals)*1.22 if max(vals)>0 else span*.2
    ax.set_ylim(lo,hi)
    for i,v in enumerate(vals):
        ax.annotate(f'{v:.4f}' if abs(v)<2 else f'{v:.2f}',(i,v),xytext=(0,5),
                    textcoords='offset points',ha='center',fontsize=10)
    fig.savefig(dest/filename,dpi=220,bbox_inches='tight');plt.close(fig)


def grouped_quality(rows, key, title, output):
    groups=['RGB noise','IR noise','RGB blur','IR blur']
    def group(r):
        kind=str(r.get('degradation_type','')).lower()
        typ='noise' if 'noise' in kind else 'blur' if 'blur' in kind else 'other'
        return f"{str(r.get('modality','')).upper()} {typ}"
    fig,ax=plt.subplots(figsize=(9.7,4.2),layout='constrained')
    xs=np.arange(len(groups));width=.24
    found=False
    for i,method in enumerate(METHODS):
        vals=[]
        for name in groups:
            v=[scalar(r.get(key)) for r in rows if r['method']==method and group(r)==name]
            v=[a for a in v if a is not None]
            vals.append(float(np.mean(v)) if v else np.nan)
        if any(np.isfinite(v) for v in vals):found=True
        ax.bar(xs+(i-1)*width,vals,width=width,label=NAMES[method],color=COLORS[method])
    if found:
        ax.set_xticks(xs,groups);ax.set_ylabel(title)
        ax.set_title(f'{title} by degradation condition',loc='left',pad=12)
        ax.grid(axis='y',color='#E9EDF1');ax.set_axisbelow(True);ax.legend(frameon=False,ncol=3)
        fig.savefig(output,dpi=220,bbox_inches='tight')
    plt.close(fig)


def read_gray(path):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)


def qualitative(run, limit):
    if limit <= 0:return
    root=run/'_dataset_degraded'/'test'
    ids=[r['sample_id'] for r in read_csv(run/'pace'/'degraded'/'per_sample.csv')][:limit]
    out=run/'figures'/'qualitative';out.mkdir(parents=True,exist_ok=True)
    preds={m:{r['sample_id']:r for r in load_json(run/m/'degraded'/'predictions.json')} for m in METHODS}
    for sid in ids:
        images=[read_gray(root/'rgb'/f'{sid}.jpg'),read_gray(root/'ir'/f'{sid}.jpg')]
        images.extend(read_gray(run/m/'degraded'/'fused'/f'{sid}.png') for m in ('addition','crossfuse','pace'))
        mask=read_gray(root/'degradation_masks'/f'{sid}.png')
        images.insert(2,mask)
        titles=['RGB input','IR input','Degradation mask','Addition','CrossFuse','PACE']
        fig,axes=plt.subplots(2,3,figsize=(13.2,7.7),layout='constrained')
        for ax,img,title in zip(axes.flat,images,titles):
            ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.axis('off');ax.set_title(title)
        fig.suptitle(f'RGB/IR fusion comparison • {sid}',fontsize=14)
        fig.savefig(out/f'{sid}_fusion.png',dpi=170,bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
        for ax,method in zip(axes,METHODS):
            ax.imshow(read_gray(run/method/'degraded'/'fused'/f'{sid}.png'),cmap='gray',vmin=0,vmax=255)
            record=preds[method][sid]
            for gt in record.get('ground_truth',[]):
                cx,cy,w,h=[float(gt[k])*256 for k in ('cx','cy','width','height')]
                ax.add_patch(plt.Rectangle((cx-w/2,cy-h/2),w,h,fill=False,edgecolor='orange',lw=.85))
            for p in record.get('predictions',[]):
                if float(p['score'])<.5:continue
                x1,y1,x2,y2=p['box_xyxy']
                ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',lw=1))
                ax.text(x1,max(0,y1-3),f"{p['score']:.2f}",color='lime',fontsize=7)
            ax.axis('off');ax.set_title(NAMES[method])
        fig.suptitle(f'Detection • {sid}  (orange GT; green prediction)',fontsize=13)
        fig.savefig(out/f'{sid}_detections.png',dpi=170,bbox_inches='tight');plt.close(fig)


def preserve_original_seven(run, limit):
    """Call untouched evaluation/compare.py on actual PACE/CrossFuse run artifacts."""
    target=run/'figures'/'existing'
    tmp=run/'_legacy_comparison';tmp.mkdir(parents=True,exist_ok=True)
    # Preserve the original plotting script while supplying BOTH clean and
    # degraded subset summaries (its standard compare call only sees one run).
    legacy=[]
    for method in ('crossfuse','pace'):
        source=run/method/'degraded'
        dest=tmp/f'{method}_input';dest.mkdir(parents=True,exist_ok=True)
        for filename in ('per_sample.csv','predictions.json','latency.json'):
            shutil.copy2(source/filename,dest/filename)
        source_viz=source/'visualizations'
        if source_viz.exists():
            shutil.copytree(source_viz,dest/'visualizations',dirs_exist_ok=True)
        report=load_json(source/'summary.json')
        clean=run/method/'clean'/'summary.json'
        if clean.is_file():
            clean_report=load_json(clean)
            report.setdefault('subsets',{})['clean']={
                'count':clean_report.get('configuration',{}).get('samples_evaluated'),
                **clean_report.get('detection',{})}
        import json
        (dest/'summary.json').write_text(json.dumps(report,indent=2))
        legacy.append(dest)
    command(EVAL/'compare.py',legacy[0],legacy[1],
            '--output',tmp,'--visualizations',limit)
    target.mkdir(parents=True,exist_ok=True)
    for src in sorted((tmp/'figures').glob('*.png')):
        shutil.copy2(src,target/src.name)
    if (tmp/'visualization_pairs').is_dir():
        shutil.copytree(tmp/'visualization_pairs',target/'visualization_pairs',dirs_exist_ok=True)
    for name in ('comparison.json','statistics.json','comparison_report.md'):
        path=tmp/name
        if path.is_file():shutil.copy2(path,target/name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--visualizations',type=int,default=12)
    args=p.parse_args()
    run=args.run.resolve();setup_style()
    out=run/'figures'/'metrics';out.mkdir(parents=True,exist_ok=True)
    preserve_original_seven(run,args.visualizations)
    rows=read_csv(run/'metrics'/'comparison.csv')
    for key,filename,title,direction in METRICS:
        bar_metric(rows,key,filename,title,direction,out)
    per=read_csv(run/'metrics'/'fusion_per_sample.csv')
    for key,title in (('degradation_transfer_ratio','Degradation transfer ratio'),
                      ('edge_f1','Edge stability F1'),('qabf','Q_AB/F')):
        grouped_quality(per,key,title,out/f'{key}_by_condition.png')
    qualitative(run,args.visualizations)
    print('Figures written to',run/'figures')


if __name__=='__main__':
    main()
