#!/usr/bin/env python3
"""Reproducible CrossFuse vs PACE end-to-end comparison.

From repository root:
  python evaluation/compare.py --num-images 100

Legacy compare-only invocation remains supported:
  python evaluation/compare.py BASELINE_RUN PACE_RUN [--output DIR]

Requires numpy, matplotlib, and seaborn for comparison; evaluation
requires the existing repository ML dependencies and checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
METRICS = [('precision','Precision'),('recall','Recall'),('f1','F1'),('map50','mAP@50'),('map50_95','mAP@50:95')]
COLORS = {'CrossFuse':'#375A7F','PACE':'#D17842'}


def args_parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('crossfuse', nargs='?', type=Path, help='Existing baseline run (legacy compare-only mode)')
    p.add_argument('pace', nargs='?', type=Path, help='Existing PACE run (legacy compare-only mode)')
    p.add_argument('--num-images', '--max-samples', type=int, default=100, dest='num_images', help='0 evaluates entire split')
    p.add_argument('--split', choices=['train','val','test'], default='test')
    p.add_argument('--device', choices=['auto','cuda','mps','cpu'], default='auto')
    p.add_argument('--dataset-root', type=Path, default=ROOT/'processed_dataset')
    p.add_argument('--output', type=Path)
    p.add_argument('--visualizations', type=int, default=12, help='Number of evaluator sample panels and matched visualization pairs (default: 12; 0 disables)')
    p.add_argument('--confidence', type=float, default=.5)
    p.add_argument('--nms-iou', type=float, default=.7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--dpi', type=int, default=220)
    a = p.parse_args(argv)
    if (a.crossfuse is None) != (a.pace is None):
        p.error('Provide both positional run directories or neither.')
    if min(a.num_images, a.visualizations, a.bootstrap, a.warmup) < 0:
        p.error('Counts must be nonnegative.')
    return a


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required evaluation artifact: {path}')
    return json.loads(path.read_text())


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required evaluation artifact: {path}')
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def find_metric(d, key):
    if not isinstance(d, dict):
        return None
    aliases = {'map50':['map50','map_50'], 'map50_95':['map50_95','map','map_50_95']}
    for k in aliases.get(key,[key]):
        v = num(d.get(k))
        if v is not None:
            return v
    return None


def get_run(path):
    summary = load_json(path/'summary.json')
    latency = load_json(path/'latency.json')
    rows = load_csv(path/'per_sample.csv')
    preds = load_json(path/'predictions.json')
    byid = {str(r['sample_id']):r for r in rows}
    bypred = {str(r['sample_id']):r for r in preds}
    if len(byid)!=len(rows) or len(bypred)!=len(preds):
        raise ValueError(f'Duplicate sample IDs in {path}')
    return {'path':path, 'summary':summary,'latency':latency,'rows':byid,'preds':bypred}


def run_evaluations(a):
    suffix = 'all' if a.num_images == 0 else str(a.num_images)
    baseline = ROOT/'baseline'/'CrossFuse'/'evaluation'/f'baseline_epoch_28_{a.split}_{suffix}'
    pace = ROOT/'pace'/'evaluation'/f'pace_{a.split}_{suffix}'
    common = ['--split',a.split,'--max-samples',str(a.num_images),'--device',a.device,
              '--dataset-root',str(a.dataset_root.resolve()),'--visualizations',str(a.visualizations),
              '--confidence',str(a.confidence),'--nms-iou',str(a.nms_iou),
              '--seed',str(a.seed),'--warmup',str(a.warmup)]
    for script,out in [(ROOT/'baseline'/'CrossFuse'/'evaluate.py',baseline),(ROOT/'pace'/'evaluate.py',pace)]:
        cmd = [sys.executable,str(script),*common,'--output',str(out)]
        print('\nRunning:', ' '.join(cmd), flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
    return baseline,pace


def common_ids(a,b):
    ids = sorted(set(a['rows']) & set(b['rows']))
    if not ids:
        raise ValueError('No matching sample IDs; cannot perform paired comparison.')
    if set(a['rows']) != set(b['rows']) or set(a['preds']) != set(b['preds']):
        raise ValueError('Evaluation sample IDs differ. Refusing an unpaired comparison; check dataset selection.')
    return ids


def bootstrap_mean_diff(x,y,seed,nboot):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)!=len(y) or not len(x):
        return None
    diff=y-x
    rng=np.random.default_rng(seed)
    if not nboot:
        return {'estimate':float(diff.mean()),'ci95':None,'n':len(diff)}
    draws=np.empty(nboot)
    for i in range(nboot):
        idx=rng.integers(0,len(diff),len(diff))
        draws[i]=diff[idx].mean()
    return {'estimate':float(diff.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'n':len(diff)}


def setup_style():
    """Minimal, consistent research-figure styling."""
    sns.set_theme(
        style="white", context="paper", font_scale=1.15,
        rc={
            "figure.dpi": 130, "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.spines.left": False, "axes.titleweight": "bold",
            "axes.titlesize": 13, "axes.labelsize": 10,
            "xtick.labelsize": 9, "ytick.labelsize": 9,
            "legend.frameon": False, "savefig.facecolor": "white",
            "axes.facecolor": "white", "figure.facecolor": "white",
        },
    )


def clean_axis(ax, axis="y"):
    """Keep only subtle, helpful gridlines."""
    ax.grid(False)
    ax.grid(axis=axis, color="#E9EDF1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig,path,dpi):
    fig.savefig(path,dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def paired_series(a,b,ids,key):
    matched=[(sid,num(a['rows'][sid].get(key)),num(b['rows'][sid].get(key))) for sid in ids]
    return [(sid,x,y) for sid,x,y in matched if x is not None and y is not None]


def figure_detection(a,b,out,dpi):
    da=a['summary'].get('detection',{});db=b['summary'].get('detection',{})
    data=[(label,find_metric(da,k),find_metric(db,k)) for k,label in METRICS]
    data=[r for r in data if r[1] is not None and r[2] is not None]
    if not data:return None
    fig,ax=plt.subplots(figsize=(8.5,4.5),layout='constrained')
    ys=np.arange(len(data));h=.31
    for name,offset,index in [('CrossFuse',-h/2,1),('PACE',h/2,2)]:
        vals=[r[index] for r in data]
        bars=ax.barh(ys+offset,vals,height=h*.9,color=COLORS[name],label=name)
        for bar,v in zip(bars,vals):
            ax.text(v+.013,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center',fontsize=9)
    ax.set_yticks(ys,[r[0] for r in data]);ax.invert_yaxis()
    ax.set_xlim(0,1.13);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Detection score (higher is better)')
    ax.set_title('Detection performance',loc='left',pad=12)
    ax.legend(loc='lower right',ncol=2);clean_axis(ax,'x')
    save(fig,out/'01_detection.png',dpi)
    return {'file':'01_detection.png','source':['baseline/summary.json: detection','pace/summary.json: detection'],
            'goal':'Compare dataset-level detection accuracy on matched samples.',
            'method':'Grouped bars show evaluator-reported dataset-level metrics; printed values are point estimates, not confidence intervals.'}

def figure_subsets(a,b,out,dpi):
    """Compact condition comparison with dedicated header and legend space."""
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0),sharex=True)
    found=False
    for ax,subset in zip(axes,['clean','degraded']):
        entries=[]
        for metric,label in [('f1','F1'),('map50','mAP@50')]:
            u=find_metric(a['summary'].get('subsets',{}).get(subset,{}),metric)
            v=find_metric(b['summary'].get('subsets',{}).get(subset,{}),metric)
            if u is not None and v is not None:entries.append((label,u,v))
        if entries:found=True
        ys=np.arange(len(entries));h=.26
        for name,offset,index in [('CrossFuse',-h/2,1),('PACE',h/2,2)]:
            vals=[r[index] for r in entries]
            bars=ax.barh(ys+offset,vals,height=h*.88,color=COLORS[name],label=name)
            for bar,v in zip(bars,vals):
                ax.text(v+.016,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center',fontsize=9)
        ax.set_yticks(ys,[r[0] for r in entries]);ax.invert_yaxis()
        ax.set_xlim(0,1.12);ax.set_xticks([0,.25,.5,.75,1.0])
        ax.set_title(subset.title(),loc='left',pad=15,fontsize=12)
        ax.set_xlabel('Detection score',labelpad=10)
        ax.tick_params(axis='y',pad=12)
        ax.margins(y=.38)
        clean_axis(ax,'x')
    if not found:plt.close(fig);return None
    fig.suptitle('Detection by image condition',x=.09,y=.98,ha='left',fontsize=15,fontweight='bold')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.96,.985),ncol=2,
               fontsize=10,frameon=False,handlelength=1.7,columnspacing=1.6)
    fig.subplots_adjust(left=.13,right=.97,top=.73,bottom=.19,wspace=.32)
    save(fig,out/'02_degradation.png',dpi)
    return {'file':'02_degradation.png','source':['baseline/summary.json: subsets','pace/summary.json: subsets'],
            'goal':'Compare detection performance on clean and degraded images.',
            'method':'Two condition panels with grouped bars for F1 and mAP@50; values are descriptive subset-level estimates.'}

def figure_latency(a,b,ids,out,dpi,seed,nboot):
    pairs=paired_series(a,b,ids,'end_to_end_ms')
    if not pairs:return None,None
    x=np.array([r[1] for r in pairs]);y=np.array([r[2] for r in pairs]);diff=y-x
    fig,ax=plt.subplots(figsize=(7.6,4.6),layout='constrained')
    bp=ax.boxplot([x,y],positions=[0,1],widths=.42,patch_artist=True,showfliers=True,
                  medianprops={'color':'#202B36','linewidth':2},
                  flierprops={'marker':'o','markersize':3,'alpha':.35})
    for patch,name in zip(bp['boxes'],['CrossFuse','PACE']):
        patch.set_facecolor(COLORS[name]);patch.set_alpha(.35);patch.set_edgecolor(COLORS[name])
    for i,vals in enumerate([x,y]):
        mean=float(np.mean(vals));median=float(np.median(vals));p95=float(np.quantile(vals,.95))
        ax.scatter(i,mean,marker='D',s=36,color=COLORS[['CrossFuse','PACE'][i]],zorder=5)
        ax.text(i+.26,median,f'Median {median:.1f} ms',fontsize=9,va='center')
        ax.text(i+.26,mean+max(2,mean*.035),f'Mean {mean:.1f} ms',fontsize=8,va='bottom',color=COLORS[['CrossFuse','PACE'][i]])
    ax.set_xlim(-.55,1.9);ax.set_xticks([0,1],['CrossFuse','PACE']);ax.set_ylabel('End-to-end latency (ms)')
    ax.set_title(f'End-to-end latency  |  PACE − CrossFuse: {diff.mean():+.1f} ms mean',loc='left',pad=15)
    clean_axis(ax,'y');save(fig,out/'03_latency.png',dpi)
    return {'file':'03_latency.png','source':['baseline/per_sample.csv: sample_id,end_to_end_ms','pace/per_sample.csv: sample_id,end_to_end_ms'],
            'goal':'Compare per-image latency distributions on matched samples.',
            'method':'Boxplots show median, IQR, whiskers and outliers; diamonds mark means. Positive PACE-minus-CrossFuse difference means PACE is slower. Paired bootstrap is reported separately.'},bootstrap_mean_diff(x,y,seed,nboot)

def figure_stages(a,b,out,dpi):
    stages=[('Encoder / quality / gate','encoder_ms','encoder_quality_gate_ms'),
            ('Fusion','fusion_ms','selective_fusion_ms'),
            ('Detection + NMS','detection_nms_ms','detection_nms_ms')]
    def mean(run,key):
        v=run['latency'].get(key);return num(v.get('mean')) if isinstance(v,dict) else num(v)
    data=[[mean(run,ka if i==0 else kb) for _,ka,kb in stages] for i,run in enumerate([a,b])]
    if any(v is None for row in data for v in row):return None
    fig,ax=plt.subplots(figsize=(9.2,3.9),layout='constrained')
    stage_colors=['#8A9AA8','#5A7F9B','#B5BEC6']
    for i,(name,vals) in enumerate(zip(['CrossFuse','PACE'],data)):
        left=0
        for j,v in enumerate(vals):
            ax.barh(i,v,left=left,height=.48,color=stage_colors[j],edgecolor='white',linewidth=1.5,
                    label=stages[j][0] if i==0 else None)
            if v>=6:ax.text(left+v/2,i,f'{v:.1f} ms',ha='center',va='center',fontsize=9,color='#14232F',fontweight='normal')
            left+=v
        ax.text(left+2,i,f'{left:.1f} ms total',va='center',fontsize=10,fontweight='bold')
    ax.set_yticks([0,1],['CrossFuse','PACE']);ax.invert_yaxis()
    ax.set_xlim(0,max(map(sum,data))*1.27);ax.set_xlabel('Mean stage runtime (ms)')
    ax.set_title('Where inference time is spent',loc='left',pad=13)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3)
    clean_axis(ax,'x');save(fig,out/'04_stage_latency.png',dpi)
    return {'file':'04_stage_latency.png','source':['baseline/latency.json: encoder_ms,fusion_ms,detection_nms_ms','pace/latency.json: encoder_quality_gate_ms,selective_fusion_ms,detection_nms_ms'],
            'goal':'Compare overall stage costs and expose fusion savings versus gate overhead.',
            'method':'Stacked mean stage times. The first segment is not functionally identical: PACE includes quality estimation and gating. Stage sums can differ slightly from separately timed end-to-end means.'}

def figure_memory(a,b,out,dpi):
    keys=['cuda_peak_allocated_mb','cuda_peak_reserved_mb']
    key=next((k for k in keys if num(a['latency'].get(k,{}).get('mean') if isinstance(a['latency'].get(k),dict) else a['latency'].get(k)) is not None and num(b['latency'].get(k,{}).get('mean') if isinstance(b['latency'].get(k),dict) else b['latency'].get(k)) is not None),None)
    if key is None:return None,None
    def val(r):
        v=r['latency'][key];return num(v.get('mean')) if isinstance(v,dict) else num(v)
    x,y=val(a),val(b)
    fig,ax=plt.subplots(figsize=(7.5,3.1),layout='constrained')
    bars=ax.barh(['CrossFuse','PACE'],[x,y],height=.43,color=[COLORS['CrossFuse'],COLORS['PACE']])
    ax.invert_yaxis()
    for bar,v in zip(bars,[x,y]):
        ax.text(v+max(x,y)*.025,bar.get_y()+bar.get_height()/2,f'{v:.0f} MB',va='center',fontsize=10)
    ax.set_xlim(0,max(x,y)*1.24)
    ax.set_xlabel('Peak CUDA memory (MB; CUDA runs only)')
    ax.set_title('Peak GPU memory',loc='left',pad=12)
    clean_axis(ax,'x')
    save(fig,out/'05_memory.png',dpi)
    return {'file':'05_memory.png','source':[f'baseline/latency.json: {key}',f'pace/latency.json: {key}'],
            'goal':'Compare observed GPU allocation overhead.',
            'method':'Single run-level peak per model; descriptive only, not a sample-level distribution or confidence interval.'},{'crossfuse_mb':x,'pace_mb':y,'pace_minus_crossfuse_mb':y-x}


def figure_gate(a,b,ids,out,dpi):
    pairs=[]
    for sid in ids:
        frac=num(b['rows'][sid].get('active_fraction'))
        latency=num(b['rows'][sid].get('end_to_end_ms'))
        if frac is not None and latency is not None:pairs.append((frac*100,latency))
    if not pairs:return None
    bins=[('0%',lambda x:x==0),('>0–10%',lambda x:0<x<=10),('>10–25%',lambda x:10<x<=25),('>25%',lambda x:x>25)]
    groups=[(label,np.array([t for f,t in pairs if pred(f)],dtype=float)) for label,pred in bins]
    groups=[(label,v) for label,v in groups if len(v)]
    fig,ax=plt.subplots(figsize=(7.7,3.9),layout='constrained')
    xs=np.arange(len(groups));means=[v.mean() for _,v in groups]
    bars=ax.bar(xs,means,width=.54,color=COLORS['PACE'],alpha=.86)
    for bar,(_,v) in zip(bars,groups):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(means)*.035,
                f'{v.mean():.1f} ms\nn={len(v)}',ha='center',va='bottom',fontsize=9)
    ax.set_ylim(0,max(means)*1.32);ax.set_xticks(xs,[g[0] for g in groups])
    ax.set_xlabel('Patches selected for fusion');ax.set_ylabel('Mean PACE end-to-end latency (ms)')
    ax.set_title('Gate activity versus runtime',loc='left',pad=12)
    clean_axis(ax,'y');save(fig,out/'06_gate_latency.png',dpi)
    return {'file':'06_gate_latency.png','source':['pace/per_sample.csv: active_fraction,end_to_end_ms'],
            'goal':'Show how observed runtime varies with the fraction of patches fused.',
            'method':'Nonoverlapping gate-activity bins, including a separate zero-activation bin. Bars are per-bin means and labels give sample counts; descriptive only.'}

def figure_tradeoff(a,b,out,dpi):
    def m(run,key):
        v=run['latency'].get(key,{});return num(v.get('mean')) if isinstance(v,dict) else num(v)
    be=m(a,'encoder_ms');bf=m(a,'fusion_ms');pe=m(b,'encoder_quality_gate_ms');pf=m(b,'selective_fusion_ms')
    if any(v is None for v in (be,bf,pe,pf)):return None
    saved=bf-pf;overhead=pe-be
    fig,ax=plt.subplots(figsize=(8.3,3.3),layout='constrained')
    values=[saved,overhead];names=['Fusion time saved','Additional encoder / gate cost']
    colors=['#36836C' if saved>=0 else '#B65C49','#B65C49' if overhead>=0 else '#36836C']
    bars=ax.barh(names,values,color=colors,height=.44)
    for bar,v in zip(bars,values):
        ax.text(v+(1 if v>=0 else -1),bar.get_y()+bar.get_height()/2,f'{v:+.1f} ms',
                ha='left' if v>=0 else 'right',va='center',fontsize=10,fontweight='bold')
    ax.axvline(0,color='#74808B',lw=1);ax.invert_yaxis()
    lim=max(1,max(abs(v) for v in values));ax.set_xlim(min(-lim*.2,min(values)*1.2),max(lim*1.32,max(values)*1.2))
    ax.set_xlabel('Time difference (ms)');ax.set_title('Fusion savings versus selection overhead',loc='left',pad=12)
    clean_axis(ax,'x');save(fig,out/'07_tradeoff.png',dpi)
    return {'file':'07_tradeoff.png','source':['baseline/latency.json: encoder_ms,fusion_ms','pace/latency.json: encoder_quality_gate_ms,selective_fusion_ms'],
            'goal':'Quantify whether fusion savings compensate for quality estimation and gating.',
            'method':'Fusion time saved = CrossFuse fusion mean − PACE fusion mean. Additional encoder/gate cost = PACE encoder-quality-gate mean − CrossFuse encoder mean. Positive overhead offsets positive savings.'}


def report_markdown(a,b,ids,figures,stats,output):
    da=a['summary'].get('detection',{});db=b['summary'].get('detection',{})
    lines=['# CrossFuse vs. PACE — paired evaluation','',f'Compared **{len(ids)}** matched samples.',
           '', '## Detection performance','', '| Metric | CrossFuse | PACE | Difference (PACE − CrossFuse) |', '|---|---:|---:|---:|']
    for key,label in METRICS:
        x,y=find_metric(da,key),find_metric(db,key)
        if x is not None and y is not None:lines.append(f'| {label} | {x:.4f} | {y:.4f} | {y-x:+.4f} |')
    lines+=['','Detection metrics above are evaluator-reported **dataset-level point estimates**. No p-values or confidence intervals are assigned to mAP without recomputing mAP over paired bootstrap resamples.','', '## Latency and memory','']
    latency=stats.get('paired_latency_mean_difference_ms')
    if latency:
        ci=latency['ci95'];lines.append(f"Mean paired end-to-end latency difference (PACE − CrossFuse): **{latency['estimate']:+.3f} ms** (n={latency['n']}).")
        if ci:lines.append(f'Paired percentile bootstrap 95% CI: [{ci[0]:+.3f}, {ci[1]:+.3f}] ms.')
    memory=stats.get('memory')
    if memory:lines.append(f"Recorded peak GPU memory: CrossFuse {memory['crossfuse_mb']:.1f} MB; PACE {memory['pace_mb']:.1f} MB. This is **one run-level measurement per model**, not an inferential estimate.")
    lines+=['','## Figures: provenance, interpretation, and purpose','']
    for i,f in enumerate(figures,1):
        lines += [f"### {i}. `{f['file']}`",'',f"**Goal:** {f['goal']}",'',f"**Data source(s):** {', '.join('`'+s+'`' for s in f['source'])}.",'',f"**How to read / calculation:** {f['method']}",'']
    lines+=['## Methodology and limitations','',
            '- Both evaluators are run sequentially with identical split, requested image count, confidence, NMS IoU, seed, device selection and warmup arguments. They retain their own existing CLI and output schemas.',
            '- Exact sample IDs must match across the two runs. A mismatch is an error rather than an invitation to silently compare different subsets.',
            '- Latency confidence intervals resample **paired images**, preserving correspondence. They quantify sample variability conditional on this device/run, not run-to-run GPU variability.',
            '- Inference timing may be affected by thermal state, GPU contention, caching and evaluator-specific instrumentation; sequential execution is not randomized.',
            '- The clean/degraded subset counts and per-subset detection metrics are descriptive; sparse categories should not be interpreted as robust effects.',
            '- GPU peak memory is measured at run level; a single observation per pipeline cannot support a statistical significance claim.',
            '- Model parameters and memory are not interchangeable. Memory includes loaded models and temporary allocations.',
            '- Gate activity is observational; no causal interpretation is warranted.',
            '', '## Inputs','',f"- CrossFuse: `{a['path']}`",f"- PACE: `{b['path']}`",'']
    (output/'comparison_report.md').write_text('\n'.join(lines))
    (output/'visualization_guide.md').write_text('# Visualization guide\n\n'+'\n'.join([f"## `{f['file']}`\n\n**Goal:** {f['goal']}\n\n**Source(s):** {', '.join(f['source'])}\n\n**Method:** {f['method']}\n" for f in figures]))


def figure_gallery(a,b,ids,out,max_images):
    if max_images == 0:
        return None, 0
    gallery=out/'visualization_pairs';gallery.mkdir(parents=True,exist_ok=True)
    chosen=[]
    for sid in ids:
        pa=a['path']/'visualizations'/f'{sid}.png';pb=b['path']/'visualizations'/f'{sid}.png'
        if not (pa.is_file() and pb.is_file()):continue
        with Image.open(pa) as src:im_a=src.convert('RGB')
        with Image.open(pb) as src:im_b=src.convert('RGB')
        target_height=340
        def resize(im):return im.resize((max(1,round(im.width*target_height/im.height)),target_height))
        im_a,im_b=resize(im_a),resize(im_b)
        canvas=Image.new('RGB',(im_a.width+im_b.width+24,im_a.height+45),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((8,10),f'CrossFuse — {sid}',fill='black')
        draw.text((im_a.width+20,10),f'PACE — {sid}',fill='black')
        canvas.paste(im_a,(0,45));canvas.paste(im_b,(im_a.width+24,45))
        path=gallery/f'{sid}_comparison.png';canvas.save(path);chosen.append(path)
        if max_images and len(chosen)>=max_images:break
    if not chosen:return None,0
    return {'file':'visualization_pairs/', 'source':['baseline/visualizations/<sample_id>.png','pace/visualizations/<sample_id>.png'],
            'goal':'Inspect differences in fused imagery and detection overlays for the same images.',
            'method':'Filename-matched qualitative side-by-side panels; not quantitative evidence of improvement.'},len(chosen)

def main(argv=None):
    a=args_parse(argv)
    setup_style()
    if a.crossfuse is not None:
        pa,pb=a.crossfuse.resolve(),a.pace.resolve()
    else:
        pa,pb=run_evaluations(a)
    run_a,run_b=get_run(pa),get_run(pb)
    ids=common_ids(run_a,run_b)
    ca=run_a['summary'].get('configuration',{});cb=run_b['summary'].get('configuration',{})
    for key in ('split','image_size','confidence','nms_iou','seed'):
        if key in ca and key in cb and str(ca[key])!=str(cb[key]):
            raise ValueError(f'Incompatible evaluation configuration for {key}: {ca[key]} vs {cb[key]}')
    suffix='all' if a.num_images==0 else str(a.num_images)
    output=(a.output or ROOT/'evaluation'/'comparisons'/f'{a.split}_{suffix}').resolve()
    figures_dir=output/'figures';figures_dir.mkdir(parents=True,exist_ok=True)
    figures=[];stats={'matched_sample_ids':ids,'n_samples':len(ids),'bootstrap_replicates':a.bootstrap,'bootstrap_seed':a.seed}
    for f in [figure_detection(run_a,run_b,figures_dir,a.dpi),figure_subsets(run_a,run_b,figures_dir,a.dpi)]:
        if f:figures.append(f)
    f,lat=figure_latency(run_a,run_b,ids,figures_dir,a.dpi,a.seed,a.bootstrap)
    if f:figures.append(f);stats['paired_latency_mean_difference_ms']=lat
    f=figure_stages(run_a,run_b,figures_dir,a.dpi)
    if f:figures.append(f)
    f,mem=figure_memory(run_a,run_b,figures_dir,a.dpi)
    if f:figures.append(f);stats['memory']=mem
    f=figure_gate(run_a,run_b,ids,figures_dir,a.dpi)
    if f:figures.append(f)
    f=figure_tradeoff(run_a,run_b,figures_dir,a.dpi)
    if f:figures.append(f)
    f,count=figure_gallery(run_a,run_b,ids,output,a.visualizations)
    if f:figures.append(f)
    stats['matched_visualization_pairs']=count
    stats['detection']={'crossfuse':run_a['summary'].get('detection',{}),'pace':run_b['summary'].get('detection',{})}
    stats['subsets']={'crossfuse':run_a['summary'].get('subsets',{}),'pace':run_b['summary'].get('subsets',{})}
    stats['source_directories']={'crossfuse':str(pa),'pace':str(pb)}
    stats['figures']=figures
    (output/'statistics.json').write_text(json.dumps(stats,indent=2,allow_nan=False))
    # Keep legacy machine-readable comparison artifact name.
    (output/'comparison.json').write_text(json.dumps(stats,indent=2,allow_nan=False))
    report_markdown(run_a,run_b,ids,figures,stats,output)
    print(f'Compared {len(ids)} matched samples; wrote {len(figures)} figure entries to {output}')
    print(f'Report: {output/"comparison_report.md"}')

if __name__=='__main__':
    main()
