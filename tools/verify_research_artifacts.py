"""Read-only streaming SHA256 manifest for known PACE research artifacts."""
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={
 "checkpoints":{
 "crossfuse":"baseline/CrossFuse/models/transfuse/fusetrans_epoch_28.model",
 "quality_rgb":"prototyping/quality_estimator/checkpoints/quality_rgb.pt",
 "quality_ir":"prototyping/quality_estimator/checkpoints/quality_ir.pt",
 "gate":"prototyping/reliability_gate/checkpoints/reliability_gate_best.pt"},
 "sources":{
 "crossfuse_autoencoder":"baseline/CrossFuse/train_autoencoder.py",
 "crossfuse_transformer":"baseline/CrossFuse/train_conv_trans.py",
 "quality_train":"prototyping/quality_estimator/train.py",
 "gate_train":"prototyping/reliability_gate/train.py",
 "evaluation":"evaluation/eval_metrics.py",
 "condition_analysis":"evaluation/condition_analysis.py"},
 "reports":{"full_test":"evaluation/comparisons/both_baselines_vs_pace_test_set/comparison_report.md",
 "training_contract":"reports/refactor/phase4_training_contract.json"}
}
def sha(path,chunk=1024*1024):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(chunk),b""):h.update(block)
    return h.hexdigest()
def collect(root,files=FILES):
    result={}
    for group,items in files.items():
        result[group]={}
        for name,rel in items.items():
            p=root/rel
            result[group][name]={"path":rel,"exists":p.is_file()}
            if p.is_file():
                result[group][name].update(size_bytes=p.stat().st_size,sha256=sha(p))
    return {"schema_version":1,"artifacts":result}
def differences(old,new):
    issues=[]
    for group,items in old["artifacts"].items():
        for name,expected in items.items():
            current=new["artifacts"].get(group,{}).get(name)
            if current is None or expected.get("exists")!=current.get("exists") or expected.get("sha256")!=current.get("sha256"):
                issues.append(group+"/"+name)
    return issues
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output",default="reports/refactor/phase6_artifact_manifest.json")
    p.add_argument("--verify",type=Path)
    p.add_argument("--strict",action="store_true")
    a=p.parse_args(argv)
    now=collect(a.root)
    missing=[f"{g}/{n}" for g in ("checkpoints","sources") for n,v in now["artifacts"][g].items() if not v["exists"]]
    if a.verify:
        issues=differences(json.loads(a.verify.read_text()),now)
        for x in issues:print("CHANGED:",x)
        if issues:return 1
        print("PASS: artifact manifest matches")
    else:
        out=a.root/a.output
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(now,indent=2,sort_keys=True)+"\n")
        print("Saved:",out)
    if missing:print("Missing required:",", ".join(missing))
    return int(a.strict and bool(missing))
if __name__=="__main__":raise SystemExit(main())
