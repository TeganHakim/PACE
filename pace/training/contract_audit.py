"""Read-only Phase 4 training contract audit. Run at repository root."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS={
    "quality_estimator":ROOT/"prototyping/quality_estimator/train.py",
    "reliability_gate":ROOT/"prototyping/reliability_gate/train.py",
}
CHECKPOINTS={
    "quality_rgb":ROOT/"prototyping/quality_estimator/checkpoints/quality_rgb.pt",
    "quality_ir":ROOT/"prototyping/quality_estimator/checkpoints/quality_ir.pt",
    "reliability_gate":ROOT/"prototyping/reliability_gate/checkpoints/reliability_gate_best.pt",
}
REQUIRED={
    "quality_estimator":{"model_state_dict","modality","in_channels","epoch","val_mse","val_mae","val_degraded_mae","loss_weight_strength","degraded_sample_fraction","extra_feature_scale","learning_rate"},
    "reliability_gate":{"model_state_dict","hidden_dim","epoch","val_bce","val_accuracy","positive_fraction","training_method","fused_image_scorer","seed"},
}
def save_dict_keys(tree):
    found=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):
            continue
        if node.func.attr!="save" or not isinstance(node.func.value,ast.Name) or node.func.value.id!="torch" or not node.args:
            continue
        value=node.args[0]
        if isinstance(value,ast.Dict):
            keys={x.value for x in value.keys if isinstance(x,ast.Constant) and isinstance(x.value,str)}
            found.append(keys)
    return found
def audit_scripts():
    result={}
    for name,path in SCRIPTS.items():
        source=path.read_bytes()
        tree=ast.parse(source,filename=str(path))
        saved=save_dict_keys(tree)
        matches=[keys for keys in saved if REQUIRED[name].issubset(keys)]
        if not matches:
            raise AssertionError(f"{name}: expected checkpoint fields missing from torch.save()")
        result[name]={"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(source).hexdigest(),"checkpoint_fields":sorted(matches[0])}
    return result
def audit_checkpoints():
    import torch
    result={}
    for name,path in CHECKPOINTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {path}")
        data=torch.load(path,map_location="cpu",weights_only=True)
        if not isinstance(data,dict) or "model_state_dict" not in data:
            raise AssertionError(f"{name}: missing model_state_dict")
        expected=REQUIRED["reliability_gate" if name=="reliability_gate" else "quality_estimator"]
        # Historical checkpoints may omit training-only metrics: report, don't reject.
        result[name]={
            "path":str(path.relative_to(ROOT)),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_fields":sorted(data.keys()),
            "missing_training_metadata":sorted(expected-set(data)),
            "state_dict_tensors":len(data["model_state_dict"]),
        }
    return result
def main():
    scripts=audit_scripts()
    checkpoints=audit_checkpoints()
    out=ROOT/"reports/refactor/phase4_training_contract.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"scripts":scripts,"pretrained_checkpoints":checkpoints},indent=2)+"\n")
    for name in scripts: print("PASS training source:",name)
    for name in checkpoints: print("PASS checkpoint:",name)
    print("Saved:",out)
if __name__=="__main__":
    main()
