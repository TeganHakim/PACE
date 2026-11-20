"""Read-only CUDA/Jetson readiness report; safe to run on macOS too."""
import argparse
import json
import platform
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS={
 "crossfuse":ROOT/"baseline/CrossFuse/models/transfuse/fusetrans_epoch_28.model",
 "quality_rgb":ROOT/"prototyping/quality_estimator/checkpoints/quality_rgb.pt",
 "quality_ir":ROOT/"prototyping/quality_estimator/checkpoints/quality_ir.pt",
 "reliability_gate":ROOT/"prototyping/reliability_gate/checkpoints/reliability_gate_best.pt",
}
def inspect(require_cuda=False):
    import torch
    info={
        "platform":platform.platform(),
        "machine":platform.machine(),
        "python":sys.version.split()[0],
        "torch":torch.__version__,
        "cuda_build":torch.version.cuda,
        "cuda_available":torch.cuda.is_available(),
        "mps_available":bool(getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()),
        "checkpoints":{k:p.is_file() for k,p in CHECKPOINTS.items()},
    }
    if info["cuda_available"]:
        info["cuda_device"]=torch.cuda.get_device_name(0)
        info["cuda_capability"]=list(torch.cuda.get_device_capability(0))
        try:
            x=torch.ones((2,2),device="cuda")
            info["cuda_tensor_smoke"]=bool((x@x).sum().item()==8)
        except Exception as e:
            info["cuda_tensor_smoke"]=False
            info["cuda_tensor_error"]=str(e)
    errors=[]
    if require_cuda and not info["cuda_available"]:errors.append("CUDA unavailable")
    if require_cuda and not info.get("cuda_tensor_smoke"):errors.append("CUDA tensor smoke failed")
    if not all(info["checkpoints"].values()):errors.append("Required checkpoint(s) missing")
    info["errors"]=errors
    return info
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda",action="store_true",help="Fail unless CUDA tensor smoke passes")
    parser.add_argument("--output",type=Path,help="Optional JSON report destination")
    args=parser.parse_args()
    report=inspect(args.require_cuda)
    print(json.dumps(report,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    if report["errors"]:raise SystemExit(1)
if __name__=="__main__":main()
