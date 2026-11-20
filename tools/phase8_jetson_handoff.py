"""Partner handoff: Jetson CUDA preflight + tests + E2E; no source edits."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--full",action="store_true",help="8980-sample evaluation instead of 3-sample smoke")
 p.add_argument("--skip-tests",action="store_true")
 a=p.parse_args()
 cmds=[
  [sys.executable,"evaluation/jetson_preflight.py","--require-cuda","--output","reports/refactor/phase8_jetson_preflight.json"],
 ]
 if not a.skip_tests:cmds.append([sys.executable,"-m","pytest","-q","tests/"])
 cmds.append([sys.executable,"evaluation/eval_metrics.py",
  "--dataset-root","processed_dataset_realistic_degraded_v2",
  "--original-root","VisDrone-DroneVehicle",
  "--max-samples","0" if a.full else "3","--device","cuda",
  "--visualizations","0","--warmup","2","--seed","42",
  "--output","evaluation/results/phase8_jetson_full" if a.full else "evaluation/results/phase8_jetson_smoke"])
 for cmd in cmds:
  print("+"," ".join(map(str,cmd)),flush=True)
  if subprocess.call(list(map(str,cmd)),cwd=ROOT):
   print("FAILED; send full log to project owner")
   return 1
 print("PASS: Jetson requested checks complete")
 return 0
if __name__=="__main__":raise SystemExit(main())
