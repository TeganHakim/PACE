"""Run all feasible Mac acceptance checks, stopping on first failure.

E2E evaluation is opt-in and requires the datasets/checkpoints.
"""
import argparse,subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(cmd):
 print("+", " ".join(map(str,cmd)),flush=True)
 return subprocess.call(list(map(str,cmd)),cwd=ROOT)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--e2e",action="store_true",help="Run actual three-method E2E evaluation")
 p.add_argument("--full",action="store_true",help="Run full dataset; implies --e2e")
 p.add_argument("--device",choices=["mps","cpu","cuda","auto"],default="mps")
 p.add_argument("--reference",type=Path,help="Existing 3-sample reference run for exact comparison")
 a=p.parse_args()
 cmds=[
  [sys.executable,"-m","pytest","-q","tests/"],
  [sys.executable,"tools/phase8_scientific_audit.py","--strict"],
  [sys.executable,"tools/phase8_cleanup_audit.py"],
  [sys.executable,"tools/verify_research_artifacts.py","--verify","reports/refactor/phase6_artifact_manifest.json","--strict"],
  [sys.executable,"evaluation/jetson_preflight.py","--output","reports/refactor/phase8_mac_preflight.json"]
 ]
 if a.full:a.e2e=True
 if a.e2e:
  out=Path("evaluation/results/phase8_mac_full" if a.full else "evaluation/results/phase8_mac_smoke")
  cmds.append([sys.executable,"evaluation/eval_metrics.py",
   "--dataset-root","processed_dataset_realistic_degraded_v2",
   "--original-root","VisDrone-DroneVehicle","--max-samples","0" if a.full else "3",
   "--device",a.device,"--visualizations","0","--warmup","2","--seed","42","--output",out])
  if a.reference:
   cmds.append([sys.executable,"evaluation/verify_regression.py","--reference",a.reference,"--candidate",out])
 for cmd in cmds:
  if run(cmd):print("FAIL:",cmd);return 1
 print("PASS: requested Mac checks complete")
 if not a.e2e:print("NOTE: E2E not run. Rerun with --e2e and --reference PATH.")
 if not a.full:print("NOTE: full-dataset evaluation not run. Use --full when feasible.")
 return 0
if __name__=="__main__":raise SystemExit(main())
