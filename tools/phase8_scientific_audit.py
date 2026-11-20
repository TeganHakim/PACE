"""Read-only scientific evidence audit; does not claim to reproduce full inference."""
import argparse,csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT="evaluation/comparisons/both_baselines_vs_pace_test_set/comparison_report.md"
METHODS=("CrossFuse","PACE","Addition")
def audit(root):
 p=root/REPORT
 out={"report":REPORT,"exists":p.is_file(),"sample_count":None,"metrics":{},"caveats":[]}
 if not p.is_file():out["caveats"].append("Full-test comparison report missing");return out
 s=p.read_text()
 m=re.search(r"Compared\s+\*\*(\d+)\s+matched samples\*\*",s)
 if m:out["sample_count"]=int(m.group(1))
 else:out["caveats"].append("Cannot parse matched sample count")
 for line in s.splitlines():
  if not line.startswith("|") or "---" in line:continue
  cols=[x.strip() for x in line.strip().strip("|").split("|")]
  if len(cols)==4 and cols[0] in ("Precision","Recall","F1","mAP@50","mAP@50:95"):
   try:out["metrics"][cols[0]]=dict(zip(METHODS,map(float,cols[1:])))
   except ValueError:out["caveats"].append("Unparseable metric: "+cols[0])
 out["expected_metrics_present"]=len(out["metrics"])==5
 out["pace_beats_addition_on_full_test_map50"]=(
   out["metrics"].get("mAP@50",{}).get("PACE",float("-inf")) >
   out["metrics"].get("mAP@50",{}).get("Addition",float("inf")))
 out["crossfuse_beats_pace_on_full_test_map50"]=(
   out["metrics"].get("mAP@50",{}).get("CrossFuse",float("-inf")) >
   out["metrics"].get("mAP@50",{}).get("PACE",float("inf")))
 out["condition_analysis_script_exists"]=(root/"evaluation/condition_analysis.py").is_file()
 out["condition_specific_rgb_noise_reproduced"]=False
 out["caveats"].append("Reported full-test metrics are prior results, not independently rerun by this audit")
 out["caveats"].append("RGB sensor-noise subgroup claim needs its original CSV/figure or fresh condition analysis")
 return out
def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument("--root",type=Path,default=ROOT)
 a.add_argument("--output",type=Path,default=Path("reports/refactor/phase8_scientific_audit.json"))
 a.add_argument("--strict",action="store_true")
 x=a.parse_args()
 r=audit(x.root)
 dst=x.output if x.output.is_absolute() else x.root/x.output
 dst.parent.mkdir(parents=True,exist_ok=True)
 dst.write_text(json.dumps(r,indent=2)+"\n")
 print(json.dumps(r,indent=2))
 if x.strict and (not r["exists"] or not r["expected_metrics_present"] or r["sample_count"]!=8980):return 1
 return 0
if __name__=="__main__":raise SystemExit(main())
