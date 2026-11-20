"""Conservative read-only repo cleanup inventory. Never deletes or moves files."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def inventory(root):
 r=subprocess.run(["git","status","--porcelain","--untracked-files=normal"],cwd=root,capture_output=True,text=True,check=True)
 rows=[]
 for line in r.stdout.splitlines():
  if len(line)<4:continue
  status,path=line[:2],line[3:]
  # Git porcelain - untracked paths may be quoted; retain exact raw spelling
  kind="review_only"
  if path.endswith(".bak") or path.endswith(".zip") or path.startswith("README-phase"):
   kind="likely_generated_review"
  if any(k in path.lower() for k in ("checkpoint","dataset","result","comparison","report")):
   kind="research_preserve"
  rows.append({"status":status,"path":path,"classification":kind})
 return {"working_tree":rows,"nothing_deleted":True,"note":"Review classifications manually; no automatic deletion is performed."}
def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument("--root",type=Path,default=ROOT)
 a.add_argument("--output",type=Path,default=Path("reports/refactor/phase8_cleanup_inventory.json"))
 x=a.parse_args()
 data=inventory(x.root)
 dst=x.output if x.output.is_absolute() else x.root/x.output
 dst.parent.mkdir(parents=True,exist_ok=True)
 dst.write_text(json.dumps(data,indent=2)+"\n")
 print("Read-only cleanup inventory:",dst,"entries:",len(data["working_tree"]))
 for row in data["working_tree"][:40]:print(row["status"],row["classification"],row["path"])
 return 0
if __name__=="__main__":raise SystemExit(main())
