"""Phase 5 regression guard: compare complete evaluation summaries and IDs.

No model inference, no modification of evaluation outputs.
Run: python evaluation/verify_regression.py --reference ... --candidate ...
"""
import argparse
import csv
import json
import math
from pathlib import Path

METHODS=("pace","crossfuse","addition")
SPLITS=("degraded","clean")
def read_json(p):
    if not p.is_file(): raise FileNotFoundError(p)
    return json.loads(p.read_text())
def compare_values(expected,actual,path="",atol=1e-10):
    errors=[]
    if isinstance(expected,dict):
        if not isinstance(actual,dict):
            return [f"{path}: type mismatch"]
        for k in expected.keys() & actual.keys():
            errors.extend(compare_values(expected[k],actual[k],f"{path}.{k}",atol))
        # Summaries can contain run-dependent metadata; compare common keys.
    elif isinstance(expected,list) and isinstance(actual,list):
        if len(expected)!=len(actual): errors.append(f"{path}: length {len(expected)} != {len(actual)}")
        for i,(a,b) in enumerate(zip(expected,actual)):
            errors.extend(compare_values(a,b,f"{path}[{i}]",atol))
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool) and isinstance(actual,(int,float)) and not isinstance(actual,bool):
        if not math.isclose(expected,actual,rel_tol=0,abs_tol=atol):
            errors.append(f"{path}: {expected} != {actual}")
    elif expected!=actual:
        errors.append(f"{path}: {expected!r} != {actual!r}")
    return errors
def ids(path):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    result=[x["sample_id"] for x in rows]
    if len(result)!=len(set(result)): raise ValueError(f"Duplicate sample IDs in {path}")
    return result
def verify(reference,candidate,atol=1e-10):
    failures=[]
    checked=[]
    for method in METHODS:
        for split in SPLITS:
            a=reference/method/split
            b=candidate/method/split
            if not a.is_dir() and not b.is_dir(): continue
            for filename in ("summary.json","latency.json","per_sample.csv","predictions.json"):
                if not (a/filename).is_file() or not (b/filename).is_file():
                    failures.append(f"{method}/{split}: missing {filename}")
            if not (a/"summary.json").is_file() or not (b/"summary.json").is_file():continue

            reference_summary = read_json(a / "summary.json")
            candidate_summary = read_json(b / "summary.json")

            reference_summary.pop("latency", None)
            candidate_summary.pop("latency", None)

            failures.extend(
                compare_values(
                    reference_summary,
                    candidate_summary,
                    f"{method}/{split}/summary",
                    atol,
                )
            )

            if (a/"per_sample.csv").is_file() and (b/"per_sample.csv").is_file():
                if ids(a/"per_sample.csv")!=ids(b/"per_sample.csv"):
                    failures.append(f"{method}/{split}: sample IDs/order differ")
            checked.append(f"{method}/{split}")
    if not checked: failures.append("No method/split results found")
    return checked,failures
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--atol",type=float,default=1e-10)
    args=p.parse_args()
    checked,failures=verify(args.reference,args.candidate,args.atol)
    for x in checked:print("CHECK",x)
    if failures:
        for e in failures:print("FAIL",e)
        raise SystemExit(1)
    print(f"PASS: {len(checked)} method/split evaluations match (numeric tolerance {args.atol})")
if __name__=="__main__":main()
