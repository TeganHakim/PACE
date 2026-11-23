from __future__ import annotations
import runpy
import sys
from common import ROOT

def main():
    source=ROOT/'baseline/CrossFuse/evaluate_color.py'
    sys.path.insert(0,str(source.parent))
    sys.argv=[str(source),*sys.argv[1:],'--save-fused']
    runpy.run_path(str(source),run_name='__main__')

if __name__=='__main__':
    main()
