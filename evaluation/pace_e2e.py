from __future__ import annotations
import runpy
import sys
from common import ROOT

def main():
    source=ROOT/'pace/evaluate_color.py'
    sys.path.insert(0,str(source.parent))
    sys.argv=[str(source),*sys.argv[1:],'--paired-jpg-samples','--save-fused']
    runpy.run_path(str(source),run_name='__main__')

if __name__=='__main__':
    main()
