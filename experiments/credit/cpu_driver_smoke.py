"""Run the existing deployment-driver diagnostic on CPU, through the gym only."""
import os,runpy,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.argv=['experiments/r1_cerl_gpu.py',*sys.argv[1:]]
runpy.run_path('experiments/r1_cerl_gpu.py',run_name='__main__')
