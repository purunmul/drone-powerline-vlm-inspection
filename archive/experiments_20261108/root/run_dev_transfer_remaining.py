import os,time,subprocess,json
from pathlib import Path
exitfile=Path('logs/dev_transfer_phase1.exit')
while not exitfile.exists():time.sleep(5)
assert exitfile.read_text().strip()=='0',exitfile.read_text()
env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['PYTHONNOUSERSITE']='1';env['HF_HOME']=str(Path('.cache/huggingface').resolve());env['HF_HUB_OFFLINE']='1'
for label,python,script in [('phase2','.venv-vlm/bin/python','run_dev_transfer_phase2.py'),('phase3','.venv-eval/bin/python','run_dev_transfer_phase3.py')]:
 print('START',label,flush=True)
 with Path('logs/dev_transfer_'+label+'.log').open('w') as log:r=subprocess.run([python,'-u',script],env=env,stdout=log,stderr=subprocess.STDOUT)
 Path('logs/dev_transfer_'+label+'.exit').write_text(str(r.returncode))
 assert r.returncode==0,(label,r.returncode)
 print('DONE',label,flush=True)
print('ALL_DONE',flush=True)
