import json,hashlib,subprocess,shutil,time,os
from pathlib import Path
ROOT=Path('results/dev_transfer_20261107_v1')
NAMES=['PUYL5573_frame_000165','KZIV5445_frame_000488']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ROOT.mkdir(parents=True,exist_ok=False); (ROOT/'frozen_sources').mkdir()
 sources=['run_langsam_baseline.py','run_langsam_tiled.py','prepare_langsam_candidates.py','review_langsam_candidates_v4.py','conservative_review_policy_v1.py','evaluate_first_image_v1.py','run_langsam.sh']
 for name in sources:shutil.copy2(name,ROOT/'frozen_sources'/name)
 config={'images':NAMES,'selection':'positions 2 and 3 of initial_8 fixed development list; different source videos and folders; no difficulty selection','data_split':'development_only','heldout_160_used':False,'prompt':'power line.','box_threshold':0.30,'text_threshold':0.25,'tile_size':1024,'minimum_overlap':256,'duplicate_iou':0.85,'review':'same foreground-only v4 prompt and v4_1 policy','tolerances_pixels':[3,5,10],'human_assisted_repair_in_this_run':False,'annotation_access':'evaluation stage only; never passed to detector/VLM or used for candidate selection','source_hashes':{n:sha(Path(n)) for n in sources},'image_hashes':{n:sha(Path('data/initial_8/images')/(n+'.jpg')) for n in NAMES}}
 (ROOT/'frozen_config.json').write_text(json.dumps(config,indent=2))
 env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['PYTHONNOUSERSITE']='1';env['HF_HUB_OFFLINE']='1'
 for name in NAMES:
  folder=ROOT/name;folder.mkdir();image=Path('data/initial_8/images')/(name+'.jpg')
  commands=[['bash','run_langsam.sh','run_langsam_baseline.py','--image',str(image),'--output',str(folder/'whole'),'--prompt','power line.','--box-threshold','0.30','--text-threshold','0.25'],['bash','run_langsam.sh','run_langsam_tiled.py','--baseline',str(folder/'whole/result.json'),'--output',str(folder/'tiles'),'--tile-size','1024','--overlap','256']]
  for label,cmd in zip(['whole','tiles'],commands):
   print('START',name,label,flush=True)
   with (folder/(label+'.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   print('DONE',name,label,flush=True)
  source=Path('prepare_langsam_candidates.py').read_text().replace("src=Path('results/langsam_baseline/BEGO7553_frame_001098_tiles1024_v1')",'src=Path('+repr(str(folder/'tiles'))+')').replace("out=Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1')",'out=Path('+repr(str(folder/'candidates'))+')')
  script=folder/'prepare_candidates.py';script.write_text(source)
  with (folder/'prepare.log').open('w') as log:subprocess.run(['.venv-langsam/bin/python',str(script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  d=json.loads((folder/'candidates/candidates.json').read_text());print('CANDIDATES_READY',name,d['raw_count'],d['organized_count'],flush=True)
 assert config['source_hashes']=={n:sha(Path(n)) for n in sources}
 (ROOT/'phase1_complete.json').write_text(json.dumps({'status':'DETECTION_AND_ORGANIZATION_COMPLETE','images':NAMES,'settings_unchanged':True},indent=2))
 print('PHASE1_COMPLETE',flush=True)
if __name__=='__main__':main()
