import os
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()))
os.environ['HF_HUB_OFFLINE']='1'
import json,time,base64,hashlib,shutil,importlib.metadata
from collections import Counter
import numpy as np,cv2
from PIL import Image
from vllm import LLM,SamplingParams
from conservative_review_policy_v1 import decide
from review_langsam_candidates_v4 import PROMPT,NONWIRE
ROOT=Path('results/dev_transfer_20261107_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (ROOT/'phase1_complete.json').exists()
 frozen=json.loads((ROOT/'frozen_config.json').read_text())
 assert frozen['source_hashes']=={n:sha(Path(n)) for n in frozen['source_hashes']}
 config=json.loads(Path('results/vlm_review/BEGO7553_frame_001098_review_v4/run_config.json').read_text())
 assert config['prompt']==PROMPT
 settings=config['settings'];sample=SamplingParams(**config['sampling'])
 (ROOT/'review_frozen_config.json').write_text(json.dumps({'source_config':config,'shared_model_load_for_two_images':True,'empty_control_source':'results/vlm_review/BEGO7553_frame_001098_review_v4/empty_target_control.jpg','policy_function_source_sha256':sha(Path('conservative_review_policy_v1.py')),'no_annotations_loaded':True},indent=2))
 start=time.perf_counter();llm=LLM(**settings);load=time.perf_counter()-start;pad=llm.get_tokenizer().convert_tokens_to_ids('<|image_pad|>')
 def ask(path):
  uri='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode();t=time.perf_counter()
  r=llm.chat([{'role':'user','content':[{'type':'image_url','image_url':{'url':uri}},{'type':'text','text':PROMPT}]}],sampling_params=sample,use_tqdm=False)[0]
  raw=r.outputs[0].text;sec=time.perf_counter()-t;n=r.prompt_token_ids.count(pad);assert n>0
  parsed=None;error=None
  try:
   parsed=json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
   assert parsed['category'] in NONWIRE|{'wire_or_cable','unclear'}
   assert parsed['visibility'] in {'clear','unclear'} and isinstance(parsed['evidence'],str) and parsed['evidence'].strip()
   assert r.outputs[0].finish_reason=='stop'
  except Exception as e:error=repr(e);parsed=None
  return {'raw':raw,'parsed':parsed,'parse_error':error,'seconds':sec,'image_tokens':n,'input':str(path),'input_sha256':sha(path),'finish_reason':r.outputs[0].finish_reason}
 for name in frozen['images']:
  src=ROOT/name/'candidates';out=ROOT/name/'vlm';out.mkdir(exist_ok=False)
  data=json.loads((src/'candidates.json').read_text());before={c['id']:sha(src/c['mask']) for c in data['candidates']}
  shutil.copy2('results/vlm_review/BEGO7553_frame_001098_review_v4/empty_target_control.jpg',out/'empty_target_control.jpg')
  control=ask(out/'empty_target_control.jpg');passed=bool(control['parsed'] and control['parsed']['category']=='unclear')
  (out/'empty_target_control.json').write_text(json.dumps(control,indent=2));print('CONTROL',name,passed,flush=True)
  results=[];policies=[]
  for c in data['candidates']:
   im=Image.open(src/c['original_context']).convert('RGB');maskim=Image.open(src/c['id']/'mask_context.png').convert('L');assert im.size==maskim.size
   mask=np.asarray(maskim)>0;canvas=Image.composite(im,Image.new('RGB',im.size,(150,150,150)),maskim);path=out/(c['id']+'_isolated.jpg');canvas.save(path,quality=95)
   dist=cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1];proxy=float(2*np.percentile(dist[mask],95)) if mask.any() else 0.;broad=bool(proxy>24 and int(mask.sum())>4096)
   r=ask(path);p=r['parsed']
   if p is None or p['visibility']=='unclear' or p['category']=='unclear':decision='uncertain'
   elif p['category'] in NONWIRE:decision='reject'
   elif broad:decision='uncertain'
   else:decision='keep'
   proposal=decision
   if not passed:decision='uncertain'
   r.update(candidate_id=c['id'],decision=decision,decision_proposal=proposal,control_gate_pass=passed,broad_mask_flag=broad,thickness_proxy_p95_original_pixels=proxy,mask_area_original_pixels=int(mask.sum()))
   conservative,action,reason=decide(r,passed);assert decide(r,False)[0]=='uncertain'
   policies.append({'candidate_id':c['id'],'model_v4_decision':decision,'model_category':p['category'] if p else None,'decision':conservative,'next_action':action,'policy_reason':reason,'broad_mask_flag':broad,'finalized':False})
   results.append(r);(out/(c['id']+'.json')).write_text(json.dumps(r,indent=2));print(name,c['id'],'RAW',decision,'CONSERVATIVE',conservative,p,flush=True)
  assert before=={c['id']:sha(src/c['mask']) for c in data['candidates']}
  summary={'status':'REVIEW_COMPLETE','candidate_count':len(results),'counts':dict(Counter(r['decision'] for r in results)),'empty_control_pass':passed,'source_masks_unchanged':True,'ground_truth_used':False,'model_load_seconds_shared':load,'inference_seconds_total':sum(r['seconds'] for r in results),'results':results}
  (out/'review.json').write_text(json.dumps(summary,indent=2))
  (out/'conservative_policy.json').write_text(json.dumps({'status':'CONSERVATIVE_POLICY_APPLIED','empty_control_pass':passed,'counts':dict(Counter(r['decision'] for r in policies)),'results':policies,'ground_truth_used':False,'source_masks_modified':False},indent=2))
  print('REVIEW_DONE',name,summary['counts'],'CONSERVATIVE',dict(Counter(r['decision'] for r in policies)),flush=True)
 (ROOT/'phase2_complete.json').write_text(json.dumps({'status':'VLM_REVIEW_COMPLETE','images':frozen['images'],'no_annotations_loaded':True},indent=2));print('PHASE2_COMPLETE',flush=True)
if __name__=='__main__':main()
