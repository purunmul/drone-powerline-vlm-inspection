import os
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()))
import json,time,base64,hashlib,zipfile,importlib.metadata
from collections import Counter
import numpy as np, cv2
from PIL import Image,ImageDraw
from vllm import LLM,SamplingParams
SRC=Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1')
OUT=Path('results/vlm_review/BEGO7553_frame_001098_review_v4')
PROMPT="This image contains only selected foreground pixels in their original colors on a uniform gray background. Ignore the gray background. Identify the visible foreground object. If no foreground is visible, or if its identity cannot be resolved, use unclear. Return one JSON object with fields category, visibility, evidence. category must be one of vehicle, road_or_ground, wire_or_cable, pole, building_edge, vegetation, other, unclear. visibility must be clear or unclear. evidence must be a short factual description of visible foreground pixels. Do not infer objects outside the selected foreground."
NONWIRE={'vehicle','road_or_ground','pole','building_edge','vegetation','other'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    d=json.loads((SRC/'candidates.json').read_text()); before={c['id']:sha(SRC/c['mask']) for c in d['candidates']}
    settings=dict(model='Qwen/Qwen2.5-VL-3B-Instruct',revision='66285546d2b821cf421d4f5eb2576359d3770cd3',max_model_len=4096,gpu_memory_utilization=0.80,limit_mm_per_prompt={'image':1,'video':0},max_num_seqs=1,max_num_batched_tokens=4096,enforce_eager=True,seed=0,mm_processor_kwargs={'min_pixels':784,'max_pixels':802816})
    config={'settings':settings,'prompt':PROMPT,'sampling':{'temperature':0,'max_tokens':160},'ground_truth_used':False,'versions':{k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','Pillow','numpy']},'broad_mask_rule':'2 * 95th percentile of pixel distance to mask exterior >24 original-image pixels AND mask area >4096; initial engineering flag only, not validated accuracy','changes_from_v3':['foreground only; context panel removed','neutral prompt describing selected foreground','empty-target control gates all automatic decisions','same original masks and model revision'],'confounded_changes':True}
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2))
    start=time.perf_counter(); llm=LLM(**settings); load=time.perf_counter()-start
    sample=SamplingParams(temperature=0,max_tokens=160); pad=llm.get_tokenizer().convert_tokens_to_ids('<|image_pad|>')
    def board(c,empty=False):
        im=Image.open(SRC/c['original_context']).convert('RGB'); mask=Image.open(SRC/c['id']/'mask_context.png').convert('L')
        assert im.size==mask.size
        gray=Image.new('RGB',im.size,(150,150,150)); fg=gray if empty else Image.composite(im,gray,mask)
        return fg,np.asarray(mask)>0
    def ask(path):
        uri='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()
        t=time.perf_counter(); r=llm.chat([{'role':'user','content':[{'type':'image_url','image_url':{'url':uri}},{'type':'text','text':PROMPT}]}],sampling_params=sample,use_tqdm=False)[0]
        raw=r.outputs[0].text; sec=time.perf_counter()-t; n=r.prompt_token_ids.count(pad); assert n>0
        parsed=None; error=None
        try:
            parsed=json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
            assert parsed['category'] in NONWIRE|{'wire_or_cable','unclear'}
            assert parsed['visibility'] in {'clear','unclear'} and isinstance(parsed['evidence'],str) and parsed['evidence'].strip()
            assert r.outputs[0].finish_reason=='stop'
        except Exception as e: error=repr(e); parsed=None
        return {'raw':raw,'parsed':parsed,'parse_error':error,'seconds':sec,'image_tokens':n,'input':str(path),'input_sha256':sha(path),'finish_reason':r.outputs[0].finish_reason}
    c13=next(c for c in d['candidates'] if c['id']=='P13'); empty,_=board(c13,True); empty.save(OUT/'empty_target_control.jpg',quality=95)
    control=ask(OUT/'empty_target_control.jpg'); (OUT/'empty_target_control.json').write_text(json.dumps(control,indent=2)); print('EMPTY_CONTROL',control['parsed'] or control['raw'],flush=True)
    control_pass=bool(control['parsed'] and control['parsed']['category']=='unclear')
    results=[]
    for c in d['candidates']:
        canvas,mask=board(c); path=OUT/(c['id']+'_isolated.jpg'); canvas.save(path,quality=95)
        dist=cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]
        proxy=float(2*np.percentile(dist[mask],95)) if mask.any() else 0
        broad=bool(proxy>24 and int(mask.sum())>4096)
        r=ask(path); p=r['parsed']
        if p is None or p['visibility']=='unclear' or p['category']=='unclear': decision='uncertain'
        elif p['category'] in NONWIRE: decision='reject'
        elif broad: decision='uncertain'
        else: decision='keep'
        proposal=decision
        if not control_pass: decision='uncertain'
        r.update(decision_proposal=proposal,control_gate_pass=control_pass,candidate_id=c['id'],decision=decision,broad_mask_flag=broad,thickness_proxy_p95_original_pixels=proxy,mask_area_original_pixels=int(mask.sum()),next_action='resegment' if decision=='uncertain' and broad else ('inspect_crop' if decision=='uncertain' else 'none'))
        results.append(r); (OUT/(c['id']+'.json')).write_text(json.dumps(r,indent=2)); print(c['id'],decision,'BROAD',broad,p,flush=True)
    assert before=={c['id']:sha(SRC/c['mask']) for c in d['candidates']}
    v1=json.loads(Path('results/vlm_review/BEGO7553_frame_001098_review_v1_final/review.json').read_text()); v2=json.loads(Path('results/vlm_review/BEGO7553_frame_001098_review_v2/review.json').read_text())
    m1={r['candidate_id']:r['decision'] for r in v1['results']}; m2={r['candidate_id']:r['decision'] for r in v2['results']}
    s={'status':'REVIEW_COMPLETE','counts':dict(Counter(r['decision'] for r in results)),'candidate_count':len(results),'valid_json_count':sum(r['parsed'] is not None for r in results),'inference_seconds_total':sum(r['seconds'] for r in results),'model_load_seconds':load,'ground_truth_used':False,'accuracy_evaluated':False,'source_masks_unchanged':True,'empty_control_pass':bool(control['parsed'] and control['parsed']['category']=='unclear'),'changes':[{'id':r['candidate_id'],'v1':m1[r['candidate_id']],'v2':m2[r['candidate_id']],'v4':r['decision']} for r in results],'results':results}
    (OUT/'review.json').write_text(json.dumps(s,indent=2))
    lines=['# Exploratory VLM review comparison','', '|ID|v1|v2|v4|v4 object|Broad flag|Visible evidence|','|---|---|---|---|---|---|---|']
    for r in results:
        p=r['parsed'] or {}; lines.append('|'+ '|'.join([r['candidate_id'],m1[r['candidate_id']],m2[r['candidate_id']],r['decision'],p.get('category','invalid'),str(r['broad_mask_flag']),p.get('evidence','').replace('|','/')])+'|')
    (OUT/'comparison.md').write_text(chr(10).join(lines))
    with zipfile.ZipFile('vlm_review_v4_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file():z.write(p,'review_v4/'+p.name)
        z.write('review_langsam_candidates_v4.py','review_langsam_candidates_v4.py')
    print('REVIEW_V3_COMPLETE',s['counts'],'JSON',s['valid_json_count'],'EMPTY_CONTROL',s['empty_control_pass'],'SECONDS',s['inference_seconds_total'],flush=True)
if __name__=='__main__':main()
