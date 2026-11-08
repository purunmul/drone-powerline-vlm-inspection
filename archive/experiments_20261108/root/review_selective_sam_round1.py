import json,time,base64,hashlib,zipfile
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from PIL import Image
from vllm import LLM,SamplingParams
from conservative_review_policy_v1 import decide
OUT=Path('results/selective_reprocess/BEGO7553_frame_001098_round1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=json.loads((OUT/'result.json').read_text());config=json.loads(Path('results/vlm_review/BEGO7553_frame_001098_review_v4/run_config.json').read_text())
    assert not (OUT/'vlm_review.json').exists()
    t=time.perf_counter();llm=LLM(**config['settings']);load=time.perf_counter()-t;sample=SamplingParams(temperature=0,max_tokens=160)
    pad=llm.get_tokenizer().convert_tokens_to_ids('<|image_pad|>')
    def ask(path):
        url='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode();t=time.perf_counter()
        r=llm.chat([{'role':'user','content':[{'type':'image_url','image_url':{'url':url}},{'type':'text','text':config['prompt']}]}],sampling_params=sample,use_tqdm=False)[0]
        raw=r.outputs[0].text;p=None;err=None;assert r.prompt_token_ids.count(pad)>0
        try:
            p=json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
            assert p['category'] in {'vehicle','road_or_ground','wire_or_cable','pole','building_edge','vegetation','other','unclear'} and p['visibility'] in {'clear','unclear'} and r.outputs[0].finish_reason=='stop'
        except Exception as e:p=None;err=repr(e)
        return {'raw':raw,'parsed':p,'parse_error':err,'seconds':time.perf_counter()-t,'image_tokens':r.prompt_token_ids.count(pad),'input':str(path),'input_sha256':sha(path)}
    controlpath=Path('results/vlm_review/BEGO7553_frame_001098_review_v4/empty_target_control.jpg');control=ask(controlpath);passed=bool(control['parsed'] and control['parsed']['category']=='unclear')
    rows=[]
    for source in d['results']:
        r=ask(Path(source['isolated_target']));mask=np.asarray(Image.open(source['mask_local']).convert('L'))>0
        dist=cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1];proxy=float(2*np.percentile(dist[mask],95)) if mask.any() else 0
        broad=bool(proxy>24 and mask.sum()>4096);r['broad_mask_flag']=broad;decision,action,reason=decide(r,passed)
        consistent=source['positive_prompt_hits']==source['positive_prompt_count'] and source['negative_prompt_hits']==0
        if not consistent:decision='uncertain';action='inspect_prompt_and_mask';reason='prompt_constraints_not_all_satisfied'
        r.update(candidate_id=source['id'],parent_id=source['parent_id'],decision=decision,next_action=action,policy_reason=reason,thickness_proxy_p95=proxy,prompt_consistency_pass=consistent,positive_prompt_hits=source['positive_prompt_hits'],positive_prompt_count=source['positive_prompt_count'],negative_prompt_hits=source['negative_prompt_hits'],finalized=False)
        rows.append(r);print(source['id'],decision,r['parsed'],'BROAD',broad,'PROMPTS_OK',consistent,flush=True)
    s={'status':'REVIEW_COMPLETE','model_config_source':'results/vlm_review/BEGO7553_frame_001098_review_v4/run_config.json','same_model_prompt_and_policy_as_v4_1':True,'empty_control_pass':passed,'control':control,'model_load_seconds':load,'candidate_inference_seconds_total':sum(r['seconds'] for r in rows),'counts':dict(Counter(r['decision'] for r in rows)),'results':rows,'ground_truth_used':False,'accuracy_evaluated':False,'source_masks_modified':False}
    (OUT/'vlm_review.json').write_text(json.dumps(s,indent=2))
    lines=['# Selective resegmentation round 1','', 'Human-assisted points/boxes; one selected wire segment per region. Not exhaustive repair. Original predictions preserved.','', '|ID|Old area in crop|New area in crop|Positive hits|Negative hits|VLM category|Decision|','|---|---|---|---|---|---|---|']
    for x,r in zip(d['results'],rows):lines.append('|'+ '|'.join([x['id'],str(x['before_area_same_scope']),str(x['after_area_same_scope']),str(x['positive_prompt_hits'])+'/'+str(x['positive_prompt_count']),str(x['negative_prompt_hits']),str(r['parsed']['category'] if r['parsed'] else 'invalid'),r['decision']])+'|')
    (OUT/'summary.md').write_text(chr(10).join(lines))
    for x in d['results']:
        a=np.asarray(Image.open(x['mask_original']).convert('L'))>0; b=np.asarray(Image.open(x['mask_local']).convert('L'))>0;x0,y0,x1,y1=x['crop_xyxy_original'];assert a.shape==(2160,3840) and np.array_equal(a[y0:y1,x0:x1],b) and a.sum()==b.sum()
    with zipfile.ZipFile('selective_sam_round1_reviewed.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,'round1/'+str(p.relative_to(OUT)))
        for name in ['run_selective_sam_round1.py','review_selective_sam_round1.py','conservative_review_policy_v1.py']:
            z.write(name,'scripts/'+name)
        z.write('logs/selective_sam_round1.log','logs/selective_sam_round1.log')
    print('REREVIEW_COMPLETE',s['counts'],'CONTROL',passed,'COORDINATE_CHECKS_OK',flush=True)
if __name__=='__main__':main()
