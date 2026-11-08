import os
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()))
import json,time,hashlib,base64,importlib.metadata,zipfile,shutil
from collections import Counter
from vllm import LLM,SamplingParams
MODEL='Qwen/Qwen2.5-VL-3B-Instruct'
REV='66285546d2b821cf421d4f5eb2576359d3770cd3'
SRC=Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1')
OUT=Path('results/vlm_review/BEGO7553_frame_001098_review_v2')
IDENTITY='''Two matching crops are provided: first original, second with a translucent red region. Identify the actual object or surface UNDER THE RED REGION. Ignore unhighlighted objects and the ID label. Choose the best visible category, without assuming the detector is correct. Return JSON only: {"category":"vehicle|road_or_ground|wire_or_cable|pole|building_edge|vegetation|other|unclear", "description":"specific visible object", "evidence":"short visible evidence"}. Use one category value, not the list. A wire is a suspended narrow strand. A road is a broad ground surface. Use unclear when the highlighted object cannot be resolved.'''
GEOMETRY='''Two matching crops are provided: original then translucent red region. Describe only the red region's geometry and how closely it covers the visible object underneath. Do not decide its object class. Return JSON only: {"shape":"thin_strand|broad_surface|mixed|unclear", "fit":"tight|loose|unclear", "evidence":"short visible evidence"}. Choose one value per key. thin_strand means a long narrow filament, broad_surface means a filled area with substantial width, mixed means both. tight means the red region closely follows the object's visible extent. If it spreads into surrounding surfaces use loose. If unresolved use unclear.'''
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    d=json.loads((SRC/'candidates.json').read_text())
    before={c['id']:sha(SRC/c['mask']) for c in d['candidates']}
    assert sha(Path(d['image']))==d['image_sha256']
    settings=dict(model=MODEL,revision=REV,max_model_len=4096,gpu_memory_utilization=0.80,limit_mm_per_prompt={'image':3,'video':0},max_num_seqs=1,max_num_batched_tokens=4096,enforce_eager=True,seed=0,mm_processor_kwargs={'min_pixels':784,'max_pixels':401408})
    config={'model':MODEL,'revision':REV,'settings':settings,'sampling':{'temperature':0,'max_tokens':192},'identity_prompt':IDENTITY,'geometry_prompt':GEOMETRY,'versions':{k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','Pillow']},'source':str(SRC/'candidates.json'),'ground_truth_used':False,'decision_rule':'reject clear non-wire categories; keep only wire_or_cable AND thin_strand AND tight; otherwise uncertain; invalid output uncertain','changes_from_v1':['remove full-scene image','neutral target-identity question','separate independent geometry question','deterministic conservative combination','image_url data encoding instead of image_pil; same image files'],'confounded_changes':True}
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2))
    start=time.perf_counter(); llm=LLM(**settings); load=time.perf_counter()-start
    sample=SamplingParams(temperature=0,max_tokens=192)
    tok=llm.get_tokenizer(); image_id=tok.convert_tokens_to_ids('<|image_pad|>')
    def ask(paths,prompt):
        content=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()}} for p in paths]
        content.append({'type':'text','text':prompt})
        start=time.perf_counter(); result=llm.chat([{'role':'user','content':content}],sampling_params=sample,use_tqdm=False)[0]
        raw=result.outputs[0].text; elapsed=time.perf_counter()-start
        n=result.prompt_token_ids.count(image_id)
        assert n>0,'No visual image tokens found'
        return {'raw':raw,'seconds':elapsed,'image_tokens':n,'prompt_tokens':len(result.prompt_token_ids),'finish_reason':result.outputs[0].finish_reason,'inputs':[{'path':str(p),'sha256':sha(p)} for p in paths]}
    def parse(r,allowed):
        try:
            p=json.JSONDecoder().raw_decode(r['raw'][r['raw'].index('{'):])[0]
            for k,values in allowed.items(): assert p[k] in values
            assert isinstance(p['evidence'],str) and p['evidence'].strip()
            assert r['finish_reason']=='stop'
            return p,None
        except Exception as e: return None,repr(e)
    c13=next(c for c in d['candidates'] if c['id']=='P13')
    sanity=ask([SRC/c13['original_context']],'Describe the main object in this photograph in one short sentence. Do not guess objects outside the image.')
    (OUT/'sanity_original_P13.json').write_text(json.dumps(sanity,indent=2)); print('SANITY',sanity['raw'],flush=True)
    oldpath=Path('results/vlm_review/BEGO7553_frame_001098_review_v1_final')
    oldconfig=json.loads((oldpath/'run_config.json').read_text())
    replay=ask([oldpath/'scene_context.jpg',SRC/c13['original_context'],SRC/c13['marked_context']],oldconfig['prompt']+' Candidate ID: P13')
    (OUT/'v1_P13_transport_control.json').write_text(json.dumps(replay,indent=2)); print('V1_CONTROL',replay['raw'],flush=True)
    results=[]
    nonwire={'vehicle','road_or_ground','pole','building_edge','vegetation','other'}
    for c in d['candidates']:
        paths=[SRC/c['original_context'],SRC/c['marked_context']]
        a=ask(paths,IDENTITY); b=ask(paths,GEOMETRY)
        ap,ae=parse(a,{'category':nonwire|{'wire_or_cable','unclear'}})
        bp,be=parse(b,{'shape':{'thin_strand','broad_surface','mixed','unclear'},'fit':{'tight','loose','unclear'}})
        if ap is None or bp is None: decision='uncertain'; action='inspect_crop'
        elif ap['category'] in nonwire: decision='reject'; action='none'
        elif ap['category']=='wire_or_cable' and bp['shape']=='thin_strand' and bp['fit']=='tight': decision='keep'; action='none'
        else: decision='uncertain'; action='resegment' if ap['category']=='wire_or_cable' else 'inspect_crop'
        r={'candidate_id':c['id'],'identity':a,'geometry':b,'identity_parsed':ap,'geometry_parsed':bp,'parse_errors':[ae,be],'decision':decision,'next_action':action}
        results.append(r); (OUT/(c['id']+'.json')).write_text(json.dumps(r,indent=2))
        print(c['id'],decision,ap,bp,flush=True)
    assert before=={c['id']:sha(SRC/c['mask']) for c in d['candidates']}
    old=json.loads((oldpath/'review.json').read_text()); baseline={r['candidate_id']:r['decision'] for r in old['results']}
    s={'status':'REVIEW_COMPLETE','candidate_count':len(results),'counts':dict(Counter(r['decision'] for r in results)),'valid_pair_count':sum(r['identity_parsed'] is not None and r['geometry_parsed'] is not None for r in results),'model_load_seconds':load,'candidate_inference_seconds_total':sum(r[k]['seconds'] for r in results for k in ['identity','geometry']),'ground_truth_used':False,'accuracy_evaluated':False,'source_masks_unchanged':True,'baseline_counts':old['counts'],'changes':[{'id':r['candidate_id'],'v1':baseline[r['candidate_id']],'v2':r['decision']} for r in results],'results':results}
    (OUT/'review.json').write_text(json.dumps(s,indent=2))
    lines=['# VLM review v2','', 'Exploratory first-image development result; not held-out accuracy.','', '|ID|v1|v2|Object|Shape|Fit|Evidence|','|---|---|---|---|---|---|---|']
    for r in results:
        a=r['identity_parsed'] or {}; b=r['geometry_parsed'] or {}; lines.append('|'+ '|'.join([r['candidate_id'],baseline[r['candidate_id']],r['decision'],a.get('category','invalid'),b.get('shape','invalid'),b.get('fit','invalid'),a.get('evidence','') .replace('|','/')])+'|')
    (OUT/'comparison.md').write_text(chr(10).join(lines))
    shutil.copy2(SRC/'P13/comparison.jpg',OUT/'P13_original_and_mask.jpg')
    with zipfile.ZipFile('vlm_review_v2_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file(): z.write(p,'review_v2/'+p.name)
        z.write('review_langsam_candidates_v2.py','review_langsam_candidates_v2.py')
    print('REVIEW_V2_COMPLETE',s['counts'],'VALID_PAIRS',s['valid_pair_count'],'SECONDS',s['candidate_inference_seconds_total'],flush=True)
if __name__=='__main__': main()
