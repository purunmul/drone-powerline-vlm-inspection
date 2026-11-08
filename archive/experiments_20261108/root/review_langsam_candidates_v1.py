import os
os.environ.setdefault('HF_HOME', str(__import__('pathlib').Path('.cache/huggingface').resolve()))
import json, time, hashlib, platform, importlib.metadata
from pathlib import Path
from collections import Counter
from PIL import Image
from vllm import LLM, SamplingParams
import torch
MODEL='Qwen/Qwen2.5-VL-3B-Instruct'
def main():
    src=Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1')
    out=Path('results/vlm_review/BEGO7553_frame_001098_review_v1_final')
    out.mkdir(parents=True, exist_ok=False)
    d=json.loads((src/'candidates.json').read_text())
    im=Path(d['image'])
    assert hashlib.sha256(im.read_bytes()).hexdigest()==d['image_sha256']
    scene=Image.open(im).convert('RGB'); scene.thumbnail((672,672)); scene.save(out/'scene_context.jpg',quality=95)
    prompt='''You review a proposed overhead power-line segmentation in a drone photograph. Three images are provided in order: (1) a reduced full scene for context, (2) an unmarked original crop, (3) exactly the same crop with the candidate mask in translucent red. Review ONLY the red candidate, not all wires visible in the crop. A detector can mistakenly highlight a road, car, pole, roof edge, or vegetation. Red color and an ID label are not evidence of a wire. Look at the original crop to check the underlying object. Thinness alone does not prove it is an overhead wire. Do not infer electrical safety, voltage, or damage.
Return one JSON object only with these keys:
"candidate_id": the supplied ID;
"decision": "keep", "reject", or "uncertain";
"target_type": "overhead_wire", "non_wire", "mixed", or "unclear";
"mask_quality": "follows_wire", "overbroad", "off_target", or "unclear";
"reason": one short English explanation grounded in visible evidence;
"next_action": "none", "resegment", or "inspect_crop".
Use keep only if the highlighted region predominantly follows a visibly supported overhead wire and is usable as a wire mask. Reject a clearly non-wire region. If a wire is present but the mask covers a broad road/background area, or the image cannot resolve the target, use uncertain and request resegment or inspect_crop. A partial wire segment can be kept; it need not span the full photograph. Do not merge IDs or invent missing wires.'''
    settings=dict(model=MODEL,max_model_len=4096,gpu_memory_utilization=0.80,limit_mm_per_prompt={'image':3,'video':0},max_num_seqs=1,max_num_batched_tokens=4096,enforce_eager=True,seed=0,mm_processor_kwargs={'min_pixels':784,'max_pixels':401408})
    meta={'stage':4,'source_candidates':str(src/'candidates.json'),'image_sha256':d['image_sha256'],'ground_truth_used':False,'model':MODEL,'settings':settings,'sampling':{'temperature':0,'max_tokens':256},'prompt':prompt,'versions':{k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','Pillow']},'python':platform.python_version(),'scene_image_size':scene.size,'candidates_total':len(d['candidates'])}
    (out/'run_config.json').write_text(json.dumps(meta,indent=2))
    start=time.perf_counter(); llm=LLM(**settings); load=time.perf_counter()-start
    print('MODEL_LOADED',load,flush=True)
    sampling=SamplingParams(temperature=0,max_tokens=256)
    results=[]
    for c in d['candidates']:
        paths=[out/'scene_context.jpg',src/c['original_context'],src/c['marked_context']]
        imgs=[Image.open(p).convert('RGB') for p in paths]
        content=[{'type':'image_pil','image_pil':x} for x in imgs]
        content.append({'type':'text','text':prompt+' Candidate ID: '+c['id']})
        t=time.perf_counter()
        response=llm.chat([{'role':'user','content':content}],sampling_params=sampling,use_tqdm=False)[0]
        elapsed=time.perf_counter()-t
        raw=response.outputs[0].text
        error=None; parsed=None
        try:
            text=raw.strip()
            parsed=json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
            assert parsed['candidate_id']==c['id']
            assert parsed['decision'] in ['keep','reject','uncertain']
            assert parsed['target_type'] in ['overhead_wire','non_wire','mixed','unclear']
            assert parsed['mask_quality'] in ['follows_wire','overbroad','off_target','unclear']
            assert parsed['next_action'] in ['none','resegment','inspect_crop']
            assert isinstance(parsed['reason'],str) and parsed['reason'].strip()
        except Exception as e:
            error=repr(e); parsed=None
        r={'candidate_id':c['id'],'raw_response':raw,'parsed':parsed,'parse_error':error,'decision':parsed['decision'] if parsed else 'uncertain','inference_seconds':elapsed,'input_images':[{'path':str(p),'size':x.size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,x in zip(paths,imgs)],'prompt_tokens':len(response.prompt_token_ids),'output_tokens':len(response.outputs[0].token_ids),'finish_reason':response.outputs[0].finish_reason}
        results.append(r)
        (out/(c['id']+'.json')).write_text(json.dumps(r,indent=2))
        print(c['id'],r['decision'],parsed['reason'] if parsed else error,round(elapsed,3),flush=True)
    summary={'status':'REVIEW_COMPLETE','model':MODEL,'ground_truth_used':False,'candidate_count':len(results),'counts':dict(Counter(r['decision'] for r in results)),'valid_json_count':sum(r['parsed'] is not None for r in results),'model_load_seconds':load,'inference_seconds_total':sum(r['inference_seconds'] for r in results),'source_masks_modified':False,'accuracy_evaluated':False,'results':results}
    (out/'review.json').write_text(json.dumps(summary,indent=2))
    lines=['# First-image VLM review','', 'Model: '+MODEL,'','These are model judgments, not ground-truth accuracy. Original masks are preserved.','', '| ID | Decision | Mask | Reason |','|---|---|---|---|']
    for r in results:
        p=r['parsed'] or {}; lines.append('| '+r['candidate_id']+' | '+r['decision']+' | '+p.get('mask_quality','invalid response')+' | '+p.get('reason',r['parse_error'] or '').replace('|','/')+' |')
    (out/'review_summary.md').write_text(chr(10).join(lines))
    print('REVIEW_COMPLETE',summary['counts'],'VALID_JSON',summary['valid_json_count'],flush=True)
if __name__=='__main__': main()
