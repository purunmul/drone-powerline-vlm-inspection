from repository_paths import image_path
import base64
import time
from collections import Counter
from pathlib import Path

import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

BASE=Path('results')
STEMS=('BEGO7553_frame_001098','PUYL5573_frame_000165','KZIV5445_frame_000488')

def input_paths(stem):
    if stem==STEMS[0]:
        base=BASE/'langsam_baseline'
        return base/f'{stem}_tiles1024_v1',base/f'{stem}_candidates_v1'
    base=BASE/'dev_transfer_20261107_v1'/stem
    return base/'tiles',base/'candidates'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path):
    return json.loads(path.read_text())

def save_json(path,data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')

OUT=BASE/'vlm_indexed_context_20261108_v1'
FIRST=BASE/'vlm_review/BEGO7553_frame_001098_review_v4'
METHODS=('gray','indexed_context')
BACKGROUND={'vehicle','road_or_ground','pole_hardware','building','vegetation','other_background'}
CATEGORIES=BACKGROUND|{'wire','mixed','unclear'}
PROMPT=(
    'Review ONLY candidate {target} in the upper-left TARGET panel of this aerial utility inspection board. '
    'The target is the visible pixels on gray OR the region inside the cyan contour. The upper-right and lower panels, when present, give context only. '
    'A target header by itself is not a visible selection. If there are no marked or isolated pixels, set target_present false and decision recheck. '
    'Power wires are elongated cables and may be faint, curved, parallel, or cross a car, road, pole, building or tree. '
    'Use keep when the marked selection follows a recognizable wire. Use background only when the selected shape clearly follows non-wire structures and has no plausible selected wire portion. '
    'Use recheck when the selection is unclear, too small, or mixes possible wire with background. A nearby object outside the selection is not evidence. '
    'Return one JSON object only: target_id (copy {target}), target_present (true/false), decision (keep/background/recheck), '
    'category (wire/mixed/unclear/vehicle/road_or_ground/pole_hardware/building/vegetation/other_background), reason (at most 18 words about the selected shape).'
)
FONT=ImageFont.load_default(size=19)
SMALL=ImageFont.load_default(size=12)
CYAN=(20,243,235)


def inset(canvas,image,area):
    x,y,w,h=area
    fit=ImageOps.contain(image,(w,h),Image.Resampling.LANCZOS)
    canvas.paste(fit,(x+(w-fit.width)//2,y+(h-fit.height)//2))


def overview(photo, candidates, masks, target, empty):
    import cv2
    scale=420/photo.width
    thumb=photo.resize((420,round(photo.height*scale)),Image.Resampling.LANCZOS)
    arr=np.asarray(thumb).copy()
    boxes=[]
    for c in candidates:
        ident=c['id']
        if empty and ident==target:
            continue
        mask=np.array(masks[ident].resize(thumb.size,Image.Resampling.BOX))>0
        if not mask.any():
            continue
        edge=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~mask
        color=CYAN if ident==target else (255,210,68)
        arr[edge]=np.rint(arr[edge]*.30+np.array(color)*.70).astype(np.uint8)
        x0,y0,x1,y1=c['mask_bounds_original']
        boxes.append((ident,min(394,max(1,round(x0*scale))),min(221,max(1,round(y0*scale))),color))
    out=Image.fromarray(arr)
    d=ImageDraw.Draw(out)
    reserved=[]
    for ident,x,y,color in sorted(boxes,key=lambda t:t[0]!=target):
        original=(x,y)
        for _ in range(18):
            if all(abs(x-a)>24 or abs(y-b)>13 for a,b in reserved):
                break
            y=(y+15)%223
        reserved.append((x,y))
        d.line([original,(x+4,y+5)],fill=color,width=1)
        d.rectangle((x,y,x+24,y+13),fill=(11,17,24),outline=color)
        d.text((x+2,y),ident,fill=color,font=SMALL)
    return out


def board_for(method, candidate, photo, candidates, masks, empty):
    import cv2
    ident=candidate['id']
    roi=candidate['context_xyxy_original']
    raw=photo.crop(roi)
    mask=masks[ident].crop(roi)
    w,h=raw.size
    size=ImageOps.contain(raw,(544,385),Image.Resampling.LANCZOS).size
    raw_fit=raw.resize(size,Image.Resampling.LANCZOS)
    if method=='gray':
        gray=Image.new('RGB',(w,h),(150,150,150))
        target=gray if empty else Image.composite(raw,gray,mask)
        target=target.resize(size,Image.Resampling.LANCZOS)
    else:
        arr=np.asarray(raw_fit).copy()
        if not empty:
            small=np.asarray(mask.resize(size,Image.Resampling.BOX))>0
            outline=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~small
            arr[outline]=np.rint(arr[outline]*.45+np.array(CYAN)*.55).astype(np.uint8)
        target=Image.fromarray(arr)
    board=Image.new('RGB',(1120,700),(28,34,44))
    d=ImageDraw.Draw(board)
    d.text((14,10),f'TARGET {ident}',fill=CYAN,font=FONT)
    d.text((574,10),'ORIGINAL CROP' if method=='indexed_context' else 'CONTEXT UNAVAILABLE',fill='white',font=FONT)
    d.text((14,427),'NUMBERED FULL IMAGE' if method=='indexed_context' else 'FULL IMAGE UNAVAILABLE',fill='white',font=FONT)
    inset(board,target,(8,35,544,385))
    if method=='indexed_context':
        inset(board,raw_fit,(568,35,544,385))
        inset(board,overview(photo,candidates,masks,ident,empty),(235,451,650,240))
    return board


def prepare():
    OUT.mkdir(exist_ok=False)
    manifest=[]
    for stem in STEMS:
        raw_dir,candidate_dir=input_paths(stem)
        report=read_json(raw_dir/'result.json')
        source=image_path(report['image'])
        assert digest(source)==report['sha256']
        photo=Image.open(source).convert('RGB')
        candidates=read_json(candidate_dir/'candidates.json')['candidates']
        masks={c['id']:Image.open(candidate_dir/c['mask']).convert('L') for c in candidates}
        folder=OUT/stem
        folder.mkdir()
        for i,c in enumerate(candidates):
            ident=c['id']
            for control in (False,True) if i==0 else (False,):
                for method in METHODS:
                    name=f'{ident}_{method}'+('_empty' if control else '')+'.jpg'
                    path=folder/name
                    board_for(method,c,photo,candidates,masks,control).save(path,quality=92)
                    manifest.append(dict(stem=stem,candidate=ident,method=method,control=control,
                        image=str(path.relative_to(OUT)),image_sha256=digest(path),mask_sha256=digest(candidate_dir/c['mask'])))
    assert len(manifest)==108
    save_json(OUT/'manifest.json',manifest)
    print('INPUTS_READY',len(manifest),flush=True)


def parse(output, target, finish):
    try:
        data=json.JSONDecoder().raw_decode(output[output.index('{'):])[0]
        assert finish=='stop' and data['target_id']==target and type(data['target_present']) is bool
        assert data['decision'] in {'keep','background','recheck'} and data['category'] in CATEGORIES
        assert isinstance(data['reason'],str) and data['reason'].strip()
        return data
    except (AssertionError,ValueError,KeyError,TypeError):
        return None


def infer():
    import json
    from vllm import LLM,SamplingParams
    baseline=read_json(FIRST/'run_config.json')
    settings=baseline['settings'].copy()
    settings.update(gpu_memory_utilization=.25,cpu_offload_gb=5.5,max_model_len=2048,max_num_batched_tokens=2048)
    sample=dict(temperature=0,max_tokens=160)
    save_json(OUT/'run_config.json',dict(settings=settings,sampling=sample,prompt=PROMPT,methods=METHODS,
        control_rule='per photo and method, target_present=false and decision=recheck required for automatic exclusions',
        selection_rule='exclude only valid background decision with present target and explicit background category, under passing control',
        reference_used=False,weights_changed=False))
    entries=read_json(OUT/'manifest.json')
    start=time.perf_counter()
    model=LLM(**settings)
    options=SamplingParams(**sample)
    token=model.get_tokenizer().convert_tokens_to_ids('<|image_pad|>')
    print('QWEN_READY',round(time.perf_counter()-start,2),flush=True)
    with (OUT/'responses.jsonl').open('x') as log:
        for index,entry in enumerate(entries,1):
            path=OUT/entry['image']
            assert digest(path)==entry['image_sha256']
            image='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
            content=[{'type':'image_url','image_url':{'url':image}},
                     {'type':'text','text':PROMPT.format(target=entry['candidate'])}]
            result=model.chat([{'role':'user','content':content}],sampling_params=options,use_tqdm=False)[0]
            value=result.outputs[0]
            parsed=parse(value.text,entry['candidate'],value.finish_reason)
            tokens=result.prompt_token_ids.count(token)
            assert tokens>0
            record=dict(entry,raw=value.text,parsed=parsed,finish_reason=value.finish_reason,image_tokens=tokens)
            log.write(json.dumps(record,ensure_ascii=False)+'\n')
            log.flush()
            if index%12==0 or entry['control']:
                print('REVIEW',index,entry['stem'][:8],entry['method'],entry['control'],parsed,flush=True)
    save_json(OUT/'inference_complete.json',dict(records=len(entries),seconds=time.perf_counter()-start))
    print('INFERENCE_COMPLETE',len(entries),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'infer':infer}[sys.argv[1]]()
