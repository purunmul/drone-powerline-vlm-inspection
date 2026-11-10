from repository_paths import image_path
import json
from pathlib import Path

from run_vlm_indexed_context_20261108_v1 import BASE,STEMS,digest,input_paths,read_json,save_json

OUT=BASE/'vlm_zoom_recheck_round1_v1'
BUDGET=4
PAD=64
PROMPT=(
 'Compare two views of the same aerial crop. Left is the original. Right is identical with cyan lines from an existing prediction. '
 'A cable may be dark or bright, thin, curved, or parallel to another cable. Cyan alone is not evidence of a real cable. '
 'Look for actual overhead wires visible on the LEFT which have missing cyan segments on the RIGHT. '
 'Return one JSON object only: region (copy {region}), visibility (none, uncertain, clear), '
 'missing (none, short, long), reason (up to 15 words). If the crop is empty or no wire is visible, choose none for both. '
 'Use long for a clearly missing cable or substantial segment. Ground markings, roof edges, and branches are background.'
)

def baseline_path(stem):
    return BASE/'skeleton_order_ablation_20261107_v1'/stem/'A_all_candidates/skeleton_then_union/centerline.png'

def prepare():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from scipy.ndimage import distance_transform_edt,maximum_filter
    from evaluate_first_image_v1 import readmask
    OUT.mkdir(exist_ok=False)
    font=ImageFont.load_default(size=20)
    manifest=[]
    for stem in STEMS:
        raw=read_json(input_paths(stem)[0]/'result.json')
        photo_path=image_path(raw['image'])
        assert digest(photo_path)==raw['sha256']
        photo=Image.open(photo_path).convert('RGB')
        w,h=photo.size
        line=readmask(baseline_path(stem))
        assert line.shape==(h,w) and w%6==h%4==0
        boxes=np.zeros_like(line)
        for c in raw['candidates']:
            a,b,d,e=c['box_xyxy_original']
            boxes[max(0,int(np.ceil(b))):min(h,int(np.floor(e))+1),max(0,int(np.ceil(a))):min(w,int(np.floor(d))+1)]=True
        gap=boxes & (distance_transform_edt(~line)>24)
        folder=OUT/stem
        folder.mkdir()
        for i in range(25):
            control=i==24
            col,row=i%6,i//6
            x,y=col*(w//6),row*(h//4)
            x1,y1=x+w//6,y+h//4
            ident='CTRL' if control else f'R{i+1:02d}'
            if control:
                a=b=Image.new('RGB',(w//6,h//4),(150,150,150))
            else:
                a=photo.crop((x,y,x1,y1))
                arr=np.array(a)
                arr[maximum_filter(line[y:y1,x:x1],size=3)]=(25,235,226)
                b=Image.fromarray(arr)
            board=Image.new('RGB',(1120,504),(23,29,40))
            draw=ImageDraw.Draw(board)
            for left,title,panel in ((0,'ORIGINAL',a),(560,'CURRENT PREDICTION (CYAN)',b)):
                draw.text((left+13,7),f'{ident} | {title}',font=font,fill='white')
                board.paste(panel.resize((540,456),Image.Resampling.LANCZOS),(left+10,34))
            path=folder/f'{ident}.jpg'
            board.save(path,quality=91)
            core=None if control else [x,y,x1,y1]
            zoom=None if control else [max(0,x-PAD),max(0,y-PAD),min(w,x1+PAD),min(h,y1+PAD)]
            manifest.append(dict(stem=stem,id=ident,core=core,zoom=zoom,control=control,
                fixed_score=0 if control else int(gap[y:y1,x:x1].sum()),board=str(path.relative_to(OUT)),board_sha=digest(path)))
    assert len(manifest)==75
    save_json(OUT/'manifest.json',manifest)
    save_json(OUT/'protocol.json',dict(images=STEMS,grid=[6,4],budget_per_method=BUDGET,padding_px=PAD,
        fixed='rank core area covered by existing raw boxes but >24px from baseline centerline',
        vlm='rank reported missing extent, then visible wire certainty, then row-major region ID',
        valid_control='no wire and no missing line on blank pair; no affected image dropped from diagnostic results',
        append='candidate skeletonized inside padded crop and only core pixels added to fixed baseline',
        evaluation_tolerances=[3,5,10], xml_access='evaluation only'))
    print('PREPARED',len(manifest),flush=True)

def select():
    import base64
    import time
    from vllm import LLM,SamplingParams
    settings=read_json(BASE/'vlm_indexed_context_neutral_20261108_v1/run_config.json')['settings']
    settings=dict(settings)
    sample=dict(temperature=0,max_tokens=110)
    save_json(OUT/'qwen_config.json',dict(settings=settings,sampling=sample,prompt=PROMPT))
    manifest=read_json(OUT/'manifest.json')
    llm=LLM(**settings)
    options=SamplingParams(**sample)
    responses=[]
    for start in range(0,len(manifest),4):
        batch=manifest[start:start+4]
        messages=[]
        for r in batch:
            path=OUT/r['board']
            assert digest(path)==r['board_sha']
            url='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
            messages.append([{'role':'user','content':[{'type':'image_url','image_url':{'url':url}},
                {'type':'text','text':PROMPT.format(region=r['id'])}]}])
        for r,result in zip(batch,llm.chat(messages,sampling_params=options,use_tqdm=False),strict=True):
            answer=result.outputs[0]
            parsed=None
            try:
                parsed=json.JSONDecoder().raw_decode(answer.text[answer.text.index('{'):])[0]
                assert answer.finish_reason=='stop' and parsed['region']==r['id']
                assert parsed['visibility'] in ('none','uncertain','clear') and parsed['missing'] in ('none','short','long')
                assert isinstance(parsed['reason'],str) and parsed['reason'].strip()
            except (AssertionError,ValueError,TypeError,KeyError):
                parsed=None
            responses.append(dict(r,raw=answer.text,parsed=parsed))
        if start%12==0:
            print('QWEN',len(responses),flush=True)
    assert len(responses)==75
    save_json(OUT/'responses.json',responses)
    choices={}
    for stem in STEMS:
        subset=[r for r in responses if r['stem']==stem and not r['control']]
        control=next(r for r in responses if r['stem']==stem and r['control'])['parsed'] or {}
        def score(r):
            v=r['parsed'] or {}
            visibility={'none':0,'uncertain':1,'clear':2}.get(v.get('visibility'),0)
            extent={'none':0,'short':1,'long':2}.get(v.get('missing'),0)
            return (extent if visibility else 0,visibility if extent else 0)
        fixed=sorted(subset,key=lambda r:(-r['fixed_score'],r['id']))[:BUDGET]
        vlm=sorted(subset,key=lambda r:(-score(r)[0],-score(r)[1],r['id']))[:BUDGET]
        choices[stem]=dict(fixed=[r['id'] for r in fixed],vlm=[r['id'] for r in vlm],
            control_pass=control.get('visibility')==control.get('missing')=='none',valid=sum(r['parsed'] is not None for r in subset),
            positive_vlm_selections=sum(score(r)[0]>0 for r in vlm),vlm_rank={r['id']:score(r) for r in subset})
    save_json(OUT/'choices.json',choices)
    print('SELECTION_COMPLETE',choices.keys(),flush=True)

def rerun():
    import time
    import numpy as np
    import torch
    from PIL import Image
    from lang_sam import LangSAM
    manifest=read_json(OUT/'manifest.json')
    choices=read_json(OUT/'choices.json')
    model=LangSAM(sam_type='sam2.1_hiera_small')
    info=[]
    for stem in STEMS:
        raw=read_json(input_paths(stem)[0]/'result.json')
        assert raw['prompt']=='power line.' and raw['sam_type']=='sam2.1_hiera_small'
        assert raw['box_threshold']==.30 and raw['text_threshold']==.25
        assert digest(image_path(raw['image']))==raw['sha256']
        photo=Image.open(image_path(raw['image'])).convert('RGB')
        ids=set(choices[stem]['fixed'])|set(choices[stem]['vlm'])
        for r in manifest:
            if r['stem']!=stem or r['id'] not in ids:
                continue
            patch=photo.crop(r['zoom'])
            start=time.perf_counter()
            with torch.inference_mode():
                pred=model.predict([patch],[raw['prompt']],box_threshold=raw['box_threshold'],text_threshold=raw['text_threshold'])[0]
            torch.cuda.synchronize()
            w,h=patch.size
            boxes=np.asarray(pred['boxes'],dtype=np.float32).reshape(-1,4)
            masks=np.asarray(pred['masks'],dtype=bool).reshape(len(boxes),h,w)
            scores=np.asarray(pred['scores'],dtype=np.float32).reshape(-1)
            assert len(scores)==len(boxes)
            target=OUT/stem/f"{r['id']}_zoom.npz"
            np.savez_compressed(target,masks=masks,boxes=boxes,scores=scores)
            info.append(dict(stem=stem,id=r['id'],count=len(boxes),seconds=time.perf_counter()-start))
        print('LANGSAM',stem[:8],len(ids),flush=True)
    save_json(OUT/'zoom_inference.json',info)
    print('ZOOM_COMPLETE',len(info),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'select':select,'rerun':rerun}[sys.argv[1]]()
