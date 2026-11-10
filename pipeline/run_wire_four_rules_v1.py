import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import run_wire_span_continuity_v1 as span
import run_wire_contacts_v1 as prior
from run_wire_color_prior_v1 import palette_profiles

OUT = Path('results/wire_four_rules_20261109_v1')
CONTACT_PROMPT = ('The right panel labels one wire end and candidate points A and B. '
                  'First check the visible letters. If there are no letters answer NONE. '
                  'Use the unmarked left panel to check whether the wire end physically approaches '
                  'the pole hardware at A or B. A nearby pole alone is insufficient. '
                  'If both positions are off the wire direction answer NONE. If ambiguous answer UNSURE. '
                  'Return exactly one of NONE, UNSURE, A, B.')
COLOR_PROMPT = ('These numbered crops are tentative aerial wire fragments. Korean conductor '
                'identification colors are brown, black, gray, blue and green-yellow. The marking '
                'may only be at an end; shade and reflections shift the apparent color. '
                'Choose apparent color variability: WIDE if shadows, reflections or mixed tones '
                'need a broad prior, MEDIUM otherwise. Return exactly WIDE or MEDIUM.')
METHODS=('previous_continuity','continuity','with_poles','with_color','four_rules')


def read(folder,name):
    return span.read_json(folder/name)


def signature(lab, points):
    offsets=np.arange(-8,9)
    ids=np.unique(np.linspace(0,len(points)-1,min(13,len(points))).round().astype(int))
    widths=[]
    colors=[]
    for i in ids:
        pt=points[i]
        if np.any(pt<10) or np.any(pt>=np.array(lab.shape[:2])-10):
            continue
        direct=span.unit(points[min(i+5,len(points)-1)]-points[max(0,i-5)])
        normal=np.array([-direct[1],direct[0]])
        coords=pt+offsets[:,None]*normal
        section=cv2.remap(lab,coords[:,1].astype(np.float32)[:,None],
                         coords[:,0].astype(np.float32)[:,None],cv2.INTER_LINEAR)[:,0]
        bg=np.median(np.r_[section[:3,0],section[-3:,0]])
        contrast=section[:,0]-bg
        center=contrast[8]
        if abs(center)<4 or abs(contrast[0])>.65*abs(center) or abs(contrast[-1])>.65*abs(center):
            continue
        support=contrast*np.sign(center)>=.5*abs(center)
        left=8
        right=8
        while left>0 and support[left-1]:
            left-=1
        while right<16 and support[right+1]:
            right+=1
        if left==0 or right==16 or right-left+1>10:
            continue
        widths.append(right-left+1)
        colors.append(section[8])
    if len(widths)<3:
        return dict(width=None,mad=None,color=None,n=len(widths))
    width=float(np.median(widths))
    return dict(width=width,mad=float(np.median(np.abs(np.array(widths)-width))),
                color=np.median(colors,axis=0).tolist(),n=len(widths))


def color_board(photo, paths, seeds, target):
    from PIL import ImageFont

    board=Image.new('RGB',(640,360),(25,29,35))
    draw=ImageDraw.Draw(board)
    font=ImageFont.load_default(size=15)
    for k,index in enumerate(seeds):
        p=np.asarray(paths[index]['points'])
        y,x=p[len(p)//2]
        xx=int(np.clip(x-110,0,max(0,photo.shape[1]-220)))
        yy=int(np.clip(y-70,0,max(0,photo.shape[0]-140)))
        crop=Image.fromarray(photo[yy:yy+140,xx:xx+220])
        a,b=(k%2)*320,(k//2)*180
        board.paste(crop.resize((310,156),Image.Resampling.LANCZOS),(a+5,b+23))
        draw.text((a+6,b+4),f'{k+1}  tentative fragment',font=font,fill='white')
    board.save(target,quality=93)


def prepare():
    OUT.mkdir(exist_ok=False)
    (OUT/'review_inputs').mkdir()
    manifest=read(prior.OUT,'manifest.json')
    jobs=[]
    for stem in span.STEMS:
        folder=OUT/stem
        folder.mkdir()
        paths=prior.get_paths(prior.OUT/stem)
        photo=np.asarray(Image.open(span.source(stem)).convert('RGB'))
        lab=cv2.cvtColor(photo.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        signatures=[signature(lab,row['points']) for row in paths]
        ranked=sorted(range(len(paths)),key=lambda k:-(paths[k]['ridge']*min(paths[k]['arc'],140)))
        seeds=[]
        for k in ranked:
            if paths[k]['ridge']<.60 or paths[k]['arc']<42 or signatures[k]['width'] is None:
                continue
            middle=paths[k]['points'][len(paths[k]['points'])//2]
            if any(np.linalg.norm(middle-p)<130 for _,p in seeds):
                continue
            seeds.append((k,middle))
            if len(seeds)==4:
                break
        seeds=[k for k,_ in seeds]
        span.save(folder/'evidence.json',dict(signatures=signatures,seeds=seeds))
        target=OUT/'review_inputs'/f'{stem}_color.jpg'
        if seeds:
            color_board(photo,paths,seeds,target)
            jobs.append(dict(stem=stem,id='COLOR',image=str(target),prompt=COLOR_PROMPT))
        for case in manifest:
            if case['stem']!=stem:
                continue
            options={k:v for k,v in case['options'].items() if 'path' in v}
            # Controls receive the same prompt and neutral measurement description.
            widths=[signatures[v['path']]['width'] for v in options.values()]
            width=next((w for w in widths if w is not None),None)
            measurement=(f' Estimated visible wire width: {width:.1f} pixels.' if width is not None else '')
            jobs.append(dict(stem=stem,id=case['id'],image=case['image'],prompt=CONTACT_PROMPT+measurement))
        print(stem,'seed',len(seeds),'width',sum(s['width'] is not None for s in signatures),flush=True)
    span.save(OUT/'jobs.json',jobs)


def review():
    from vllm import LLM, SamplingParams

    jobs=read(OUT,'jobs.json')
    options=read(prior.OUT,'7b_config.json')['settings']
    options.update(max_num_seqs=1)
    llm=LLM(**options)
    settings=SamplingParams(temperature=0,max_tokens=10)
    rows=[]
    for i in range(0,len(jobs),3):
        batch=jobs[i:i+3]
        messages=[]
        for item in batch:
            data=base64.b64encode(Path(item['image']).read_bytes()).decode('ascii')
            messages.append([dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+data)),
                                                       dict(type='text',text=item['prompt'])])])
        for item,answer in zip(batch,llm.chat(messages,settings,use_tqdm=False),strict=True):
            output=answer.outputs[0]
            token=output.text.strip().strip('.*!').upper()
            allowed=('WIDE','MEDIUM') if item['id']=='COLOR' else ('A','B','NONE','UNSURE')
            rows.append(dict(stem=item['stem'],id=item['id'],choice=token if token in allowed else None,
                             raw=output.text,prompt_tokens=len(answer.prompt_token_ids or []),
                             output_tokens=len(output.token_ids)))
    assert len(rows)==len(jobs)
    target=OUT/'responses.json'
    if target.exists():
        raise FileExistsError(target)
    tmp=target.with_suffix('.tmp')
    span.save(tmp,rows)
    tmp.replace(target)
    print('7B_SAVED',len(rows),flush=True)
    llm.llm_engine.engine_core.shutdown()

def pole_contacts(stem, paths, manifest, answers):
    poles={row['id']:row for row in read(prior.OUT/stem,'poles.json')}
    labels={r['id']:r['choice'] for r in answers if r['stem']==stem}
    controls=[labels.get(k) for k in ('CTRL_EMPTY','CTRL_WRONG')]
    trusted=controls==['NONE','NONE']
    contacts=[]
    for case in manifest:
        if case['stem']!=stem or case['id'].startswith('CTRL'):
            continue
        for letter,row in case['options'].items():
            if 'pole' not in row:
                continue
            path=paths[row['path']]
            pole=poles[row['pole']]
            supported=pole['score']>=.34 and path['arc']>=36 and path['ridge']>=.50
            confirmed=trusted and labels.get(case['id'])==letter
            if supported or confirmed:
                contacts.append(dict(**row,vlm=confirmed))
    return contacts,controls,trusted


def color_adjust(a,b,seeds,standard):
    if a is None or b is None:
        return 0.0
    a,b=np.array(a),np.array(b)
    mid=(a+b)/2
    match=max(0.0,1-float(np.linalg.norm((a-b)/[44,20,20])))
    observed=max((max(0.0,1-float(np.linalg.norm((mid-c)/[42,20,20]))) for c in seeds),default=0.0)
    initial=max(max(0.0,1-float(np.linalg.norm((mid-c)/r))) for c,r in standard)
    return .13*match*(.78*observed+.22*initial)


def width_adjust(a,b):
    if a['width'] is None or b['width'] is None or max(a['mad'],b['mad'])>2:
        return 0.0
    tolerance=1.6+.2*(a['mad']+b['mad'])
    ratio=max(a['width'],b['width'])/min(a['width'],b['width'])
    return .10*max(0.0,1-(ratio-1)/tolerance) if ratio<=tolerance else -.06


def select(observed,links,contacts,evidence,standard,use_color=False,use_width=False):
    output=observed.copy()
    taken=set()
    blocked=[]
    selected=[]
    evaluated=[]
    for link in links:
        a,b=(evidence[link[k]] for k in ('a','b'))
        score=link['priority']-link['color_bonus']
        c=color_adjust(a['color'],b['color'],*standard) if use_color else 0.0
        w=width_adjust(a,b) if use_width else 0.0
        evaluated.append((score+c+w,link,c,w))
    for score,link,c,w in sorted(evaluated,key=lambda r:(not r[1]['strong'],-r[0])):
        if not link['strong'] and (score<.55 or link['ridge']<.36):
            continue
        ends={(link['a'],link['side_a']),(link['b'],link['side_b'])}
        if ends & taken:
            continue
        points=np.array(link['pixels'])
        if any((p['path'],p['side']) in ends or np.min(np.linalg.norm(points-p['yx'],axis=1))<=8 for p in contacts):
            blocked.append(link['id'])
            continue
        output[tuple(points.T)]=True
        taken |= ends
        selected.append(dict(id=link['id'],color=round(c,4),width=round(w,4),score=round(score,4)))
    return output,dict(links=selected,blocked_at_pole=blocked)


def predictions(stem,manifest,answers):
    from evaluate_first_image_v1 import readmask
    folder=prior.OUT/stem
    paths=prior.get_paths(folder)
    links=read(folder,'links.json')
    ev=read(OUT/stem,'evidence.json')
    sig=ev['signatures']
    assert len(paths)==len(sig)
    base=readmask(span.baseline_path(stem))
    original=readmask(prior.OUT/'review'/stem/'previous_rules_centerline.png')
    old, _=span.span_policy(base.shape,paths,links,read(folder,'poles.json'))
    saved=readmask(prior.OUT/'review'/stem/'preserve_fragments_centerline.png')
    assert np.array_equal(old,saved),stem
    observed=np.zeros_like(base)
    for p in paths:
        observed[tuple(p['points'].T)]=True
    assert not (observed&~base).any()
    contacts,controls,passed=pole_contacts(stem,paths,manifest,answers)
    light=next((r['choice'] for r in answers if r['stem']==stem and r['id']=='COLOR'),None)
    scale=1.30 if light=='WIDE' else 1.12
    seeds=[np.asarray(sig[i]['color']) for i in ev['seeds']]
    standard=(seeds,[(center,radius*scale) for center,radius in palette_profiles()])
    variants=dict(previous_continuity=original)
    audit={}
    for name,stops,color,width in [('continuity',[],False,False),('with_poles',contacts,False,False),
                                   ('with_color',contacts,True,False),('four_rules',contacts,True,True)]:
        variants[name],audit[name]=select(observed,links,stops,sig,standard,color,width)
    assert all(np.all(line[observed]) for name,line in variants.items() if name!='previous_continuity')
    return variants,dict(policies=audit,contacts=contacts,controls=controls,vlm_gate=passed,
                         color_range=light,seed_count=len(seeds),measured_widths=sum(x['width'] is not None for x in sig),
                         candidate_links=len(links),observed_pixels=int(observed.sum()),
                         changed_by_poles=int(np.count_nonzero(variants['continuity']^variants['with_poles'])),
                         changed_by_color=int(np.count_nonzero(variants['with_poles']^variants['with_color'])),
                         changed_by_width=int(np.count_nonzero(variants['with_color']^variants['four_rules'])))


def render(folder,stem,variants):
    from PIL import ImageFont

    photo=np.asarray(Image.open(span.source(stem)).convert('RGB'))
    overlay_images=[]
    for name in ('previous_continuity','four_rules'):
        line=variants[name]
        rgb=photo.copy()
        pixels=line.astype(np.uint8)
        rgb[cv2.dilate(pixels,np.ones((5,5),np.uint8))>0]=(3,245,226)
        image=Image.fromarray(rgb)
        overlay_images.append(image.resize((800,450),Image.Resampling.LANCZOS))
        if name=='four_rules':
            image.save(folder/'prediction_on_original.jpg',quality=93)
    board=Image.new('RGB',(1600,482),(20,24,30))
    draw=ImageDraw.Draw(board)
    font=ImageFont.load_default(size=18)
    for index,(label,image) in enumerate(zip(('Previous continuity','Four rules'),overlay_images,strict=True)):
        x=800*index
        draw.text((x+7,4),label,font=font,fill='white')
        board.paste(image,(x,32))
    board.save(folder/'comparison.jpg',quality=93)


def evaluate():
    import csv
    from evaluate_first_image_v1 import metrics,readmask
    from scipy.ndimage import distance_transform_edt

    manifest=read(prior.OUT,'manifest.json')
    jobs=read(OUT,'jobs.json')
    answers=read(OUT,'responses.json')
    assert [(r['stem'],r['id']) for r in jobs]==[(r['stem'],r['id']) for r in answers]
    frozen={stem:predictions(stem,manifest,answers) for stem in span.STEMS}
    dst=OUT/'review'
    dst.mkdir(exist_ok=False)
    rows=[]
    for stem,(variants,audit) in frozen.items():
        folder=dst/stem
        folder.mkdir()
        render(folder,stem,variants)
        for name,line in variants.items():
            Image.fromarray(line.astype(np.uint8)*255).save(folder/f'{name}_centerline.png')
        directory=(span.BASE/'diagnostic_eval'/f'{stem}_v1' if stem==span.STEMS[0]
                   else span.BASE/'dev_transfer_20261107_v1'/stem/'evaluation')
        ref=readmask(directory/'reference_centerlines.png')
        refdist=distance_transform_edt(~ref)
        xml=span.annotation_paths(stem,ref.shape)
        for name,line in variants.items():
            scores,dist=metrics(line,ref,refdist)
            score=next(x for x in scores if x['tolerance_pixels']==5)
            rows.append(dict(stem=stem,method=name,**score,**span.continuity(dist,xml)))
    with (dst/'metrics.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys=('centerline_precision','centerline_recall','centerline_f1','longest_run_mean')
    macro={name:{k:float(np.mean([r[k] for r in rows if r['method']==name])) for k in keys} for name in METHODS}
    usage={key:sum(r[key] for r in answers) for key in ('prompt_tokens','output_tokens')}
    span.save(dst/'summary.json',dict(macro=macro,audit={s:a for s,(_,a) in frozen.items()},
              qwen_usage=usage,tolerance_pixels=5,development_only=True))
    for name,scores in macro.items():
        print(name,*(f'{100*v:.2f}' for v in scores.values()),flush=True)
    for stem,(_,a) in frozen.items():
        print(stem,'pole',len(a['contacts']),'vlm',sum(p['vlm'] for p in a['contacts']),
              'ctrl',a['controls'],'range',a['color_range'],'seed',a['seed_count'],'width',a['measured_widths'],
              'changed',a['changed_by_poles'],a['changed_by_color'],a['changed_by_width'],flush=True)
    print('TOKENS',usage,flush=True)
    print('VERIFIED',dst,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','review','evaluate'))
    {'prepare':prepare,'review':review,'evaluate':evaluate}[parser.parse_args().stage]()
