from repository_paths import image_path
import argparse
import base64
import csv
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from run_vlm_zoom_recheck_round1_v1 import BASE, STEMS, baseline_path, input_paths, read_json

NAME = 'wire_span_continuity_20261108_v1'
OUT = BASE / NAME
MODELS = {'3b':'Qwen/Qwen2.5-VL-3B-Instruct','7b':'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'}
GAP, LATERAL, ANGLE, MIN_LENGTH = 24, 5, 23, 22
COLORS=((3,245,226),(255,215,40),(255,105,92))


def save(path, data):
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def source(stem):
    return image_path(read_json(input_paths(stem)[0]/'result.json')['image'])


def unit(vector):
    import numpy as np
    return vector/max(float(np.linalg.norm(vector)),1e-6)


def trace_without_corner_edges(line):
    import numpy as np

    h,w=line.shape
    pixels={tuple(p) for p in np.argwhere(line)}
    offsets=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
    graph={}
    for y,x in pixels:
        graph[y,x]=[(y+dy,x+dx) for dy,dx in offsets if (y+dy,x+dx) in pixels
                    and (not (dy and dx) or ((y+dy,x) not in pixels and (y,x+dx) not in pixels))]
    visited=set()
    for first in sorted(graph,key=lambda p:(len(graph[p])==2,p)):
        for second in graph[first]:
            edge=frozenset((first,second))
            if edge in visited:
                continue
            path=[first,second]
            visited.add(edge)
            while len(graph[path[-1]])==2:
                tail=next(p for p in graph[path[-1]] if p!=path[-2])
                edge=frozenset((path[-1],tail))
                if edge in visited:
                    break
                visited.add(edge)
                path.append(tail)
            yield np.asarray(path,dtype=int)


def line_evidence(lab, points, direction):
    import numpy as np

    y,x=points.T
    normal=np.rint(4*np.array([-direction[1],direction[0]])).astype(int)
    valid=((points>=4)&(points<np.array(lab.shape[:2])-4)).all(axis=1)
    if not valid.any():
        return 0.0
    y,x=y[valid],x[valid]
    center=lab[y,x,0]
    left=lab[y+normal[0],x+normal[1],0]-center
    right=lab[y-normal[0],x-normal[1],0]-center
    return float(np.mean(((left>=3)&(right>=3))|((left<=-3)&(right<=-3))))


def fragments(stem, lab):
    import cv2
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    from evaluate_first_image_v1 import readmask

    _,folder=input_paths(stem)
    info=read_json(folder/'candidates.json')
    baseline=readmask(baseline_path(stem))
    raw=np.zeros_like(baseline)
    paths=[]
    for item in info['candidates']:
        mask=readmask(folder/item['mask'])
        x,y,w,h=cv2.boundingRect(mask.astype(np.uint8))
        if not (w and h):
            continue
        # A one-pixel margin reproduces full-image skeletonization at crop borders.
        y0,x0=max(0,y-1),max(0,x-1)
        y1,x1=min(mask.shape[0],y+h+1),min(mask.shape[1],x+w+1)
        cropped=mask[y0:y1,x0:x1]
        centerline=skeletonize(cropped)
        raw[y0:y1,x0:x1] |= centerline
        width=distance_transform_edt(np.pad(cropped,1))[1:-1,1:-1]
        for piece in trace_without_corner_edges(centerline):
            if len(piece)<MIN_LENGTH:
                continue
            displacement=piece[-1]-piece[0]
            chord=float(np.linalg.norm(displacement))
            arc=float(np.linalg.norm(np.diff(piece,axis=0),axis=1).sum())
            if chord<MIN_LENGTH-2 or arc>1.28*chord:
                continue
            direction=unit(displacement)
            normal=np.array([-direction[1],direction[0]])
            deviation=np.abs((piece-piece[0])@normal)
            if np.quantile(deviation,.90)>max(3,min(11,chord*.035)):
                continue
            narrow=float(np.mean(width[tuple(piece.T)]<=6))
            piece=piece+(y0,x0)
            ridge=line_evidence(lab,piece,direction)
            if narrow<.67 and ridge<.38:
                continue
            paths.append(dict(points=piece,arc=arc,narrow=narrow,ridge=ridge))
    assert np.array_equal(raw,baseline),stem
    return paths


def gap_candidates(paths, lab, baseline):
    import numpy as np
    from scipy.spatial import cKDTree
    from skimage.draw import line

    ends=[]
    for index,path in enumerate(paths):
        for side,ordered in enumerate((path['points'],path['points'][::-1])):
            endpoint=ordered[0]
            look=min(12,len(ordered)-1)
            ends.append((index,side,endpoint,unit(endpoint-ordered[look]),
                         np.median(lab[tuple(ordered[:look+1].T)],axis=0)))
    if not ends:
        return []
    tree=cKDTree([item[2] for item in ends])
    proposals=[]
    for ai,bi in sorted(tree.query_pairs(GAP)):
        a,aside,p,u,c=ends[ai]
        b,bside,q,v,d=ends[bi]
        if a==b:
            continue
        change=q-p
        distance=float(np.linalg.norm(change))
        if distance<2:
            continue
        normal=np.array([-u[1],u[0]])
        other_normal=np.array([-v[1],v[0]])
        lateral=max(abs(float(normal@change)),abs(float(other_normal@change)))
        heading=change/distance
        turn=max(float(np.degrees(np.arccos(np.clip(u@heading,-1,1)))),
                 float(np.degrees(np.arccos(np.clip(-v@heading,-1,1)))))
        if lateral>LATERAL or turn>ANGLE:
            continue
        pixels=np.column_stack(line(*p,*q))
        interior=pixels[1:-1]
        if len(interior)<2 or np.mean(baseline[tuple(interior.T)])>.65:
            continue
        ridge=line_evidence(lab,interior,heading)
        # A weak color tie-break changes ordering only, never geometry or eligibility.
        color=.06*max(0,1-float(np.linalg.norm((c-d)/np.array([38,18,18]))))
        strong=bool((distance<=8 and turn<=14) or (distance<=16 and turn<=14 and ridge>=.55))
        if not strong and ridge<.20 and distance>10:
            continue
        priority=(1-distance/GAP)*.50+(1-turn/ANGLE)*.30+ridge*.20+color
        proposals.append(dict(a=a,b=b,side_a=aside,side_b=bside,ends=[p.tolist(),q.tolist()],
                              pixels=pixels.tolist(),gap=distance,lateral=lateral,turn=turn,ridge=ridge,
                              color_bonus=color,priority=priority,strong=strong))
    proposals.sort(key=lambda item:(not item['strong'],-item['priority']))
    return [dict(id=f'G{i+1:03d}',**row) for i,row in enumerate(proposals)]


def terminal_zones(poles, paths):
    import numpy as np

    if not paths:
        return []
    ends=np.array([p['points'][i] for p in paths for i in (0,-1)])
    output=[]
    for p in poles:
        x0,y0,x1,y1=p['box']
        top=y0+(y1-y0)*.58
        candidates=ends[(ends[:,0]>=y0-22)&(ends[:,0]<=top+14)&
                        (ends[:,1]>=x0-20)&(ends[:,1]<=x1+20)]
        if len(candidates):
            output.append(dict(p,contacts=np.unique(candidates,axis=0).tolist()))
        else:
            output.append(dict(p,contacts=[]))
    return output


def span_policy(shape,paths,links,poles,reviews=None):
    import numpy as np

    line=np.zeros(shape,dtype=bool)
    for path in paths:
        line[tuple(path['points'].T)]=True
    reviews=reviews or {}
    pole_states={p['id']:reviews.get(p['id'])=='YES' or
                 (p['score']>=.32 and reviews.get(p['id'])!='NO') for p in poles}
    points=np.asarray([xy for p in poles if pole_states[p['id']] for xy in p['contacts']])
    taken=set()
    joined=[]
    stopped=[]
    for link in links:
        choice=reviews.get(link['id'])
        if choice=='NO' or (not link['strong'] and choice!='YES'):
            continue
        endpoints={(link['a'],link['side_a']),(link['b'],link['side_b'])}
        if endpoints & taken:
            continue
        xy=np.asarray(link['pixels'])
        if len(points) and np.any(np.linalg.norm(xy[:,None]-points[None,:],axis=2)<12):
            stopped.append(link['id'])
            continue
        taken.update(endpoints)
        line[tuple(xy.T)]=True
        joined.append(link['id'])
    return line,dict(active_poles=[p['id'] for p in poles if pole_states[p['id']]],
                     with_contact=[p['id'] for p in poles if pole_states[p['id']] and p['contacts']],
                     bridges=joined,blocked_at_pole=stopped)


def board(image, center, target, kind, output, mark=True):
    from PIL import Image, ImageDraw, ImageFont

    w,h=image.size
    cy,cx=center
    crop_size=420 if kind=='pole' else 240
    if kind=='pole' and target is not None:
        a,b,c,d=target
        crop_size=min(h,w,max(420,round(1.18*max(c-a,d-b))))
    x0=max(0,min(round(cx)-crop_size//2,w-crop_size))
    y0=max(0,min(round(cy)-crop_size//2,h-crop_size))
    crop=image.crop((x0,y0,x0+crop_size,y0+crop_size))
    marked=crop.copy()
    draw=ImageDraw.Draw(marked)
    if mark and target is not None and kind=='pole':
        a,b,c,d=target
        draw.rectangle((a-x0,b-y0,c-x0,d-y0),outline=(0,247,237),width=3)
    elif mark and target is not None:
        for y,x in target:
            x,y=x-x0,y-y0
            draw.ellipse((x-5,y-5,x+5,y+5),outline=(0,247,237),width=2)
    size=320
    plate=Image.new('RGB',(size*2,size+26),(18,22,28))
    plate.paste(crop.resize((size,size),Image.Resampling.LANCZOS),(0,26))
    plate.paste(marked.resize((size,size),Image.Resampling.LANCZOS),(size,26))
    draw=ImageDraw.Draw(plate)
    font=ImageFont.load_default(size=15)
    draw.text((7,4),'ORIGINAL',fill='white',font=font)
    draw.text((size+7,4),'MARKED TARGET',fill='white',font=font)
    plate.save(output,quality=95)


def prepare():
    import cv2
    import numpy as np
    from PIL import Image
    from evaluate_first_image_v1 import readmask

    inputs=OUT/'review_inputs'
    inputs.mkdir(exist_ok=False)
    poles=read_json(OUT/'poles.json')['poles']
    manifest=[]
    for stem in STEMS:
        image=Image.open(source(stem)).convert('RGB')
        lab=cv2.cvtColor(np.asarray(image,dtype=np.float32)/255,cv2.COLOR_RGB2LAB)
        base=readmask(baseline_path(stem))
        paths=fragments(stem,lab)
        links=gap_candidates(paths,lab,base)
        zones=terminal_zones(poles[stem],paths)
        output=OUT/stem
        output.mkdir()
        save(output/'paths.json',[dict(p,points=p['points'].tolist()) for p in paths])
        save(output/'links.json',links)
        save(output/'poles_with_contacts.json',zones)
        items=[]
        for p in zones:
            a,b,c,d=p['box']
            items.append((p['id'],'pole',((b+d)/2,(a+c)/2),p['box']))
        reviewed=[]
        # Sample strong and borderline geometrically admissible joins with spatial diversity.
        for strong in (True,False):
            chosen=0
            for link in links:
                if link['strong']!=strong:
                    continue
                center=np.mean(link['ends'],axis=0)
                if any(np.linalg.norm(center-earlier)<85 for earlier in reviewed):
                    continue
                reviewed.append(center)
                items.append((link['id'],'gap',center,link['ends']))
                chosen+=1
                if chosen==6:
                    break
        for kind in ('pole','gap'):
            picked=next((x for x in items if x[1]==kind),None)
            center=picked[2] if picked else (image.height/2,image.width/2)
            items.append(('CTRL_'+kind,kind,center,None))
        for ident,kind,center,target in items:
            dest=inputs/f'{stem}_{ident}.jpg'
            board(image,center,target,kind,dest)
            manifest.append(dict(stem=stem,id=ident,kind=kind,path=str(dest)))
        print(stem,len(paths),'fragments',len(links),'eligible gaps',len(zones),'pole candidates',
              len(items),'review items',flush=True)
    save(OUT/'review_manifest.json',manifest)

PROMPTS={
    'pole':('The two panels show the same place. Inspect only the object enclosed by the cyan rectangle '
            'in the right panel. Use the left panel to see the original. Answer NO if there is no cyan '
            'rectangle or it encloses a fence post, lamp or building. Answer YES only if it identifies an '
            'overhead electrical utility pole. If ambiguous answer UNSURE. Reply with one word.'),
    'gap':('The two panels show the same place. In the right panel the two cyan rings mark ends of two '
           'short candidate fragments. In the unmarked left panel, check whether the evidence supports '
           'joining exactly those two endpoints along the same overhead electrical wire. Answer YES only '
           'if supported. Answer NO if there are no two rings or they are different wires or background. '
           'Answer UNSURE if the connection cannot be judged. Reply with one word.'),
}


def refresh_pole_boards():
    from PIL import Image

    for stem in STEMS:
        image=Image.open(source(stem)).convert('RGB')
        poles=read_json(OUT/stem/'poles_with_contacts.json')
        for pole in poles:
            x0,y0,x1,y1=pole['box']
            center=((y0+y1)/2,(x0+x1)/2)
            board(image,center,pole['box'],'pole',OUT/'review_inputs'/f'{stem}_{pole["id"]}.jpg')
        if poles:
            x0,y0,x1,y1=poles[0]['box']
            center=((y0+y1)/2,(x0+x1)/2)
            board(image,center,poles[0]['box'],'pole',OUT/'review_inputs'/f'{stem}_CTRL_pole.jpg',mark=False)


def review(model):
    from vllm import LLM, SamplingParams

    target=OUT/f'{model}_responses.json'
    if target.exists():
        raise FileExistsError(target)
    manifest=read_json(OUT/'review_manifest.json')
    settings=dict(model=MODELS[model], dtype='float16', gpu_memory_utilization=.69 if model=='7b' else .65,
                  max_model_len=2048,max_num_batched_tokens=2048,max_num_seqs=2,enforce_eager=True,
                  limit_mm_per_prompt={'image':1,'video':0},seed=0,
                  mm_processor_kwargs={'min_pixels':784,'max_pixels':250880})
    if model=='7b':
        settings['quantization']='awq'
        settings['model']=os.environ.get('WIRE_SPAN_7B_LOCAL',settings['model'])
        settings['revision']='536a35794df8831aa814970ee8f89eff577e7718'
        import torch
        free,total=torch.cuda.mem_get_info()
        if free/total < .72:
            settings.update(gpu_memory_utilization=min(.69,(free-512*1024**2)/total),
                            cpu_offload_gb=4,max_num_seqs=1)

    else:
        settings['revision']='66285546d2b821cf421d4f5eb2576359d3770cd3'
    sampling=dict(temperature=0,max_tokens=14)
    save(OUT/f'{model}_config.json',dict(settings=settings,sampling=sampling,prompts=PROMPTS))
    llm=LLM(**settings)
    options=SamplingParams(**sampling)
    responses=[]
    for start in range(0,len(manifest),4):
        batch=manifest[start:start+4]
        messages=[]
        for item in batch:
            uri='data:image/jpeg;base64,'+base64.b64encode(Path(item['path']).read_bytes()).decode('ascii')
            messages.append([{'role':'user','content':[{'type':'image_url','image_url':{'url':uri}},
                                                       {'type':'text','text':PROMPTS[item['kind']]}]}])
        for item,result in zip(batch,llm.chat(messages,options,use_tqdm=False),strict=True):
            raw=result.outputs[0].text
            label=raw.strip().strip('.*!').upper()
            responses.append(dict(**item,raw=raw,label=label if label in ('YES','NO','UNSURE') else None))
        print(model,len(responses),'/',len(manifest),flush=True)
    save(target,responses)

def annotation_paths(stem, shape):
    import xml.etree.ElementTree as ET
    import numpy as np

    xml=ET.parse('data/initial_8/annotations.xml')
    matches=[row for row in xml.getroot().findall('image') if Path(row.attrib['name']).stem==stem]
    assert len(matches)==1,stem
    paths=[]
    for item in matches[0].findall('polyline'):
        if item.attrib['label'].casefold() not in ('powerline','polyline'):
            continue
        points=np.asarray([[float(v) for v in pair.split(',')] for pair in item.attrib['points'].split(';')])
        steps=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        t=np.linspace(0,steps[-1],int(np.ceil(steps[-1]))+1)
        x=np.rint(np.interp(t,steps,points[:,0])).astype(int).clip(0,shape[1]-1)
        y=np.rint(np.interp(t,steps,points[:,1])).astype(int).clip(0,shape[0]-1)
        if len(x)>1:
            paths.append(np.column_stack((y,x)))
    assert paths,stem
    return paths


def continuity(distance, paths):
    import numpy as np

    runs=[]
    for path in paths:
        near=distance[tuple(path.T)]<=5
        switches=np.flatnonzero(np.diff(np.r_[False,near,False]))
        runs.append(float(np.max(switches[1::2]-switches[::2],initial=0)/len(near)))
    return dict(xml_fragments=len(runs),longest_run_mean=float(np.mean(runs)),
                fragments_with_80pct_run=sum(v>=.8 for v in runs))


def review_choices(stem, responses):
    from collections import Counter

    labels={x['id']:x['label'] for x in responses if x['stem']==stem}
    assert len(labels)==sum(x['stem']==stem for x in responses)
    passed={kind:labels.get('CTRL_'+kind)=='NO' for kind in ('pole','gap')}
    choices={item['id']:labels[item['id']] for item in responses if item['stem']==stem
             and not item['id'].startswith('CTRL_') and passed[item['kind']]}
    return choices,dict(controls={k:labels.get('CTRL_'+k) for k in passed},passed=passed,
                        usable_choices=len(choices),labels=dict(Counter(str(v) for v in labels.values())))


def views(folder, photo, variants, selected, poles, policy):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    def layer(line,width):
        pixels=line.astype('uint8')
        paint=photo.copy()
        halo=cv2.dilate(pixels,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width+4,width+4)))>0
        visible=cv2.dilate(pixels,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width)))>0
        paint[halo]=(paint[halo]*.38).astype(np.uint8)
        paint[visible]=COLORS[0]
        return Image.fromarray(paint)
    labels={'baseline':'Initial LangSAM','rules':'Continuity rules','3b':'Rules + Qwen 3B','7b':'Rules + Qwen 7B AWQ'}
    layer(variants[selected],5).save(folder/'prediction_on_original.jpg',quality=93)
    layer(variants['rules'],5).save(folder/'rules_on_original.jpg',quality=93)
    for mode in ('3b','7b'):
        if mode in variants and np.any(variants[mode]^variants['rules']):
            layer(variants[mode],5).save(folder/f'{mode}_on_original.jpg',quality=93)
    pole_image=layer(variants[selected],5)
    draw=ImageDraw.Draw(pole_image)
    font=ImageFont.load_default(size=32)
    active=set(policy.get('active_poles',[]))
    for item in poles:
        x0,y0,x1,y1=item['box']
        color=COLORS[1] if item['id'] in active else COLORS[2]
        draw.rectangle((x0,y0,x1,y1),outline=color,width=6)
        draw.text((x0,max(0,y0-38)),item['id'],fill=color,font=font,stroke_width=2,stroke_fill='black')
        if item['id'] in active:
            for y,x in item['contacts']:
                draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=4)
    pole_image.resize((1600,900),Image.Resampling.LANCZOS).save(folder/'pole_candidate_map.jpg',quality=94)
    width,height,header=760,428,30
    chart=Image.new('RGB',(width*2,(height+header)*2),(18,22,29))
    draw=ImageDraw.Draw(chart)
    font=ImageFont.load_default(size=18)
    for index,name in enumerate(labels):
        x,y=index%2*width,index//2*(height+header)
        draw.text((x+8,y+4),labels[name] if name in variants else labels[name]+' (pending)',fill='white',font=font)
        if name in variants:
            chart.paste(layer(variants[name],7).resize((width,height),Image.Resampling.LANCZOS),(x,y+header))
    chart.save(folder/'comparison.jpg',quality=93)


def evaluate(partial=False):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    from evaluate_first_image_v1 import metrics,readmask

    target=OUT/('review_partial' if partial else 'review')
    archive_path=BASE/(NAME+('_partial' if partial else '')+'.zip')
    if target.exists() or archive_path.exists():
        raise FileExistsError(target)
    responses={mode:read_json(OUT/f'{mode}_responses.json') for mode in MODELS
               if (OUT/f'{mode}_responses.json').exists()}
    assert '3b' in responses and (partial or '7b' in responses)
    target.mkdir()
    predictions={}
    audits={}
    for stem in STEMS:
        folder=OUT/stem
        base=readmask(baseline_path(stem))
        paths=read_json(folder/'paths.json')
        for path in paths:
            path['points']=np.asarray(path['points'])
        links=read_json(folder/'links.json')
        poles=read_json(folder/'poles_with_contacts.json')
        lines=dict(baseline=base)
        audit=dict(link_candidates=len(links),fragments=len(paths),pole_candidates=len(poles))
        lines['rules'],audit['rules']=span_policy(base.shape,paths,links,poles)
        for mode, items in responses.items():
            choices,gate=review_choices(stem,items)
            lines[mode],policy=span_policy(base.shape,paths,links,poles,choices)
            audit[mode]=dict(**policy,review=gate,
                             changed_pixels_vs_rules=int(np.count_nonzero(lines[mode]^lines['rules'])))
        predictions[stem]=(lines,poles,links)
        audits[stem]=audit
    # Read annotations only once every prediction has been frozen.
    rows=[]
    for stem,(variants,poles,links) in predictions.items():
        folder=target/stem
        folder.mkdir()
        base=variants['baseline']
        ref_folder=(BASE/'diagnostic_eval'/f'{stem}_v1' if stem==STEMS[0]
                    else BASE/'dev_transfer_20261107_v1'/stem/'evaluation')
        reference=readmask(ref_folder/'reference_centerlines.png')
        ref_distance=distance_transform_edt(~reference)
        xml=annotation_paths(stem,reference.shape)
        trees=[cKDTree(path) for path in xml]
        by_id={item['id']:item for item in links}
        for mode,line in variants.items():
            scores,distance=metrics(line,reference,ref_distance)
            score=next(row for row in scores if row['tolerance_pixels']==5)
            if distance is None:
                distance=np.full(reference.shape,np.inf)
            joins=audits[stem].get(mode,{}).get('bridges',[])
            unsupported=0
            for ident in joins:
                pixels=np.asarray(by_id[ident]['pixels'])
                supported=False
                for tree in trees:
                    d=tree.query(pixels)[0]
                    if d[0]<=5 and d[-1]<=5 and np.mean(d<=5)>=.70:
                        supported=True
                        break
                unsupported+=int(not supported)
            rows.append(dict(stem=stem,method=mode,**score,**continuity(distance,xml),
                             bridges=len(joins),xml_unsupported_bridges=unsupported,
                             off_reference_pixels=int((line&(ref_distance>5)).sum())))
            Image.fromarray(line.astype(np.uint8)*255).save(folder/f'{mode}_centerline.png')
    keys=('centerline_precision','centerline_recall','centerline_f1','longest_run_mean')
    macro={mode:{k:float(np.mean([r[k] for r in rows if r['method']==mode])) for k in keys}
           for mode in predictions[STEMS[0]][0]}
    best=max(macro,key=lambda k:macro[k]['centerline_f1'])
    for stem,(lines,poles,_) in predictions.items():
        photo=np.asarray(Image.open(source(stem)).convert('RGB'))
        views(target/stem,photo,lines,best,poles,audits[stem].get(best,audits[stem]['rules']))
    with (target/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(target/'summary.json',dict(macro=macro,audit=audits,selected=best,
                                  tolerance_px=5,stage='first LangSAM candidates',partial=partial,
                                  rule=dict(max_gap=GAP,max_lateral=LATERAL,max_turn=ANGLE,min_fragment=MIN_LENGTH)))
    (target/'README.txt').write_text(
        'Development only. Same frozen first LangSAM masks. Zoom addition is outside this test.\n'
        'Fragments follow observed centerline pixels; short bridges are bounded by direction and offset.\n'
        'Color can add at most 0.06 to ranking and does not change eligibility.\n'
        'Pole contact zones use nearby fragment endpoints near the upper part of a pole candidate box.\n'
        'These zones block proposed bridges but preserve already observed line pixels.\n'
        'Each VLM can accept borderline bridges, veto reviewed ones or change pole state only if the\n'
        'corresponding unmarked control passes on the same image. Unreviewed cases use the rules.\n'
        'XML bridge support requires one annotated polyline near both ends and at least 70 percent of the bridge.\n'
        'An unsupported bridge is not definitive evidence of a physical misconnection. No pole ground truth.\n'
        'Original overlay selects the best mean F1 on the same development set.\n',encoding='utf-8')
    with ZipFile(archive_path,'x',ZIP_DEFLATED) as archive:
        for path in sorted(target.rglob('*')):
            if path.is_file():
                archive.write(path,Path(NAME)/path.relative_to(target))
        for mode in responses:
            for ending in ('responses','config'):
                path=OUT/f'{mode}_{ending}.json'
                archive.write(path,Path(NAME)/path.name)
        for leaf in ('poles.json','review_manifest.json'):
            archive.write(OUT/leaf,Path(NAME)/leaf)
        for path in sorted((OUT/'review_inputs').glob('*.jpg')):
            archive.write(path,Path(NAME)/'review_inputs'/path.name)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    for mode,scores in macro.items():
        print(mode,' '.join(f'{100*x:.2f}' for x in scores.values()))
    for stem,audit in audits.items():
        print(stem,{k:(len(v['bridges']),len(v['with_contact']),v.get('changed_pixels_vs_rules'),
                       v.get('review',{}).get('controls')) for k,v in audit.items() if isinstance(v,dict)},flush=True)
    print('VERIFIED',archive_path,'SELECTED',best,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','refresh','3b','7b','evaluate'))
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    if args.stage=='evaluate':
        evaluate(args.partial)
    elif args.stage in MODELS:
        review(args.stage)
    else:
        {'prepare':prepare,'refresh':refresh_pole_boards}[args.stage]()
