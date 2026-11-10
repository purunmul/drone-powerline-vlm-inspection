import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

import run_wire_span_continuity_v1 as span

OUT = Path('results/wire_pole_contacts_20261108_v1')
PROMPT = ('Both panels show the same crop. The right panel marks the last part of candidate wire 1 '
          'in cyan and shows lettered candidate connection points. Decide where the marked end of '
          'wire 1 visibly attaches to an electrical utility pole. Reply with exactly one of A, B, '
          'NONE, UNSURE. Use NONE when no lettered point exists or both points are visibly wrong. '
          'Use UNSURE if there is a possible point but the attachment is ambiguous. '
          'A nearby pole by itself is insufficient. Check the unmarked original on the left.')


def get_paths(folder):
    paths = span.read_json(folder/'paths.json')
    for path in paths:
        path['points'] = np.asarray(path['points'])
    return paths


def restore_fragments(base, paths, lab):
    samples = []
    directions = []
    for item in paths:
        p = item['points']
        for i in range(0,len(p),4):
            samples.append(p[i])
            directions.append(span.unit(p[min(i+5,len(p)-1)]-p[max(i-5,0)]))
    if not samples:
        return []
    tree = cKDTree(samples)
    directions = np.asarray(directions)
    core = np.zeros_like(base)
    for path in paths:
        core[tuple(path['points'].T)] = True
    restored = []
    for points in span.trace_without_corner_edges(base):
        if len(points)<9 or np.mean(core[tuple(points.T)])>.7:
            continue
        chord = points[-1]-points[0]
        length = np.linalg.norm(chord)
        arc = np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
        if length<8 or arc>1.30*length:
            continue
        direction = span.unit(chord)
        ridge = span.line_evidence(lab,points,direction)
        near = [i for edge in (points[0],points[-1]) for i in tree.query_ball_point(edge,8)]
        aligned = bool(near) and bool(np.any(np.abs(directions[near]@direction)>=np.cos(np.deg2rad(20))))
        if (aligned and ridge>=.42) or (length>=30 and ridge>=.65):
            restored.append(dict(points=points,arc=float(arc),ridge=ridge,narrow=None))
    return restored


def make_board(photo, trail, choices, dest):
    height,width=photo.shape[:2]
    ey,ex=trail[-1]
    size=420
    x0=int(np.clip(ex-size//2,0,max(0,width-size)))
    y0=int(np.clip(ey-size//2,0,max(0,height-size)))
    img=Image.fromarray(photo[y0:y0+size,x0:x0+size])
    right=img.copy()
    draw=ImageDraw.Draw(right)
    draw.line([(int(x-x0),int(y-y0)) for y,x in trail],fill=(0,245,237),width=2)
    draw.text((int(ex-x0+6),int(ey-y0+3)),'1',fill='white',stroke_width=2,stroke_fill='black')
    for letter, data in choices.items():
        y,x=data['yx']
        x,y=x-x0,y-y0
        draw.ellipse((x-6,y-6,x+6,y+6),outline=(255,217,60),width=2)
        draw.text((x+8,y-12),letter,fill=(255,217,60),stroke_width=2,stroke_fill='black')
    board=Image.new('RGB',(640,344),(20,24,30))
    for i,panel in enumerate((img,right)):
        board.paste(panel.resize((320,320),Image.Resampling.LANCZOS),(320*i,24))
    draw=ImageDraw.Draw(board)
    draw.text((6,4),'ORIGINAL',fill='white')
    draw.text((326,4),'WIRE 1 AND OPTIONS',fill='white')
    board.save(dest,quality=94)


def review_cases(stem, photo, paths, poles, base, *, output_dir=None, max_cases=4):
    output_dir=Path(output_dir) if output_dir is not None else OUT/'review_inputs'
    candidates = []
    for index,item in enumerate(paths):
        for side,points in enumerate((item['points'],item['points'][::-1])):
            y,x=points[-1]
            for pole in poles:
                x0,y0,x1,y1=pole['box']
                if pole['score']<.27 or not (y0-14<=y<=y0+.60*(y1-y0) and x0-10<=x<=x1+10):
                    continue
                center=(x0+x1)/2
                if abs(x-center)>max(22,.30*(x1-x0)):
                    continue
                strength=pole['score']+.25*item['ridge']+.10*min(item['arc']/100,1)
                candidates.append((strength,index,side,points[-48:],pole))
    if not candidates:
        return []
    distance=distance_transform_edt(~base)
    used=[]
    entries=[]
    for _,index,side,trail,pole in sorted(candidates,key=lambda row:-row[0]):
        if any(np.linalg.norm(trail[-1]-p)<80 for p in used):
            continue
        direction=span.unit(trail[-1]-trail[0])
        perpendicular=np.array([-direction[1],direction[0]])
        offsets=[trail[-1]+np.rint(perpendicular*step).astype(int) for step in (52,-52,74,-74,96,-96)]
        wrong=[p.tolist() for p in offsets if np.all(p>=0) and np.all(p<base.shape) and distance[tuple(p)]>10]
        if len(wrong)<2:
            continue
        used.append(trail[-1])
        i=len(used)
        target='A' if i%2 else 'B'
        other='B' if target=='A' else 'A'
        eligible={target:dict(yx=trail[-1].tolist(),pole=pole['id'],path=index,side=side),
                  other:dict(yx=wrong[0])}
        cases=[(f'C{i:02d}',eligible)]
        if i==1:
            cases.extend([('CTRL_EMPTY',{}),('CTRL_WRONG',{'A':dict(yx=wrong[0]),'B':dict(yx=wrong[1])})])
        for ident,options in cases:
            dest=output_dir/f'{stem}_{ident}.jpg'
            make_board(photo,trail,options,dest)
            entries.append(dict(stem=stem,id=ident,options=options,image=str(dest)))
        if i==max_cases:
            break
    return entries


def prepare():
    from evaluate_first_image_v1 import readmask
    OUT.mkdir(exist_ok=False)
    (OUT/'review_inputs').mkdir()
    entries=[]
    for stem in span.STEMS:
        prior=span.OUT/stem
        folder=OUT/stem
        folder.mkdir()
        photo=np.asarray(Image.open(span.source(stem)).convert('RGB'))
        lab=cv2.cvtColor(photo.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        base=readmask(span.baseline_path(stem))
        paths=get_paths(prior)
        poles=span.read_json(prior/'poles_with_contacts.json')
        restored=restore_fragments(base,paths,lab)
        paths.extend(restored)
        links=span.gap_candidates(paths,lab,base)
        span.save(folder/'paths.json',[dict(p,points=p['points'].tolist()) for p in paths])
        span.save(folder/'links.json',links)
        span.save(folder/'poles.json',poles)
        new=review_cases(stem,photo,paths,poles,base)
        entries.extend(new)
        print(stem,'restored',len(restored),'links',len(links),'review cases',len(new),flush=True)
    span.save(OUT/'manifest.json',entries)


def review():
    from vllm import LLM,SamplingParams

    manifest=span.read_json(OUT/'manifest.json')
    output=OUT/'7b_responses.json'
    if output.exists():
        raise FileExistsError(output)
    previous=span.read_json(span.OUT/'7b_config.json')['settings']
    previous.update(max_num_seqs=1)
    span.save(OUT/'7b_config.json',dict(settings=previous,prompt=PROMPT))
    llm=LLM(**previous)
    params=SamplingParams(temperature=0,max_tokens=12)
    rows=[]
    for start in range(0,len(manifest),4):
        batch=manifest[start:start+4]
        messages=[]
        for item in batch:
            uri='data:image/jpeg;base64,'+base64.b64encode(Path(item['image']).read_bytes()).decode('ascii')
            messages.append([dict(role='user',content=[dict(type='image_url',image_url=dict(url=uri)),
                                                       dict(type='text',text=PROMPT)])])
        for item,answer in zip(batch,llm.chat(messages,params,use_tqdm=False),strict=True):
            raw=answer.outputs[0].text
            token=raw.strip().strip('.*!').upper()
            rows.append(dict(stem=item['stem'],id=item['id'],raw=raw,
                             choice=token if token in ('A','B','NONE','UNSURE') else None))
    assert len(rows)==len(manifest)
    tmp=output.with_suffix('.tmp')
    span.save(tmp,rows)
    tmp.replace(output)
    print('REVIEW_SAVED',len(rows),flush=True)
    llm.llm_engine.engine_core.shutdown()

def read_review(stem, manifest, answers):
    cases={row['id']:row for row in manifest if row['stem']==stem}
    choices={row['id']:row['choice'] for row in answers if row['stem']==stem}
    assert cases.keys()==choices.keys(),stem
    passed=all(choices.get(k)=='NONE' for k in ('CTRL_EMPTY','CTRL_WRONG'))
    contacts=[]
    if passed:
        for ident,case in cases.items():
            if ident.startswith('CTRL_'):
                continue
            selected=case['options'].get(choices[ident],{})
            if 'pole' in selected:
                contacts.append(dict(case=ident,**selected))
    return contacts,dict(passed=passed,blank=choices.get('CTRL_EMPTY'),wrong=choices.get('CTRL_WRONG'),
                         cases=sum(not ident.startswith('CTRL_') for ident in cases),confirmed=len(contacts))


def predictions(stem,manifest,answers):
    from evaluate_first_image_v1 import readmask
    prior=span.OUT/stem
    folder=OUT/stem
    baseline=readmask(span.baseline_path(stem))
    previous=readmask(span.OUT/'review'/stem/'rules_centerline.png')
    original_links=span.read_json(prior/'links.json')
    old_paths=get_paths(prior)
    poles=span.read_json(folder/'poles.json')
    reproduction,_=span.span_policy(baseline.shape,old_paths,original_links,poles)
    assert np.array_equal(reproduction,previous),stem
    paths=get_paths(folder)
    links=span.read_json(folder/'links.json')
    preserved,audit=span.span_policy(baseline.shape,paths,links,poles)
    contacts,gate=read_review(stem,manifest,answers)
    observed=np.zeros_like(baseline)
    for p in paths:
        observed[tuple(p['points'].T)]=True
    assert not (observed&~baseline).any()
    with_contacts=observed.copy()
    blocked=[]
    by_id={link['id']:link for link in links}
    for ident in audit['bridges']:
        link=by_id[ident]
        pixels=np.asarray(link['pixels'])
        ends={(link['a'],link['side_a']),(link['b'],link['side_b'])}
        if any((p['path'],p['side']) in ends or np.min(np.linalg.norm(pixels-p['yx'],axis=1))<=8 for p in contacts):
            blocked.append(ident)
        else:
            with_contacts[tuple(pixels.T)]=True
    assert not (with_contacts&~preserved).any()
    if not blocked:
        assert np.array_equal(with_contacts,preserved)
    return dict(initial_langsam=baseline,previous_rules=previous,preserve_fragments=preserved,
                preserve_plus_7b_contacts=with_contacts),dict(gate,restored_fragments=len(paths)-len(old_paths),
                bridges=len(audit['bridges']),blocked_by_7b=blocked,contacts=contacts,
                changed_pixels=int(np.count_nonzero(preserved^with_contacts)))


def render(folder,stem,variants):
    from PIL import ImageFont

    photo=np.asarray(Image.open(span.source(stem)).convert('RGB'))
    width,height,header=760,428,30
    board=Image.new('RGB',(width*2,(height+header)*2),(18,22,29))
    draw=ImageDraw.Draw(board)
    font=ImageFont.load_default(size=18)
    titles=('Initial LangSAM','Previous continuity rules','Preserve observed fragments','Preserve + 7B pole contacts')
    for index,((name,mask),title) in enumerate(zip(variants.items(),titles,strict=True)):
        arr=mask.astype(np.uint8)
        Image.fromarray(arr*255).save(folder/f'{name}_centerline.png')
        image=photo.copy()
        image[cv2.dilate(arr,np.ones((5,5),np.uint8))>0]=(3,245,226)
        overlay=Image.fromarray(image)
        if index>=2:
            overlay.save(folder/f'{name}_on_original.jpg',quality=93)
        x,y=index%2*width,index//2*(height+header)
        draw.text((x+8,y+4),title,font=font,fill='white')
        board.paste(overlay.resize((width,height),Image.Resampling.LANCZOS),(x,y+header))
    board.save(folder/'comparison.jpg',quality=93)


def evaluate():
    from evaluate_first_image_v1 import metrics, readmask
    manifest=span.read_json(OUT/'manifest.json')
    answers=span.read_json(OUT/'7b_responses.json')
    assert [(x['stem'],x['id']) for x in manifest]==[(x['stem'],x['id']) for x in answers]
    frozen={stem:predictions(stem,manifest,answers) for stem in span.STEMS}
    destination=OUT/'review'
    destination.mkdir(exist_ok=False)
    results=[]
    for stem,(variants,audit) in frozen.items():
        folder=destination/stem
        folder.mkdir()
        render(folder,stem,variants)
        reference=(span.BASE/'diagnostic_eval'/f'{stem}_v1' if stem==span.STEMS[0]
                   else span.BASE/'dev_transfer_20261107_v1'/stem/'evaluation')
        ref=readmask(reference/'reference_centerlines.png')
        d=distance_transform_edt(~ref)
        xml=span.annotation_paths(stem,ref.shape)
        for name,line in variants.items():
            metrics_by_tol,distance=metrics(line,ref,d)
            score=next(x for x in metrics_by_tol if x['tolerance_pixels']==5)
            if distance is None:
                distance=np.full(ref.shape,np.inf)
            results.append(dict(stem=stem,method=name,**score,**span.continuity(distance,xml)))
    import csv
    with (destination/'metrics.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    keys=('centerline_precision','centerline_recall','centerline_f1','longest_run_mean')
    macro={name:{key:float(np.mean([row[key] for row in results if row['method']==name])) for key in keys}
           for name in frozen[span.STEMS[0]][0]}
    span.save(destination/'summary.json',dict(macro=macro,audit={stem:audit for stem,(_,audit) in frozen.items()},
              tolerance_pixels=5,development_only=True,color_used_for_contacts=False,width_added=False))
    for name,scores in macro.items():
        print(name,*(f'{100*value:.2f}' for value in scores.values()),flush=True)
    for stem,(_,a) in frozen.items():
        print(stem,'restored',a['restored_fragments'],'controls',a['blank'],a['wrong'],
              'cases',a['cases'],'confirmed',a['confirmed'],'blocked',len(a['blocked_by_7b']),
              'changed_pixels',a['changed_pixels'],flush=True)
    print('VERIFIED',destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','review','evaluate'))
    {'prepare':prepare,'review':review,'evaluate':evaluate}[parser.parse_args().stage]()
