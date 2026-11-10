from pathlib import Path
import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

import run_wire_four_rules_v1 as four
import run_wire_contacts_v1 as prior
import run_wire_span_continuity_v1 as span
from evaluate_first_image_v1 import metrics, readmask

OUT = Path('results/wire_long_trace_20261109_v2')
MIN_LENGTH = 160
MIN_TERMINAL_LENGTH = 80
MAX_EXTENSION = 192
METHODS=('previous','trace_only','trace_min160','trace_min160_poles')


def at_pole(yx, direction, contacts):
    for i,row in enumerate(contacts):
        delta=np.asarray(row['yx'])-yx
        if np.linalg.norm(delta)<=8 and delta@direction>=-3:
            return i
    return None


def walk_end(points, lab, maskdist, contacts, *, max_extension=MAX_EXTENSION):
    if len(points)<18:
        return [],None
    start=points[-1].astype(float)
    direction=span.unit(points[-1]-points[-min(len(points),12)])
    contact=at_pole(start,direction,contacts)
    if contact is not None:
        return [],contact
    initial=direction.copy()
    current=start.copy()
    novel=0
    traveled=0.0
    extension=[]
    height,width=lab.shape
    for _ in range(max_extension//3):
        stop=at_pole(current,direction,contacts)
        if stop is not None:
            return extension,stop
        normal=np.array([-direction[1],direction[0]])
        candidates=[]
        for lateral in (0,-1,1,-2,2):
            proposed=np.rint(current+3*direction+lateral*normal).astype(int)
            y,x=proposed
            if y<5 or y>=height-5 or x<5 or x>=width-5:
                continue
            step=span.unit(proposed-current)
            if step@direction<.87 or step@initial<.87:
                continue
            side=np.rint(4*normal).astype(int)
            a=lab[y+side[0],x+side[1]]-lab[y,x]
            b=lab[y-side[0],x-side[1]]-lab[y,x]
            contrast=min(abs(a),abs(b)) if a*b>0 else 0
            near=maskdist[y,x]<=3
            if contrast<(2.8 if near else 5.5):
                continue
            if not near and novel>=8:
                continue
            score=min(contrast,20)+3*near-.85*abs(lateral)
            candidates.append((score,proposed,near,step))
        if not candidates:
            break
        _,proposed,near,step=max(candidates,key=lambda row:row[0])
        traveled+=float(np.linalg.norm(proposed-current))
        if traveled>max_extension:
            break
        extension.append(proposed)
        direction=span.unit(.80*direction+.20*step)
        current=proposed
        novel=0 if near else novel+1
    return extension,None


def chain_paths(paths, links, selected, contacts):
    ends=[p['points'][side] for p in paths for side in (0,-1)]
    if not ends:
        return []
    headings=[]
    for row in paths:
        p=row['points']
        offset=min(9,len(p)-1)
        headings.extend((span.unit(p[0]-p[offset]),span.unit(p[-1]-p[-offset-1])))
    graph={}
    blocked={(2*p['path']+p['side']) for p in contacts}
    for item in links:
        if item['id'] not in selected:
            continue
        a,b=2*item['a']+item['side_a'],2*item['b']+item['side_b']
        if a in blocked or b in blocked:
            continue
        graph[a]=b
        graph[b]=a
    tree=cKDTree(ends)
    touching=[]
    for a,b in tree.query_pairs(3.1):
        if a//2==b//2 or a in blocked or b in blocked:
            continue
        alignment=-float(headings[a]@headings[b])
        delta=np.array(ends[b])-ends[a]
        distance=np.linalg.norm(delta)
        if alignment<.88 or (distance>0 and (headings[a]@delta<-.1 or headings[b]@delta>.1)):
            continue
        touching.append((float(distance)-alignment,a,b))
    for _,a,b in sorted(touching):
        if a in graph or b in graph:
            continue
        graph[a]=b
        graph[b]=a
    visited=set()
    result=[]
    for start in sorted(range(len(ends)),key=lambda i:(i in graph,i)):
        edge=start
        chain=[]
        while edge//2 not in visited:
            index,side=divmod(edge,2)
            p=paths[index]['points']
            chain.extend(p if side==0 else p[::-1])
            visited.add(index)
            if (edge^1) not in graph:
                break
            edge=graph[edge^1]
        if chain:
            result.append(np.asarray(chain))
    return result


def render_paths(chains, lab, maskdist, contacts, *, min_length=MIN_LENGTH,
                 min_terminal_length=MIN_TERMINAL_LENGTH, max_extension=MAX_EXTENSION):
    shape=lab.shape
    raw=np.zeros(shape,np.uint8)
    filtered=np.zeros(shape,np.uint8)
    short=0
    stops=[]
    retained=0
    expanded=0
    lengths=[]
    for path in chains:
        if len(path)<18:
            continue
        head,ph=walk_end(path[::-1],lab,maskdist,contacts,max_extension=max_extension)
        tail,pt=walk_end(path,lab,maskdist,contacts,max_extension=max_extension)
        parts=[np.asarray(head)[::-1],path,np.asarray(tail)]
        points=np.concatenate([part for part in parts if len(part)])
        expanded+=len(head)+len(tail)
        arc=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        chord=float(np.linalg.norm(points[-1]-points[0]))
        if arc>1.40*max(1,chord):
            continue
        edge=any(min(y,x,shape[0]-1-y,shape[1]-1-x)<=6 for y,x in (points[0],points[-1]))
        anchored=ph is not None or pt is not None or edge
        cv2.polylines(raw,[points[:,::-1].astype(np.int32)],False,1,1)
        if arc<min_length and not (anchored and arc>=min_terminal_length):
            short+=1
            continue
        retained+=1
        lengths.append(round(arc,1))
        cv2.polylines(filtered,[points[:,::-1].astype(np.int32)],False,1,1)
        for i,p,steps in ((ph,points[0],len(head)),(pt,points[-1],len(tail))):
            if i is not None:
                stops.append(dict(pole=contacts[i]['pole'],yx=p.tolist(),extension_steps=steps))
    return raw.astype(bool),filtered.astype(bool),dict(removed_short=short,kept=retained,
                extension_steps=expanded,stops=stops,kept_lengths=lengths)


def predict(stem, manifest, answers):
    src=prior.OUT/stem
    photo=np.asarray(Image.open(span.source(stem)).convert('RGB'))
    lab=cv2.cvtColor(photo.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[:,:,0]
    first=readmask(span.baseline_path(stem))
    previous=readmask(four.OUT/'review'/stem/'four_rules_centerline.png')
    paths=prior.get_paths(src)
    links=four.read(src,'links.json')
    evidence=four.read(four.OUT/stem,'evidence.json')
    sig=evidence['signatures']
    light=next((r['choice'] for r in answers if r['stem']==stem and r['id']=='COLOR'),None)
    standard=([np.asarray(sig[i]['color']) for i in evidence['seeds']],
              [(c,r*(1.30 if light=='WIDE' else 1.12)) for c,r in four.palette_profiles()])
    contacts,_,_=four.pole_contacts(stem,paths,manifest,answers)
    # Contact review numbers the approached end opposite to the bridge convention.
    contacts=[dict(p,side=0 if np.array_equal(paths[p['path']]['points'][0],p['yx']) else 1) for p in contacts]
    assert all(np.array_equal(paths[p['path']]['points'][0 if p['side']==0 else -1],p['yx']) for p in contacts)
    observed=np.zeros_like(first)
    for row in paths:
        observed[tuple(row['points'].T)]=True
    dist=distance_transform_edt(~first)
    variants=dict(previous=previous)
    audits={}
    for mode,stops in (('no_poles',[]),('poles',contacts)):
        _,used=four.select(observed,links,stops,sig,standard,True,True)
        selected={row['id'] for row in used['links']}
        chains=chain_paths(paths,links,selected,stops)
        all_paths,filtered,record=render_paths(chains,lab,dist,stops)
        audits[mode]=dict(record,chains=len(chains),blocked_links=used['blocked_at_pole'])
        if mode=='no_poles':
            variants.update(trace_only=all_paths,trace_min160=filtered)
        else:
            variants['trace_min160_poles']=filtered
    return variants,dict(audits,contacts=contacts,changed_by_poles=int(np.count_nonzero(variants['trace_min160']^variants['trace_min160_poles'])))


def render(folder, stem, variants, audit):
    photo=np.asarray(Image.open(span.source(stem)).convert('RGB'))
    panels=[]
    for name in ('previous','trace_min160_poles'):
        color=photo.copy()
        mask=variants[name].astype(np.uint8)
        color[cv2.dilate(mask,np.ones((5,5),np.uint8))>0]=(4,244,229)
        panels.append(Image.fromarray(color))
    panels[1].save(folder/'prediction_on_original.jpg',quality=93)
    marked=panels[1].copy()
    draw=ImageDraw.Draw(marked)
    font=ImageFont.load_default(size=34)
    for p in four.read(prior.OUT/stem,'poles.json'):
        x0,y0,x1,y1=p['box']
        draw.rectangle((x0,y0,x1,y1),outline=(255,154,44),width=6)
        draw.text((x0,max(0,y0-40)),p['id'],font=font,fill=(255,154,44),stroke_width=2,stroke_fill='black')
    for p in audit['poles']['stops']:
        y,x=p['yx']
        draw.ellipse((x-10,y-10,x+10,y+10),fill=(255,154,44),outline='white',width=2)
    marked.save(folder/'poles_and_stop_points.jpg',quality=93)
    board=Image.new('RGB',(1600,482),(20,24,30))
    draw=ImageDraw.Draw(board)
    font=ImageFont.load_default(size=19)
    for i,(title,panel) in enumerate(zip(('Previous','Tracking + 160 px + pole stops'),(panels[0],marked),strict=True)):
        draw.text((800*i+8,4),title,font=font,fill='white')
        board.paste(panel.resize((800,450),Image.Resampling.LANCZOS),(800*i,32))
    board.save(folder/'comparison.jpg',quality=93)


def main():
    OUT.mkdir(exist_ok=False)
    manifest=four.read(prior.OUT,'manifest.json')
    answers=four.read(four.OUT,'responses.json')
    frozen={stem:predict(stem,manifest,answers) for stem in span.STEMS}
    rows=[]
    for stem,(variants,audit) in frozen.items():
        folder=OUT/stem
        folder.mkdir()
        render(folder,stem,variants,audit)
        refroot=(span.BASE/'diagnostic_eval'/f'{stem}_v1' if stem==span.STEMS[0]
                 else span.BASE/'dev_transfer_20261107_v1'/stem/'evaluation')
        ref=readmask(refroot/'reference_centerlines.png')
        d=distance_transform_edt(~ref)
        xml=span.annotation_paths(stem,ref.shape)
        for name,line in variants.items():
            Image.fromarray(line.astype(np.uint8)*255).save(folder/f'{name}_centerline.png')
            scores,dist=metrics(line,ref,d)
            score=next(row for row in scores if row['tolerance_pixels']==5)
            if dist is None:
                dist=np.full(ref.shape,np.inf)
            rows.append(dict(stem=stem,method=name,**score,**span.continuity(dist,xml)))
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys=('centerline_precision','centerline_recall','centerline_f1','longest_run_mean')
    macro={method:{k:float(np.mean([r[k] for r in rows if r['method']==method])) for k in keys} for method in METHODS}
    span.save(OUT/'summary.json',dict(macro=macro,minimum_arc_px=MIN_LENGTH,terminal_minimum_arc_px=MIN_TERMINAL_LENGTH,
              max_extension_px_per_end=MAX_EXTENSION,pole_source='frozen DINO proposals with geometry',
              evaluation_tolerance_px=5,development_only=True,audit={s:a for s,(_,a) in frozen.items()}))
    for name,s in macro.items():
        print(name,*(f'{100*v:.2f}' for v in s.values()),flush=True)
    for stem,(_,a) in frozen.items():
        p=a['poles']
        print(stem,'short_removed',p['removed_short'],'kept',p['kept'],'stops',len(p['stops']),
              'new_stop',sum(x['extension_steps']>0 for x in p['stops']),
              'blocked_links',len(p['blocked_links']),'pole_delta',a['changed_by_poles'],flush=True)
    print('VERIFIED',OUT,flush=True)


if __name__=='__main__':
    main()
