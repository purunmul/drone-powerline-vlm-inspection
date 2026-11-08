import csv
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from evaluate_first_image_v1 import metrics,readmask
from evaluate_sam_auto_points_20261108_v1 import REPLAY,display_lines
from run_sam_narrow_strip_20261108_v1 import BASE,OUT,PROMPT_RUN,STEMS,back_project,input_paths,read_json,save_json

POLICIES=('box_only','full_tile_supplement','clip_only_supplement','strip_replace','strip_supplement')


def reference_samples(vectors, shape):
    paths=[]
    for vector in vectors:
        points=np.array(vector['original_points_xy'])
        lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
        cum=np.r_[0,np.cumsum(lengths)]
        steps=np.linspace(0,cum[-1],int(np.ceil(cum[-1]))+1)
        x=np.rint(np.interp(steps,cum,points[:,0])).astype(int).clip(0,shape[1]-1)
        y=np.rint(np.interp(steps,cum,points[:,1])).astype(int).clip(0,shape[0]-1)
        paths.append((y,x))
    return paths


def continuous_reference_proximity(dist, paths, tolerance):
    runs=[]
    for y,x in paths:
        near=dist[y,x]<=tolerance
        transitions=np.flatnonzero(np.diff(np.r_[False,near,False]))
        longest=np.max(transitions[1::2]-transitions[::2],initial=0)
        runs.append(float(longest/len(x)))
    return len(runs),sum(v>=.8 for v in runs),float(np.mean(runs))


def compare(photo, ref, lines, roi, target):
    x0,y0,x1,y1=roi
    w,h=x1-x0,y1-y0
    region=np.s_[y0:y1,x0:x1]
    source=photo.crop(roi)
    labels=('Original + XML reference (yellow)','1  Box only','2  Full tile + points','3  Full tile result clipped to strip','4  Strip re-run, replacement','5  Strip re-run, supplement')
    panels=[display_lines(source,ref[region],(255,232,65))]
    panels.extend(display_lines(source,lines[k][region],(25,238,220)) for k in POLICIES)
    board=Image.new('RGB',(2*w,3*(h+24)),(16,20,29))
    draw=ImageDraw.Draw(board)
    font=ImageFont.load_default(size=15)
    for i,(label,panel) in enumerate(zip(labels,panels)):
        x,y=(i%2)*w,(i//2)*(h+24)
        draw.text((x+6,y+4),label,font=font,fill='white')
        board.paste(panel,(x,y+24))
    board.save(target,quality=92)


def evaluate():
    cases=read_json(BASE/'wire_stage_attribution_20261108_v1/summary.json')['cases']
    rows=[]; diagnostics=[]
    for stem in STEMS:
        folder=OUT/stem
        records=read_json(folder/'prompts.json')
        saved=read_json(PROMPT_RUN/stem/'prompts.json')
        assert len(records)==len(saved)
        for case,previous in zip(records,saved):
            assert {k:v for k,v in case.items() if k!='prompts'}=={k:v for k,v in previous.items() if k!='prompts'}
            assert [{k:v for k,v in p.items() if k!='strip'} for p in case['prompts']]==previous['prompts']
        raw=read_json(input_paths(stem)[0]/'result.json')
        photo=Image.open(raw['image']).convert('RGB')
        if stem==STEMS[0]:
            evaluation=BASE/'diagnostic_eval'/f'{stem}_v1'
        else:
            evaluation=BASE/'dev_transfer_20261107_v1'/stem/'evaluation'
        ref=readmask(evaluation/'reference_centerlines.png')
        refdist=distance_transform_edt(~ref)
        paths=reference_samples(read_json(evaluation/'reference_vectors.json'),ref.shape)
        original=PROMPT_RUN/stem
        baseline=readmask(original/'box_only_centerline.png')
        prior=BASE/'skeleton_order_ablation_20261107_v1'/stem/'candidate_skeletons'
        lines={k:np.zeros_like(ref) for k in POLICIES}
        lines['box_only']=baseline
        lines['full_tile_supplement']=readmask(original/'point_support_supplement_centerline.png')
        lines['clip_only_supplement']=baseline.copy()
        lines['strip_supplement']=baseline.copy()
        recreated=np.zeros_like(ref)
        strip_boxes=0
        strip_count=0
        full_count=0
        area_pairs=[]
        with np.load(folder/'accepted_masks.npz') as strips, np.load(REPLAY/stem/'accepted_masks.npz') as full:
            for case in records:
                x0,y0,x1,y1=case['roi']
                region=np.s_[y0:y1,x0:x1]
                size=(x1-x0,y1-y0)
                old=readmask(prior/f"{case['id']}.png")[region]
                recreated[region]|=old
                strip_line=np.zeros_like(old)
                any_strip=False
                for p in case['prompts']:
                    key=p['key']
                    info=p['strip']
                    transform=np.asarray(info['matrix'])
                    width,height=info['size']
                    footprint=back_project(np.ones((height,width),dtype=bool),transform,size)
                    assert (key in full.files)==all(p['supported'])
                    assert (key in strips.files)==info['kept']
                    if key in full.files:
                        clipped=full[key] & footprint
                        lines['clip_only_supplement'][region]|=skeletonize(clipped,method='zhang')
                        full_count+=1
                    if key in strips.files:
                        mask=strips[key]
                        assert int(mask.sum())==info['area'] and not (mask & ~footprint).any()
                        strip_line|=skeletonize(mask,method='zhang')
                        any_strip=True
                        strip_count+=1
                    if key in strips.files and key in full.files:
                        area_pairs.append([p['area'],info['area']])
                strip_boxes+=int(any_strip)
                lines['strip_supplement'][region]|=strip_line
                lines['strip_replace'][region]|=strip_line if any_strip else old
        assert np.array_equal(recreated,baseline)
        base_dist=distance_transform_edt(~baseline)
        for policy,line in lines.items():
            if policy.endswith('supplement'):
                assert not (baseline & ~line).any()
            scores,dist=metrics(line,ref,refdist)
            for score in scores:
                t=score['tolerance_pixels']
                before=base_dist[ref]<=t
                after=dist[ref]<=t
                total,long,mean=continuous_reference_proximity(dist,paths,t)
                added=line & ~baseline
                score.update(stem=stem,policy=policy,reference_fragments=total,
                    fragments_contiguous_near_80pct=long,mean_longest_near_fraction=mean,
                    recovered_reference_pixels=int((after & ~before).sum()),lost_reference_pixels=int((before & ~after).sum()),
                    added_pred_pixels=int(added.sum()),added_pred_outside_tolerance=int((added & (refdist>t)).sum()))
                rows.append(score)
            Image.fromarray(line.astype(np.uint8)*255).save(folder/f'{policy}_centerline.png')
        pair=np.asarray(area_pairs)
        diagnostics.append(dict(stem=stem,boxes=len(records),attempted=sum(len(c['prompts']) for c in records),
            full_tile_supported=full_count,strip_supported=strip_count,boxes_with_strip_mask=strip_boxes,
            paired_supported=len(pair),median_full_tile_mask_area=float(np.median(pair[:,0])) if len(pair) else None,
            median_strip_mask_area=float(np.median(pair[:,1])) if len(pair) else None))
        roi=next(c['crop_xyxy'] for c in cases if c['stem']==stem and c['category']=='box_nearby_no_raw_mask')
        compare(photo,ref,lines,roi,folder/'fixed_crop_comparison.jpg')
        display_lines(photo,lines['strip_supplement'],(25,238,220)).save(folder/'strip_supplement_on_original.jpg',quality=86)
        print('EVAL_IMAGE',stem[:8],flush=True)
    fields=('centerline_precision','centerline_recall','centerline_f1','mean_longest_near_fraction')
    macro={}
    for t in (3,5,10):
        macro[t]={}
        for policy in POLICIES:
            subset=[r for r in rows if r['policy']==policy and r['tolerance_pixels']==t]
            avg={k:float(np.mean([r[k] for r in subset])) for k in fields}
            avg.update({k:sum(r[k] for r in subset) for k in ('reference_fragments','fragments_contiguous_near_80pct','added_pred_pixels','added_pred_outside_tolerance')})
            macro[t][policy]=avg
    with (OUT/'centerline_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    save_json(OUT/'summary.json',dict(rows=rows,macro=macro,prompt_diagnostics=diagnostics,
        same_fixed_boxes_segments_and_points=True,
        continuity_proxy='longest contiguous run of uniformly spaced samples on an annotated fragment within distance of any predicted pixel; does not require a connected prediction'))
    with ZipFile('sam_narrow_strip_20261108_v1.zip','x',ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(OUT))
        for name in ('run_sam_narrow_strip_20261108_v1.py','evaluate_sam_narrow_strip_20261108_v1.py'):
            z.write(name,name)
    for policy,v in macro[5].items():
        print(policy,[round(100*v[k],2) for k in fields],v['fragments_contiguous_near_80pct'],flush=True)
    print('EVALUATION_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    evaluate()
