import csv
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt

from evaluate_first_image_v1 import metrics,readmask
from evaluate_sam_auto_points_20261108_v1 import display_lines
from evaluate_sam_narrow_strip_20261108_v1 import reference_samples,continuous_reference_proximity
from run_langsam_ridge_paths_20261108_v1 import BASE,OUT,STEMS,POLICIES,input_paths,read_json,save_json


def compare(photo,ref,lines,roi,target):
    x0,y0,x1,y1=roi
    width,height=x1-x0,y1-y0
    source=photo.crop(roi)
    region=np.s_[y0:y1,x0:x1]
    panels=[('Original + XML reference',display_lines(source,ref[region],(255,231,64)))]
    titles=('LangSAM masks thinned','Original-image paths inside LangSAM boxes','Original-image paths with SAM guidance')
    for (key,line),label in zip(lines.items(),titles):
        panels.append((label,display_lines(source,line[region],(24,238,217))))
    board=Image.new('RGB',(2*width,2*(height+23)),(14,20,29))
    font=ImageFont.load_default(size=15)
    draw=ImageDraw.Draw(board)
    for i,(label,im) in enumerate(panels):
        x,y=(i%2)*width,(i//2)*(height+23)
        draw.text((x+5,y+3),label,fill='white',font=font)
        board.paste(im,(x,y+23))
    board.save(target,quality=92)


def evaluate():
    assert (OUT/'runtime.json').exists()
    options=['baseline',*POLICIES]
    cases=read_json(BASE/'wire_stage_attribution_20261108_v1/summary.json')['cases']
    rows=[]; diagnostic=[]
    for stem in STEMS:
        folder=OUT/stem
        if stem==STEMS[0]:
            evaluation=BASE/'diagnostic_eval'/f'{stem}_v1'
        else:
            evaluation=BASE/'dev_transfer_20261107_v1'/stem/'evaluation'
        ref=readmask(evaluation/'reference_centerlines.png')
        paths=reference_samples(read_json(evaluation/'reference_vectors.json'),ref.shape)
        refdist=distance_transform_edt(~ref)
        old=BASE/'skeleton_order_ablation_20261107_v1'/stem/'A_all_candidates/skeleton_then_union/centerline.png'
        lines={'baseline':readmask(old)}
        stats=read_json(folder/'box_stats.json')
        boxes={c['id']:c['box'] for c in stats}
        counts={}
        for key in POLICIES:
            line=readmask(folder/f'{key}_centerline.png')
            reconstructed=np.zeros(ref.shape,dtype=np.uint8)
            data=read_json(folder/f'{key}_paths.json')
            for d in data:
                xy=np.array(d['points'],dtype=np.int32)
                l,t,r,b=boxes[d['id']]
                assert (xy[:,0]>=l).all() and (xy[:,0]<r).all()
                assert (xy[:,1]>=t).all() and (xy[:,1]<b).all()
                cv2.polylines(reconstructed,[xy],False,1,1,lineType=cv2.LINE_8)
            assert np.array_equal(line,reconstructed>0)
            lines[key]=line
            counts[key]=len(data)
        baseline=lines['baseline']
        before=distance_transform_edt(~baseline)[ref]
        for key,line in lines.items():
            scores,dist=metrics(line,ref,refdist)
            if dist is None:
                dist=np.full(ref.shape,np.inf)
            for score in scores:
                t=score['tolerance_pixels']
                after=dist[ref]<=t
                total,long,mean=continuous_reference_proximity(dist,paths,t)
                score.update(stem=stem,policy=key,reference_fragments=total,
                    fragments_contiguous_near_80pct=long,mean_longest_near_fraction=mean,
                    recovered_reference_pixels=int((after & (before>t)).sum()),
                    lost_reference_pixels=int((~after & (before<=t)).sum()))
                rows.append(score)
        diagnostic.append(dict(stem=stem,boxes=len(stats),seeds=sum(c['seeds'] for c in stats),paths=counts,
            changed_pixels_when_using_SAM=int(np.count_nonzero(lines['box_photo'] ^ lines['box_photo_sam']))))
        photo=Image.open(read_json(input_paths(stem)[0]/'result.json')['image']).convert('RGB')
        assert photo.size==ref.shape[::-1]
        for category in ('box_nearby_no_raw_mask','organized_mask_nearby_no_line'):
            roi=next(c['crop_xyxy'] for c in cases if c['stem']==stem and c['category']==category)
            compare(photo,ref,lines,roi,folder/f'{category}_comparison.jpg')
        display_lines(photo,lines['box_photo_sam'],(24,238,217)).save(folder/'prediction_on_original.jpg',quality=85)
        print('SCORED',stem[:8],counts,flush=True)
    macro={}
    fields=('centerline_precision','centerline_recall','centerline_f1','mean_longest_near_fraction')
    for t in (3,5,10):
        macro[t]={}
        for policy in options:
            subset=[r for r in rows if r['tolerance_pixels']==t and r['policy']==policy]
            result={k:float(np.mean([r[k] for r in subset])) for k in fields}
            result.update({k:sum(r[k] for r in subset) for k in ('reference_fragments','fragments_contiguous_near_80pct',
                'recovered_reference_pixels','lost_reference_pixels')})
            macro[t][policy]=result
    with (OUT/'centerline_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    save_json(OUT/'summary.json',dict(rows=rows,macro=macro,diagnostic=diagnostic,
        continuity_proxy='longest uninterrupted sequence of reference samples within tolerance of a predicted pixel',
        frozen_inputs=True,reference_used_for_tracking=False))
    with ZipFile('langsam_ridge_paths_20261108_v1.zip','x',ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(OUT))
        for name in ('run_langsam_ridge_paths_20261108_v1.py','evaluate_langsam_ridge_paths_20261108_v1.py'):
            z.write(name,name)
        assert z.testzip() is None
    for k,v in macro[5].items():
        print(k,[round(100*v[f],2) for f in fields],v['fragments_contiguous_near_80pct'],flush=True)
    print('EVALUATION_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    evaluate()
