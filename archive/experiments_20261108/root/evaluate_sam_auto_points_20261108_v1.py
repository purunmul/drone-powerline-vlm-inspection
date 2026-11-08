from pathlib import Path
import csv
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.morphology import skeletonize

from evaluate_first_image_v1 import metrics, readmask
from run_sam_auto_points_20261108_v1 import BASE,OUT,STEMS,input_paths,read_json,save_json

REPLAY = BASE/'sam_auto_points_20261108_replay_v1'
POLICIES = ('box_only','strict_replace','strict_supplement','point_support_replace','point_support_supplement')

def display_lines(photo, pred, color):
    out = np.array(photo)
    out[maximum_filter(pred,size=3)] = color
    return Image.fromarray(out)

def compare_crop(photo, ref, lines, records, roi, target):
    x0,y0,x1,y1=roi
    w,h=x1-x0,y1-y0
    src=photo.crop(roi)
    crop=np.s_[y0:y1,x0:x1]
    font=ImageFont.load_default(size=15)
    prompt_panel=display_lines(src,ref[crop],(255,235,65))
    draw=ImageDraw.Draw(prompt_panel)
    for case in records:
        dx,dy=case['roi'][:2]
        for p in case['prompts']:
            for x,y in p['points']:
                x,y=x+dx-x0,y+dy-y0
                draw.ellipse((x-3,y-3,x+3,y+3),fill=(255,76,153),outline='black')
    color=(25,238,220)
    panels=[('XML reference (yellow) + auto points (pink)',prompt_panel)]
    for policy,label in [('box_only','Box-only SAM'),('point_support_replace','Auto-point SAM replaces box masks'),('point_support_supplement','Auto-point SAM supplements box masks')]:
        panels.append((label,display_lines(src,lines[policy][crop],color)))
    board=Image.new('RGB',(2*w,2*(h+24)),(16,20,29))
    draw=ImageDraw.Draw(board)
    for i,(label,panel) in enumerate(panels):
        x,y=(i%2)*w,(i//2)*(h+24)
        draw.text((x+6,y+4),label,font=font,fill='white')
        board.paste(panel,(x,y+24))
    board.save(target,quality=93)

def evaluate():
    rows=[]; diagnostics=[]
    cases=read_json(BASE/'wire_stage_attribution_20261108_v1/summary.json')['cases']
    for stem in STEMS:
        raw_dir,_ = input_paths(stem)
        folder=OUT/stem
        run=read_json(folder/'prompts.json')
        capture=REPLAY/stem
        assert run==read_json(capture/'prompts.json'), 'Replay differs from original inference'
        raw=read_json(raw_dir/'result.json')
        photo=Image.open(raw['image']).convert('RGB')
        if stem == STEMS[0]:
            evaluation=BASE/'diagnostic_eval'/f'{stem}_v1'
        else:
            evaluation=BASE/'dev_transfer_20261107_v1'/stem/'evaluation'
        reference=readmask(evaluation/'reference_centerlines.png')
        refdist=distance_transform_edt(~reference)
        prior=BASE/'skeleton_order_ablation_20261107_v1'/stem
        baseline=readmask(prior/'A_all_candidates/skeleton_then_union/centerline.png')
        lines={policy:np.zeros_like(reference) for policy in POLICIES}
        point_hits=[]; lengths=[]; mask_areas=[]; prompt_total=0; kept_boxes=0; support_boxes=0
        strict_count=0; supported_count=0
        with np.load(folder/'accepted_masks.npz') as strict, np.load(capture/'accepted_masks.npz') as supported:
            for case in run:
                x0,y0,x1,y1=case['roi']
                region=np.s_[y0:y1,x0:x1]
                old=readmask(prior/'candidate_skeletons'/f"{case['id']}.png")
                assert old.shape==reference.shape
                assert int(old.sum())==int(old[region].sum())
                old=old[region]
                lines['box_only'][region]|=old
                new_strict=np.zeros_like(old)
                new_support=np.zeros_like(old)
                any_strict=False; any_support=False
                for p in case['prompts']:
                    points=np.rint(p['points']).astype(int)+[x0,y0]
                    point_hits.append([bool(refdist[y,x]<=5) for x,y in points])
                    lengths.append(p['length']); mask_areas.append(p['area'])
                    prompt_total+=1
                    key=p['key']
                    assert (key in supported.files)==all(p['supported'])
                    assert (key in strict.files)==p['kept']
                    if key not in supported.files:
                        continue
                    mask=supported[key]
                    assert int(mask.sum())==p['area']
                    new_line=skeletonize(mask,method='zhang')
                    new_support |= new_line
                    any_support=True
                    supported_count+=1
                    if key in strict.files:
                        assert np.array_equal(mask,strict[key])
                        new_strict |= new_line
                        any_strict=True
                        strict_count+=1
                kept_boxes += int(any_strict); support_boxes += int(any_support)
                lines['strict_replace'][region] |= new_strict if any_strict else old
                lines['strict_supplement'][region] |= new_strict | old
                lines['point_support_replace'][region] |= new_support if any_support else old
                lines['point_support_supplement'][region] |= new_support | old
        assert np.array_equal(lines['box_only'],baseline), 'Baseline centerline did not replay'
        assert not (baseline & ~lines['strict_supplement']).any()
        assert not (baseline & ~lines['point_support_supplement']).any()
        baseline_dist=distance_transform_edt(~baseline)
        for policy,line in lines.items():
            scores,dist=metrics(line,reference,refdist)
            for score in scores:
                t=score['tolerance_pixels']
                before=baseline_dist[reference]<=t
                after=dist[reference]<=t if dist is not None else np.zeros(reference.sum(),dtype=bool)
                score.update(stem=stem,policy=policy,recovered_reference_pixels=int((after & ~before).sum()),lost_reference_pixels=int((before & ~after).sum()))
                rows.append(score)
            Image.fromarray(line.astype(np.uint8)*255).save(folder/f'{policy}_centerline.png')
        hits=np.array(point_hits,dtype=bool)
        diagnostics.append(dict(stem=stem,boxes=len(run),prompts=prompt_total,strict_kept=strict_count,
            support_kept=supported_count,boxes_strict=kept_boxes,boxes_support=support_boxes,
            seed_points_within5=int(hits.sum()),prompts_all_points_within5=int(hits.all(axis=1).sum()),
            median_mask_area=float(np.median(mask_areas)),median_segment_length=float(np.median(lengths))))
        roi=next(c['crop_xyxy'] for c in cases if c['stem']==stem and c['category']=='box_nearby_no_raw_mask')
        compare_crop(photo,reference,lines,run,roi,folder/'fixed_crop_comparison.jpg')
        display_lines(photo,lines['point_support_supplement'],(25,238,220)).save(folder/'supplement_prediction_on_original.jpg',quality=86)
    macro={}
    for t in (3,5,10):
        macro[t]={}
        for policy in POLICIES:
            subset=[r for r in rows if r['policy']==policy and r['tolerance_pixels']==t]
            macro[t][policy]={k:float(np.mean([r[k] for r in subset])) for k in ('centerline_precision','centerline_recall','centerline_f1')}
    with (OUT/'centerline_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    save_json(OUT/'summary.json',dict(rows=rows,macro=macro,prompt_diagnostics=diagnostics,baseline_replayed_pixel_exact=True,original_and_replay_prompts_equal=True))
    with ZipFile('sam_auto_points_20261108_v1.zip','x',ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(OUT))
        for name in ('run_sam_auto_points_20261108_v1.py','replay_sam_auto_points_20261108_v1.py','evaluate_sam_auto_points_20261108_v1.py'):
            z.write(name,name)
    for policy,score in macro[5].items():
        print(policy, [round(100*v,2) for v in score.values()],flush=True)
    print('EVALUATION_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    evaluate()
