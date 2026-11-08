import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from sam_clip_transfer_config_20261108_v1 import ROOT,STEMS
from run_sam_auto_points_20261108_v1 import digest,read_json,save_json
from run_sam_narrow_strip_20261108_v1 import back_project,strip_geometry
from evaluate_first_image_v1 import metrics,readmask
from evaluate_sam_auto_points_20261108_v1 import display_lines
from evaluate_sam_narrow_strip_20261108_v1 import reference_samples,continuous_reference_proximity

POLICIES=('box_only','clip_supplement')

def make_reference(image_annotation, size):
    w,h=size
    assert (int(image_annotation.attrib['width']),int(image_annotation.attrib['height']))==(w,h)
    ref=np.zeros((h,w),dtype=np.uint8)
    vectors=[]
    for line in image_annotation.findall('polyline'):
        assert line.attrib['label'] in {'powerline','polyline'}
        points=np.array([list(map(float,p.split(','))) for p in line.attrib['points'].split(';')])
        assert len(points)>=2
        raster=np.rint(points).astype(np.int32).clip(0,[w-1,h-1])
        cv2.polylines(ref,[raster],False,1,1,lineType=cv2.LINE_8)
        vectors.append(dict(original_points_xy=points.tolist()))
    assert ref.any()
    return ref.astype(bool),vectors


def compare(photo, ref, lines, target):
    width=940; height=529
    color=(25,238,220)
    panels=(('Original',photo),('XML reference (yellow)',display_lines(photo,ref,(255,230,65))),
            ('Box-only prediction',display_lines(photo,lines['box_only'],color)),
            ('Box + clipped supplemental masks',display_lines(photo,lines['clip_supplement'],color)))
    board=Image.new('RGB',(2*width,2*(height+24)),(16,20,29))
    draw=ImageDraw.Draw(board)
    font=ImageFont.load_default(size=17)
    for i,(label,panel) in enumerate(panels):
        x,y=(i%2)*width,(i//2)*(height+24)
        draw.text((x+8,y+4),label,font=font,fill='white')
        board.paste(panel.resize((width,height),Image.Resampling.LANCZOS),(x,y+24))
    board.save(target,quality=86)


def evaluate():
    config=read_json(ROOT/'config.json')
    for name,checksum in config['source_sha256'].items():
        assert digest(Path(name))==checksum
    prompt_root=ROOT/'full_tile_prompts'
    assert (prompt_root/'runtime.json').exists()
    annotations=Path('data/initial_8/annotations.xml')
    tree=ET.parse(annotations)
    by_stem={Path(a.attrib['name']).stem:a for a in tree.findall('image')}
    assert set(STEMS)<=by_stem.keys()
    with Path('data/initial_8/selection.csv').open(newline='',encoding='utf-8-sig') as f:
        expected={Path(r['image']).stem:int(r['line_count']) for r in csv.DictReader(f) if r['score_enabled']=='True'}
    out=ROOT/'review'
    out.mkdir(exist_ok=False)
    rows=[]; diagnostics=[]
    for stem in STEMS:
        src=ROOT/stem
        folder=out/stem
        folder.mkdir()
        raw=read_json(src/'tiles/result.json')
        organized=read_json(src/'candidates/candidates.json')['candidates']
        photo_path=Path(raw['image'])
        assert digest(photo_path)==raw['sha256']==config['image_sha256'][stem]
        photo=Image.open(photo_path).convert('RGB')
        ref,vectors=make_reference(by_stem[stem],photo.size)
        assert len(vectors)==expected[stem]
        refdist=distance_transform_edt(~ref)
        paths=reference_samples(vectors,ref.shape)
        prompt_dir=prompt_root/stem
        records=read_json(prompt_dir/'prompts.json')
        assert {r['id']:r['raw_id'] for r in records}=={c['id']:c['representative_raw_id'] for c in organized}
        baseline=np.zeros_like(ref)
        extra=np.zeros_like(ref)
        for c in organized:
            mask=readmask(src/'candidates'/c['mask'])
            assert mask.shape==ref.shape and int(mask.sum())==c['mask_area_pixels']
            baseline |= skeletonize(mask,method='zhang')
        tested=kept=0
        with np.load(prompt_dir/'accepted_masks.npz') as masks:
            for case in records:
                x0,y0,x1,y1=case['roi']
                size=(x1-x0,y1-y0)
                for prompt in case['prompts']:
                    tested+=1
                    key=prompt['key']
                    assert (key in masks.files)==all(prompt['supported'])
                    if key not in masks.files:
                        continue
                    kept+=1
                    matrix,(w,h),_=strip_geometry(prompt['segment'],case['box'])
                    footprint=back_project(np.ones((h,w),dtype=bool),matrix,size)
                    mask=masks[key]
                    assert mask.shape==footprint.shape and int(mask.sum())==prompt['area']
                    extra[y0:y1,x0:x1] |= skeletonize(mask & footprint,method='zhang')
        lines=dict(box_only=baseline,clip_supplement=baseline|extra)
        added=extra & ~baseline
        before_dist=distance_transform_edt(~baseline) if baseline.any() else np.full(ref.shape,np.inf)
        for policy,line in lines.items():
            scores,dist=metrics(line,ref,refdist)
            if dist is None:
                dist=np.full(ref.shape,np.inf)
            for score in scores:
                t=score['tolerance_pixels']
                total,long,mean=continuous_reference_proximity(dist,paths,t)
                before=before_dist[ref]<=t
                after=dist[ref]<=t
                assert (after|~before).all()
                score.update(stem=stem,policy=policy,reference_fragments=total,
                    fragments_contiguous_near_80pct=long,mean_longest_near_fraction=mean,
                    recovered_reference_pixels=int((after&~before).sum()),
                    added_pred_pixels=int(added.sum()) if policy=='clip_supplement' else 0,
                    added_pred_outside_tolerance=int((added&(refdist>t)).sum()) if policy=='clip_supplement' else 0)
                rows.append(score)
            Image.fromarray(line.astype(np.uint8)*255).save(folder/f'{policy}_centerline.png')
        display_lines(photo,lines['clip_supplement'],(25,238,220)).save(folder/'prediction_on_original.jpg',quality=82)
        compare(photo,ref,lines,folder/'comparison.jpg')
        diagnostics.append(dict(stem=stem,raw_candidates=raw['candidate_count_raw'],organized_candidates=len(records),
                                prompts=tested,prompts_supported=kept,langsam_inference_seconds=raw['inference_seconds_total'],
                                langsam_peak_gib=raw['gpu_peak_allocated_gib']))
        print('SCORED',stem[:8],len(records),tested,kept,flush=True)
    macro={}
    fields=('centerline_precision','centerline_recall','centerline_f1','mean_longest_near_fraction')
    for t in (3,5,10):
        macro[t]={}
        for policy in POLICIES:
            subset=[r for r in rows if r['policy']==policy and r['tolerance_pixels']==t]
            result={k:float(np.mean([r[k] for r in subset])) for k in fields}
            result.update({k:sum(r[k] for r in subset) for k in ('reference_fragments','fragments_contiguous_near_80pct',
                                                                'added_pred_pixels','added_pred_outside_tolerance')})
            macro[t][policy]=result
    with (out/'centerline_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    save_json(out/'summary.json',dict(rows=rows,macro=macro,diagnostics=diagnostics,
        annotation_sha256=digest(annotations),inference_runtime=read_json(prompt_root/'runtime.json'),
        independent_final_eval_160_used=False))
    with ZipFile('sam_clip_transfer_20261108_v1.zip','x',ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(out))
        z.write(ROOT/'config.json','config.json')
        z.write(prompt_root/'config.json','prompt_config.json')
        for name in ('sam_clip_transfer_config_20261108_v1.py','detect_sam_clip_transfer_20261108_v1.py',
                     'prompt_sam_clip_transfer_20261108_v1.py','evaluate_sam_clip_transfer_20261108_v1.py'):
            z.write(name,name)
        assert z.testzip() is None
    for policy,result in macro[5].items():
        print(policy,[round(result[k]*100,2) for k in fields],result['fragments_contiguous_near_80pct'],flush=True)
    print('EVALUATION_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    evaluate()
