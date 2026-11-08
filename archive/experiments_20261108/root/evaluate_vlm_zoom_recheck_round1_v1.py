import csv
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from evaluate_first_image_v1 import metrics,readmask
from evaluate_sam_auto_points_20261108_v1 import display_lines
from run_vlm_zoom_recheck_round1_v1 import BASE,STEMS,OUT,BUDGET,baseline_path,input_paths,read_json,save_json

POLICIES=('baseline','fixed','vlm')
KEYS=('centerline_precision','centerline_recall','centerline_f1')

def selection_map(photo, regions, choice, output):
    thumb=photo.resize((1152,648),Image.Resampling.LANCZOS)
    scale=thumb.width/photo.width
    board=Image.new('RGB',(thumb.width,thumb.height+36),(20,27,39))
    board.paste(thumb,(0,36))
    draw=ImageDraw.Draw(board)
    font=ImageFont.load_default(size=18)
    draw.text((10,7),'Selected regions | cyan: Qwen   orange: fixed rule   yellow: both',fill='white',font=font)
    for k,rec in regions.items():
        a=k in choice['fixed']; b=k in choice['vlm']
        if not (a or b):
            continue
        color=(255,231,71) if a and b else (255,150,62) if a else (30,236,220)
        x0,y0,x1,y1=[int(v*scale) for v in rec['core']]
        draw.rectangle((x0,y0+36,x1-1,y1+35),outline=color,width=3)
        draw.rectangle((x0+5,y0+42,x0+44,y0+62),fill=(20,27,39))
        draw.text((x0+7,y0+41),k,fill=color,font=font)
    board.save(output,quality=88)


def evaluate():
    manifest=read_json(OUT/'manifest.json')
    choices=read_json(OUT/'choices.json')
    responses=read_json(OUT/'responses.json')
    zoom=read_json(OUT/'zoom_inference.json')
    review=OUT/'review'
    review.mkdir(exist_ok=False)
    rows=[]; region_rows=[]; views=[]
    for stem in STEMS:
        source=OUT/stem
        dest=review/stem
        dest.mkdir()
        if stem==STEMS[0]:
            evaluation=BASE/'diagnostic_eval'/f'{stem}_v1'
        else:
            evaluation=BASE/'dev_transfer_20261107_v1'/stem/'evaluation'
        ref=readmask(evaluation/'reference_centerlines.png')
        base=readmask(baseline_path(stem))
        refdist=distance_transform_edt(~ref)
        before=distance_transform_edt(~base)[ref]
        regions={r['id']:r for r in manifest if r['stem']==stem and not r['control']}
        choice=choices[stem]
        assert all(len(set(choice[k]))==BUDGET for k in ('fixed','vlm'))
        ids=set(choice['fixed'])|set(choice['vlm'])
        scans={r['id']:r for r in zoom if r['stem']==stem}
        assert ids==scans.keys()
        lines={k:base.copy() for k in POLICIES}
        for ident in sorted(ids):
            r=regions[ident]
            x0,y0,x1,y1=r['core']
            a,b,c,d=r['zoom']
            region=np.s_[y0:y1,x0:x1]
            with np.load(source/f'{ident}_zoom.npz') as pack:
                masks=pack['masks']
                assert masks.shape==(scans[ident]['count'],d-b,c-a)
                new=np.zeros((d-b,c-a),dtype=bool)
                for mask in masks:
                    new|=skeletonize(mask,method='zhang')
            core=new[y0-b:y1-b,x0-a:x1-a]
            assert core.shape==base[region].shape
            novel=core&~base[region]
            for k in ('fixed','vlm'):
                if ident in choice[k]:
                    lines[k][region]|=core
            region_rows.append(dict(stem=stem,region=ident,**{k:ident in choice[k] for k in ('fixed','vlm')},
                new_pixels=int(novel.sum()),new_within5=int((novel&(refdist[region]<=5)).sum()),candidates=scans[ident]['count']))
        for policy,line in lines.items():
            assert not (base&~line).any()
            extra=line&~base
            scores,dist=metrics(line,ref,refdist)
            for score in scores:
                t=score['tolerance_pixels']
                after=dist[ref]<=t
                assert (after|(before>t)).all()
                score.update(stem=stem,policy=policy,additional_pred_pixels=int(extra.sum()),
                    additional_pred_matching=int((extra&(refdist<=t)).sum()),recovered_gt=int((after&(before>t)).sum()))
                rows.append(score)
            Image.fromarray(line.astype(np.uint8)*255).save(dest/f'{policy}_centerline.png')
        photo=Image.open(read_json(input_paths(stem)[0]/'result.json')['image']).convert('RGB')
        assert photo.size==ref.shape[::-1]
        selection_map(photo,regions,choice,dest/'selection_map.jpg')
        display_lines(photo,lines['vlm'],(25,235,226)).save(dest/'vlm_prediction_on_original.jpg',quality=84)
        views.append(regions[choice['vlm'][0]]['board'])
        print('SCORED',stem[:8],choice['fixed'],choice['vlm'],'ctrl',choice['control_pass'],flush=True)
    macro={}
    for t in (3,5,10):
        macro[t]={}
        for k in POLICIES:
            subset=[r for r in rows if r['policy']==k and r['tolerance_pixels']==t]
            out={key:float(np.mean([r[key] for r in subset])) for key in KEYS}
            out.update({key:sum(r[key] for r in subset) for key in ('additional_pred_pixels','additional_pred_matching','recovered_gt')})
            macro[t][k]=out
    for name,records in [('metrics',rows),('regions',region_rows)]:
        with (review/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    save_json(review/'summary.json',dict(macro=macro,rows=rows,choices=choices,region_rows=region_rows))
    with ZipFile('vlm_zoom_recheck_round1_v1.zip','x',ZIP_DEFLATED) as z:
        for path in sorted(review.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(review))
        for name in ('manifest.json','protocol.json','qwen_config.json','choices.json','responses.json','zoom_inference.json','selection_diagnostics.json'):
            z.write(OUT/name,name)
        for rel in views:
            z.write(OUT/rel,'boards/'+rel)
        for name in ('run_vlm_zoom_recheck_round1_v1.py','evaluate_vlm_zoom_recheck_round1_v1.py'):
            z.write(name,name)
        assert z.testzip() is None
    for k,data in macro[5].items():
        print(k,[round(data[key]*100,2) for key in KEYS],'recovered',data['recovered_gt'],flush=True)
    print('EVAL_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    evaluate()
