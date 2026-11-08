from pathlib import Path
import csv
import json
import math
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, maximum_filter

STEMS = ('BEGO7553_frame_001098', 'PUYL5573_frame_000165', 'KZIV5445_frame_000488')
OUT = Path('results/wire_stage_attribution_20261108_v1')
CATEGORIES = ('no_box_nearby', 'box_nearby_no_raw_mask', 'raw_mask_lost_at_dedup', 'organized_mask_nearby_no_line', 'centerline_match')
BASE = Path('results')

def read_mask(path):
    return np.asarray(Image.open(path).convert('L')) > 0

def distances(mask, ref):
    if not mask.any():
        return np.full(ref.sum(), np.inf)
    return distance_transform_edt(~mask)[ref]

def bounds(xyxy, w, h):
    x0, y0, x1, y1 = xyxy
    return (max(0, math.ceil(x0)), max(0, math.ceil(y0)), min(w, math.floor(x1) + 1), min(h, math.floor(y1) + 1))

def overlay(image, pixels, rgb, opacity):
    colors = np.array(image)
    colors[pixels] = np.rint(colors[pixels] * (1-opacity) + np.array(rgb) * opacity).astype(np.uint8)
    return Image.fromarray(colors)

def label_at(draw, xy, text, color, font):
    rect = draw.textbbox(xy, text, font=font)
    draw.rectangle((rect[0]-2, rect[1]-1, rect[2]+2, rect[3]+1), fill=(15,20,28))
    draw.text(xy, text, fill=color, font=font)

def draw_case(photo, target, raw, line, boxes, candidate_ids, category, path):
    width, height = 470, 285
    h, w = target.shape
    points = np.argwhere(target)
    if len(points) == 0:
        return None
    # Sample actual failed pixels and select the crop with the most of this failure.
    centers = points[::max(1, len(points)//750)]
    integral = np.pad(target.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1,0),(1,0)))
    ys = np.clip(centers[:,0]-height//2, 0, h-height)
    xs = np.clip(centers[:,1]-width//2, 0, w-width)
    counts = integral[ys+height,xs+width]-integral[ys,xs+width]-integral[ys+height,xs]+integral[ys,xs]
    best = int(counts.argmax())
    x, y = int(xs[best]), int(ys[best]); x1, y1 = x+width, y+height
    region = np.s_[y:y1,x:x1]
    orig = photo.crop((x,y,x1,y1))
    box_panel = orig.copy()
    d = ImageDraw.Draw(box_panel)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').exists() else ImageFont.load_default()
    ids = []
    for (l,t,r,b), candidate in zip(boxes,candidate_ids):
        if max(x,l) < min(x1,r) and max(y,t) < min(y1,b):
            ids.append(candidate)
            d.rectangle((l-x,t-y,r-1-x,b-1-y), outline=(255,174,51), width=2)
    yellow, orange, teal, violet = (255,239,78), (255,174,51), (20,232,208), (227,122,253)
    mark = maximum_filter(target[region], size=3)
    board = Image.new('RGB',(2*width, 2*height+85),(15,20,28))
    header = ImageDraw.Draw(board)
    label_at(header,(8,5),f'{path.stem} | crop ({x}, {y}) to ({x1}, {y1})', (240,243,248),font)
    label_at(header,(8,26),'Yellow = XML reference pixels in this failure category. Same crop in all panels.', yellow,font)
    panels = [(overlay(orig,mark,yellow,0.88),'1   Original + affected reference',yellow),
              (overlay(box_panel,mark,yellow,0.88),'2   Grounding DINO boxes + affected reference',orange),
              (overlay(overlay(orig,raw[region],teal,0.63),mark,yellow,0.88),'3   Raw SAM masks + affected reference',teal),
              (overlay(overlay(orig,maximum_filter(line[region],size=3),violet,0.88),mark,yellow,0.88),'4   Final centerlines + affected reference',violet)]
    for i,(panel,title,color) in enumerate(panels):
        left=(i%2)*width; top=43+(i//2)*(height+21)
        label_at(header,(left+8,top+1),title,color,font)
        board.paste(panel,(left,top+21))
    board.save(path,quality=92)
    return dict(category=category,crop_xyxy=[x,y,x1,y1],reference_pixels=int(counts[best]),intersecting_raw_ids=ids,board=path.name)

OUT.mkdir(exist_ok=True)
all_rows, cases, provenance = [], [], []
for stem in STEMS:
    if stem.startswith('BEGO'):
        tile_dir=BASE/'langsam_baseline'/f'{stem}_tiles1024_v1'
        evaluation=BASE/'diagnostic_eval'/f'{stem}_v1'
    else:
        root=BASE/'dev_transfer_20261107_v1'/stem
        tile_dir=root/'tiles'
        evaluation=root/'evaluation'
    sk_path=BASE/'skeleton_order_ablation_20261107_v1'/stem/'A_all_candidates/skeleton_then_union/centerline.png'
    photo_path=Path('data/initial_8/images')/f'{stem}.jpg'
    report=json.loads((tile_dir/'result.json').read_text())
    ref=read_mask(evaluation/'reference_centerlines.png')
    raw=read_mask(tile_dir/'union_mask.png')
    organized=read_mask(evaluation/'A_all_candidates/mask.png')
    line=read_mask(sk_path)
    h,w=ref.shape
    assert (w,h)==tuple(report['original_size']) and ref.any()
    assert raw.sum()==report['union_mask_area_pixels']
    assert not (organized & ~raw).any() and not (line & ~organized).any()
    box_mask=np.zeros_like(ref)
    boxes=[]; ids=[]; candidates=[]
    tiles={tile['tile_id']: tile['crop_xyxy_original'] for tile in report['tiles']}
    for c in report['candidates']:
        l,t,r,b=bounds(c['box_xyxy_original'],w,h)
        tl,tt,tr,tb=tiles[c['tile_id']]
        l,t,r,b=max(l,tl),max(t,tt),min(r,tr),min(b,tb)
        if l < r and t < b:
            box_mask[t:b,l:r]=True
            boxes.append((l,t,r,b)); ids.append(c['id'])
        candidates.append({k:c[k] for k in ('id','tile_id','box_xyxy_original','detector_score','sam_mask_score','mask_area_pixels')})
    box_d, raw_d, org_d, line_d = (distances(m,ref) for m in (box_mask,raw,organized,line))
    assert np.all(raw_d<=org_d) and np.all(org_d<=line_d)
    provenance.append(dict(stem=stem, report=str(tile_dir/'result.json'), mask=str(tile_dir/'union_mask.png'), reference=str(evaluation/'reference_centerlines.png'), organized_mask=str(evaluation/'A_all_candidates/mask.png'), line=str(sk_path), raw_candidates=candidates, box_image_pct=100*float(box_mask.mean()), dedup_removed_mask_pixels=int((raw & ~organized).sum())))
    for tol in (3,5,10):
        no_raw, no_org, no_line = raw_d>tol, org_d>tol, line_d>tol
        categories=(no_raw & (box_d>tol),no_raw & (box_d<=tol),~no_raw & no_org,~no_org & no_line,~no_line)
        assert np.all(np.sum(categories,axis=0)==1)
        row=dict(stem=stem,tolerance=tol,reference_pixels=int(ref.sum()))
        for key, flags in zip(CATEGORIES,categories):
            row[key+'_px']=int(flags.sum())
            row[key+'_pct']=round(100*float(flags.mean()),4)
        row['within_box_exact_no_raw_mask_pct']=round(100*float((no_raw & (box_d==0)).mean()),4)
        row['within_any_box_pct']=round(100*float((box_d<=tol).mean()),4)
        all_rows.append(row)
        if tol==5:
            print(stem[:8], int(ref.sum()), [round(row[k+'_pct'],2) for k in CATEGORIES], 'exact',row['within_box_exact_no_raw_mask_pct'],'box_total',row['within_any_box_pct'],flush=True)
            photo=Image.open(photo_path).convert('RGB')
            assert photo.size==(w,h)
            for key, flags in zip(CATEGORIES[:4],categories[:4]):
                target=np.zeros_like(ref); target[ref]=flags
                case=draw_case(photo,target,raw,line,boxes,ids,key,OUT/f'{stem[:8]}_{key}.jpg')
                if case:
                    cases.append(dict(stem=stem,**case))
means={tol:{k:round(float(np.mean([r[k] for r in all_rows if r['tolerance']==tol])),4) for k in all_rows[0] if k.endswith('_pct')} for tol in (3,5,10)}
with (OUT/'stage_counts.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=all_rows[0]); writer.writeheader(); writer.writerows(all_rows)
(OUT/'summary.json').write_text(json.dumps(dict(rows=all_rows,macro_means=means,cases=cases,provenance=provenance),indent=2))
print('MEAN',json.dumps(means),flush=True)
with ZipFile('wire_stage_attribution_20261108_v1.zip','x',ZIP_DEFLATED) as z:
    for path in sorted(OUT.iterdir()):
        z.write(path,path.name)
    z.write(__file__,Path(__file__).name)
print('DONE',len(all_rows),len(cases),flush=True)
