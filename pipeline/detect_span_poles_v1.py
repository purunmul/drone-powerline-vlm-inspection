from repository_paths import image_path
import json
from pathlib import Path
from PIL import Image
from lang_sam.models.gdino import GDINO
from run_vlm_zoom_recheck_round1_v1 import STEMS, input_paths, read_json

OUT = Path('results/wire_span_continuity_20261108_v1')
PROMPT = 'utility pole. electricity pole. power pole.'


def overlap(a,b):
    x0,y0,x1,y1 = a
    u0,v0,u1,v1 = b
    area = max(0,min(x1,u1)-max(x0,u0))*max(0,min(y1,v1)-max(y0,v0))
    return area/max(1,min((x1-x0)*(y1-y0),(u1-u0)*(v1-v0)))


def main():
    OUT.mkdir(exist_ok=True)
    output = OUT/'poles.json'
    if output.exists():
        raise FileExistsError(output)
    model=GDINO()
    model.build_model()
    all_poles={}
    for stem in STEMS:
        source=image_path(read_json(input_paths(stem)[0]/'result.json')['image'])
        with Image.open(source) as opened:
            image=opened.convert('RGB')
        w,h=image.size
        tiles=[(0,0,w,h)]+[(x,y,x+1280,y+1280) for y in (0,h-1280) for x in (0,850,1700,w-1280)]
        boxes=[]
        for x,y,x1,y1 in tiles:
            crop=image.crop((x,y,x1,y1))
            result=model.predict([crop],[PROMPT],.23,.23)[0]
            for box,score in zip(result['boxes'].tolist(),result['scores'].tolist(),strict=True):
                a,b,c,d=box
                if d-b<45 or (d-b)/(max(1,c-a))<1.15:
                    continue
                if (y and b<9) or (y1<h and d>crop.height-9):
                    continue
                bounds=[max(x,round(a+x)),max(y,round(b+y)),min(x1,round(c+x)),min(y1,round(d+y))]
                boxes.append(dict(box=bounds,score=score,view='whole' if crop.size==image.size else 'tile'))
        kept=[]
        for row in sorted(boxes,key=lambda row:-row['score']):
            if any(overlap(row['box'],other['box'])>.58 for other in kept):
                continue
            kept.append(dict(id=f'P{len(kept)+1:02d}',**row))
            if len(kept)>=8:
                break
        all_poles[stem]=kept
        print(stem,[(p['id'],p['box'],round(p['score'],3)) for p in kept],flush=True)
    output.write_text(json.dumps(dict(prompt=PROMPT,poles=all_poles),indent=2)+'\n')


if __name__=='__main__':
    main()
