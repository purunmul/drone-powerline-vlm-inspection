from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from run_sam_auto_points_20261108_v1 import (
    BASE, OUT as PROMPT_RUN, STEMS, CHECKPOINT, CHECKPOINT_SHA, POINT_TOL,
    digest, input_paths, read_json, save_json,
)

OUT=BASE/'sam_narrow_strip_20261108_v1'
HALF_WIDTH=24
END_MARGIN=24

def strip_geometry(segment, box):
    ends=np.asarray(segment,dtype=np.float64)
    length=np.linalg.norm(ends[1]-ends[0])
    direction=(ends[1]-ends[0])/length
    normal=np.array([-direction[1], direction[0]])
    axes=np.stack([direction,normal])
    transform=np.column_stack([axes,[END_MARGIN,HALF_WIDTH]-axes@ends[0]])
    width=int(np.ceil(length))+2*END_MARGIN+1
    height=2*HALF_WIDTH+1
    x0,y0,x1,y1=box
    corners=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    corners=(corners@axes.T+transform[:,2]).astype(np.float32)
    bounds=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype=np.float32)
    area, polygon=cv2.intersectConvexConvex(corners,bounds)
    assert area>0 and polygon is not None
    vertices=polygon.reshape(-1,2)
    local_box=np.concatenate([vertices.min(axis=0),vertices.max(axis=0)])
    local_box=np.clip(local_box,0,[width-1,height-1,width-1,height-1]).astype(np.float32)
    return transform,(width,height),local_box

def back_project(pixels, transform, tile_size):
    return cv2.warpAffine(pixels.astype(np.uint8),transform,tile_size,
                          flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP).astype(bool)

def support_at(mask, points):
    h,w=mask.shape
    supported=[]
    for x,y in np.rint(points).astype(int):
        r=POINT_TOL
        x0,x1,y0,y1=max(0,x-r),min(w,x+r+1),max(0,y-r),min(h,y+r+1)
        yy,xx=np.ogrid[y0:y1,x0:x1]
        supported.append(bool((mask[y0:y1,x0:x1]&((xx-x)**2+(yy-y)**2<=r*r)).any()))
    return supported

def infer():
    import torch
    from lang_sam.models.sam import SAM

    assert digest(CHECKPOINT)==CHECKPOINT_SHA and torch.cuda.is_available()
    OUT.mkdir(exist_ok=False)
    save_json(OUT/'config.json',dict(half_width_px=HALF_WIDTH, strip_height_px=2*HALF_WIDTH+1,
        segment_end_margin_px=END_MARGIN, input_prompt_run=str(PROMPT_RUN),
        original_box_projection='axis-aligned envelope of intersection of rotated original box with strip',
        rgb_sampling='bilinear from the original tile; black border outside the tile',
        mask_sampling='nearest-neighbor inverse projection into original tile',
        selection='single SAM2.1 mask per prompt; all saved points within 3 original pixels of foreground',
        model='sam2.1_hiera_small', checkpoint_sha256=CHECKPOINT_SHA,
        reference_used=False,manual_points=False,vlm_used=False))
    sam=SAM()
    sam.build_model('sam2.1_hiera_small',ckpt_path=str(CHECKPOINT))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for stem in STEMS:
        report=read_json(input_paths(stem)[0]/'result.json')
        photo_path=Path(report['image'])
        assert digest(photo_path)==report['sha256']
        photo=Image.open(photo_path).convert('RGB')
        records=read_json(PROMPT_RUN/stem/'prompts.json')
        folder=OUT/stem
        folder.mkdir()
        tiles={}
        masks={}
        for case in records:
            tile_id=case['tile']
            if tile_id not in tiles:
                tiles[tile_id]=np.array(photo.crop(case['roi']))
            rgb=tiles[tile_id]
            tile_size=rgb.shape[1::-1]
            for prompt in case['prompts']:
                transform, strip_size, box=strip_geometry(prompt['segment'],case['box'])
                points=np.asarray(prompt['points'],dtype=np.float32)
                local=(points@transform[:,:2].T+transform[:,2]).astype(np.float32)
                assert np.allclose((local-transform[:,2])@transform[:,:2],points,atol=1e-4)
                assert np.all((local>=box[:2]-.01)&(local<=box[2:]+.01))
                strip=cv2.warpAffine(rgb,transform,strip_size,flags=cv2.INTER_LINEAR)
                with torch.inference_mode():
                    sam.predictor.set_image(strip)
                    predicted,scores,_=sam.predictor.predict(box=box,point_coords=local,
                        point_labels=np.ones(len(local),dtype=np.int32),multimask_output=False)
                assert predicted.shape==(1,*strip.shape[:2])
                mask=back_project(predicted[0],transform,tile_size)
                support=support_at(mask,points)
                kept=all(support)
                prompt['strip']=dict(matrix=transform.tolist(),size=list(strip_size),box=box.tolist(),
                    points=local.tolist(),area=int(mask.sum()),score=float(np.ravel(scores)[0]),
                    point_support=support,kept=kept)
                if kept:
                    masks[prompt['key']]=mask
        save_json(folder/'prompts.json',records)
        np.savez_compressed(folder/'accepted_masks.npz',**masks)
        print('STRIP',stem[:8],sum(len(c['prompts']) for c in records),len(masks),flush=True)
    torch.cuda.synchronize()
    save_json(OUT/'runtime.json',dict(seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    print('INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':
    infer()
