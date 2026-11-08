from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from run_sam_auto_points_20261108_v1 import BASE,STEMS,digest,input_paths,read_json,save_json

OUT=BASE/'langsam_ridge_paths_20261108_v1'
ANGLES=tuple(range(0,180,15))
WIDTHS=(2,4,6)
MIN_RIDGE=5.0
MIN_LENGTH=44
SEED_LIMIT=80
MARGIN=24
OFFSETS=np.arange(-8,9,dtype=np.float32)
SAM_BONUS=2.0
POLICIES={'box_photo':0.0,'box_photo_sam':SAM_BONUS}


def shifted(image, dx, dy):
    h,w=image.shape
    matrix=np.array([[1,0,dx],[0,1,dy]],dtype=np.float32)
    return cv2.warpAffine(image,matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT_101)


def bilateral_ridge(center, left, right):
    dark=np.minimum(left-center,right-center)
    bright=np.minimum(center-left,center-right)
    return np.maximum(dark,bright).clip(0,None)


def seed_segments(gray, inside):
    ridge=np.zeros_like(gray)
    for angle in ANGLES:
        rad=np.deg2rad(angle)
        nx,ny=-np.sin(rad),np.cos(rad)
        score=np.maximum.reduce([bilateral_ridge(gray,shifted(gray,nx*d,ny*d),
                                                shifted(gray,-nx*d,-ny*d)) for d in WIDTHS])
        peak=(score>=shifted(score,nx,ny)) & (score>=shifted(score,-nx,-ny))
        np.maximum(ridge,np.where(peak,score,0),out=ridge)
    edges=np.ascontiguousarray(((ridge>=MIN_RIDGE)&inside).astype(np.uint8)*255)
    found=cv2.HoughLinesP(edges,1,np.pi/180,threshold=20,minLineLength=MIN_LENGTH,maxLineGap=9)
    if found is None:
        return []
    lines=found.reshape(-1,2,2)
    lengths=np.linalg.norm(lines[:,1]-lines[:,0],axis=1)
    return [lines[i] for i in np.argsort(-lengths,kind='stable')[:SEED_LIMIT]]


def sample(image,x,y,interpolation=cv2.INTER_LINEAR):
    return cv2.remap(image,np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),
                     interpolation,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def make_lattice(gray, inside, sam_near, segment):
    delta=segment[1]-segment[0]
    length=float(np.linalg.norm(delta))
    direction=delta/length
    normal=np.array([-direction[1],direction[0]])
    steps=np.arange(-MARGIN,length+MARGIN+1,2)
    grid=segment[0]+steps[:,None,None]*direction+OFFSETS[None,:,None]*normal
    x,y=grid[:,:,0],grid[:,:,1]
    middle=sample(gray,x,y)
    strength=np.maximum.reduce([bilateral_ridge(middle,sample(gray,x+normal[0]*d,y+normal[1]*d),
                                               sample(gray,x-normal[0]*d,y-normal[1]*d)) for d in WIDTHS])
    valid=sample(inside,x,y,cv2.INTER_NEAREST)>0
    near=sample(sam_near,x,y,cv2.INTER_NEAREST)>0
    return grid,strength,valid,near


def trace_lattice(grid,strength,valid,near,bonus):
    signal=np.minimum(strength,20)-MIN_RIDGE-.15*np.abs(OFFSETS)
    signal+=bonus*(near & (strength>=MIN_RIDGE))
    signal[~valid]=-1e4
    n,width=signal.shape
    scores=np.zeros_like(signal)
    parent=np.full((n,width),-1,dtype=np.int8)
    offsets=np.arange(width)
    for i,row in enumerate(signal):
        previous=scores[i-1] if i else np.zeros(width,dtype=np.float32)
        left=np.r_[-np.inf,previous[:-1]-1.8]
        right=np.r_[previous[1:]-1.8,-np.inf]
        options=np.stack([np.zeros(width),left,previous,right])
        best=options.argmax(axis=0)
        scores[i]=np.maximum(row+options[best,offsets],0)
        parent[i]=np.array([-1,-1,0,1],dtype=np.int8)[best]
        parent[i,best==0]=127
    i,j=np.unravel_index(scores.argmax(),scores.shape)
    route=[]
    while i>=0 and 0<=j<width and scores[i,j]>0:
        route.append((i,j))
        move=parent[i,j]
        if move==127:
            break
        i-=1
        j+=move
    if len(route)<MIN_LENGTH//2+1:
        return None
    indexes=np.array(route[::-1])
    a,b=indexes.T
    response=strength[a,b]
    supported=response>=MIN_RIDGE
    good=np.flatnonzero(supported)
    if len(good)==0:
        return None
    # Split at unsupported spans longer than 6 px, then retain the longest span.
    spans=np.split(good,np.flatnonzero(np.diff(good)>4)+1)
    chosen=max(spans,key=lambda s:s[-1]-s[0])
    start,end=int(chosen[0]),int(chosen[-1])+1
    if end-start<MIN_LENGTH//2+1 or supported[start:end].mean()<.70:
        return None
    xy=np.rint(grid[a[start:end],b[start:end]]).astype(np.int32)
    return xy,float(response[start:end].mean()),float(near[a[start:end],b[start:end]].mean())


def path_overlap(xy, occupied):
    x,y=xy.T
    return float(occupied[y,x].mean())


def main():
    cv2.setNumThreads(2)
    cv2.setRNGSeed(20261108)
    OUT.mkdir(exist_ok=False)
    save_json(OUT/'config.json',dict(images=STEMS,angles=ANGLES,ridge_half_widths_px=WIDTHS,
        minimum_ridge_graylevels=MIN_RIDGE,min_length_px=MIN_LENGTH,max_hough_seeds_per_box=SEED_LIMIT,
        path_transverse_radius_px=8,along_track_step_px=2,endpoint_search_margin_px=MARGIN,
        sam_proximity_px=5,sam_score_bonus=SAM_BONUS,maximum_unsupported_gap_px=6,
        intra_box_path_dedup_radius_px=2,whole_image_dedup_radius_px=2,
        paths_must_stay_in_box=True,inference_inputs='RGB JPEGs and frozen LangSAM representative boxes/masks',
        annotation_access='evaluation only',development_only=True))
    started=time.perf_counter()
    for stem in STEMS:
        tile_dir,candidate_dir=input_paths(stem)
        raw=read_json(tile_dir/'result.json')
        by_id={c['id']:c for c in raw['candidates']}
        tiles={t['tile_id']:t['crop_xyxy_original'] for t in raw['tiles']}
        organized=read_json(candidate_dir/'candidates.json')['candidates']
        assert len(organized)>0
        image_path=Path(raw['image'])
        assert digest(image_path)==raw['sha256']
        rgb=np.array(Image.open(image_path).convert('RGB'))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray=cv2.GaussianBlur(gray,(0,0),.70)
        h,w=gray.shape
        prior_masks={}
        candidates={key:[] for key in POLICIES}
        box_stats=[]
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        for c in organized:
            entry=by_id[c['representative_raw_id']]
            tx,ty,tx1,ty1=tiles[entry['tile_id']]
            bx,by,bx1,by1=entry['box_xyxy_original']
            l,t,r,b=max(tx,int(np.ceil(bx))),max(ty,int(np.ceil(by))),min(tx1,int(np.floor(bx1))+1),min(ty1,int(np.floor(by1))+1)
            stat=dict(id=c['id'],raw_id=entry['id'],tile=entry['tile_id'],box=[l,t,r,b],seeds=0)
            box_stats.append(stat)
            if r-l<8 or b-t<8:
                continue
            x0,y0,x1,y1=max(0,l-8),max(0,t-8),min(w,r+8),min(h,b+8)
            crop=np.ascontiguousarray(gray[y0:y1,x0:x1])
            inside=np.zeros(crop.shape,dtype=np.uint8)
            inside[t-y0:b-y0,l-x0:r-x0]=1
            source=candidate_dir/c['mask']
            raw_mask=np.asarray(Image.open(source).convert('L').crop((x0,y0,x1,y1)))>0
            sam_near=cv2.dilate(raw_mask.astype(np.uint8),kernel)
            seeds=seed_segments(crop,inside)
            stat['seeds']=len(seeds)
            box_paths={key:[] for key in POLICIES}
            for seed in seeds:
                lattice=make_lattice(crop,inside,sam_near,seed)
                for key,bonus in POLICIES.items():
                    result=trace_lattice(*lattice,bonus)
                    if result is not None:
                        xy,strength,mask_fraction=result
                        if np.all(inside[xy[:,1],xy[:,0]]):
                            box_paths[key].append((xy,strength,mask_fraction))
            for key,paths in box_paths.items():
                occupied=np.zeros(crop.shape,dtype=np.uint8)
                for xy,strength,mask_fraction in sorted(paths,key=lambda p:len(p[0])*(p[1]+POLICIES[key]*p[2]),reverse=True):
                    if path_overlap(xy,occupied)>=.70:
                        continue
                    cv2.polylines(occupied,[xy],False,1,thickness=3,lineType=cv2.LINE_8)
                    original=xy+np.array([x0,y0],dtype=np.int32)
                    candidates[key].append(dict(points=original,id=c['id'],raw_id=entry['id'],
                        mean_ridge=strength,sam_near_fraction=mask_fraction))
        folder=OUT/stem
        folder.mkdir()
        for key,paths in candidates.items():
            line=np.zeros((h,w),dtype=np.uint8)
            occupied=np.zeros_like(line)
            output=[]
            for path in sorted(paths,key=lambda p:len(p['points'])*(p['mean_ridge']+POLICIES[key]*p['sam_near_fraction']),reverse=True):
                xy=path['points']
                if path_overlap(xy,occupied)>=.80:
                    continue
                cv2.polylines(line,[xy],False,255,1,lineType=cv2.LINE_8)
                cv2.polylines(occupied,[xy],False,1,thickness=3,lineType=cv2.LINE_8)
                output.append(dict(path,points=xy.tolist()))
            Image.fromarray(line).save(folder/f'{key}_centerline.png')
            save_json(folder/f'{key}_paths.json',output)
            print(stem[:8],key,len(output),int(np.count_nonzero(line)),flush=True)
        save_json(folder/'box_stats.json',box_stats)
    save_json(OUT/'runtime.json',dict(seconds=time.perf_counter()-started,compute='CPU'))
    print('TRACKING_COMPLETE',flush=True)

if __name__=='__main__':
    main()
