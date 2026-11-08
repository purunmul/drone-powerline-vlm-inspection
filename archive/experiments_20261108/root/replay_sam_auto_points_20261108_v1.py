from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np
from PIL import Image

BASE = Path('results')
OUT = BASE / 'sam_auto_points_20261108_replay_v1'
STEMS = ('BEGO7553_frame_001098','PUYL5573_frame_000165','KZIV5445_frame_000488')
CHECKPOINT = Path('.cache/torch/hub/checkpoints/sam2.1_hiera_small.pt')
CHECKPOINT_SHA = '6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
MIN_LENGTH = 36
MAX_SEGMENTS = 3
MAX_AREA_PER_LENGTH = 16
POINT_TOL = 3
FRACTIONS = np.array([.2, .5, .8],dtype=np.float32)[:,None]

def read_json(path):
    return json.loads(path.read_text())

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2),encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def input_paths(stem):
    if stem == STEMS[0]:
        root = BASE/'langsam_baseline'
        return root/f'{stem}_tiles1024_v1', root/f'{stem}_candidates_v1'
    root = BASE/'dev_transfer_20261107_v1'/stem
    return root/'tiles', root/'candidates'

def select_segments(gray, box, detector):
    h,w = gray.shape
    x0,y0 = np.maximum(np.ceil(box[:2]).astype(int), 0)
    x1,y1 = np.minimum(np.floor(box[2:]).astype(int)+1, (w,h))
    if x1-x0 < 8 or y1-y0 < 8:
        return []
    found = detector.detect(gray[y0:y1,x0:x1])[0]
    if found is None:
        return []
    segments = found.reshape(-1,2,2) + np.array([x0,y0],dtype=np.float32)
    lengths = np.linalg.norm(segments[:,1]-segments[:,0],axis=1)
    chosen = []
    for i in np.argsort(-lengths,kind='stable'):
        segment, length = segments[i], float(lengths[i])
        if length < MIN_LENGTH or len(chosen) == MAX_SEGMENTS:
            break
        direction = (segment[1]-segment[0])/length
        midpoint = segment.mean(axis=0)
        duplicate = False
        for previous, prev_length in chosen:
            axis = (previous[1]-previous[0])/prev_length
            normal = np.array([-axis[1],axis[0]])
            offset = midpoint - previous.mean(axis=0)
            duplicate |= (abs(direction@axis) > .98 and abs(offset@normal) < 7
                          and abs(offset@axis) < .45*(prev_length+length))
        if not duplicate:
            chosen.append((segment,length))
    return chosen

def infer():
    import torch
    from lang_sam.models.sam import SAM

    assert digest(CHECKPOINT) == CHECKPOINT_SHA and torch.cuda.is_available()
    OUT.mkdir(exist_ok=False)
    save_json(OUT/'config.json',dict(model='sam2.1_hiera_small',checkpoint_sha256=CHECKPOINT_SHA,
        cv2=cv2.__version__,torch=torch.__version__, min_length=MIN_LENGTH,max_segments_per_box=MAX_SEGMENTS,
        point_fractions=FRACTIONS.ravel().tolist(),point_tolerance=POINT_TOL,max_area_per_segment_length=MAX_AREA_PER_LENGTH,
        line_detector='OpenCV LSD_REFINE_STD on grayscale within original detector box',
        dedup='frozen baseline representatives; near-overlapping segments at <7px perpendicular midpoint distance suppressed within each box',
        output='single SAM mask per segment, using original tile and original box plus three positive points',
        policies=['box_only_saved','points_replace_with_box_fallback','points_supplement_box'],
        reference_used=False,manual_points=False,vlm_used=False))
    sam = SAM()
    sam.build_model('sam2.1_hiera_small',ckpt_path=str(CHECKPOINT))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    started = time.perf_counter()
    for stem in STEMS:
        raw_dir, organized_dir = input_paths(stem)
        report = read_json(raw_dir/'result.json')
        organized = read_json(organized_dir/'candidates.json')['candidates']
        reps = {c['representative_raw_id']:c['id'] for c in organized}
        assert len(reps)==len(organized)
        image_path = Path(report['image'])
        assert digest(image_path) == report['sha256']
        photo = Image.open(image_path).convert('RGB')
        grouped = {t['tile_id']: [] for t in report['tiles']}
        for c in report['candidates']:
            if c['id'] in reps:
                grouped[c['tile_id']].append(c)
        assert sum(map(len,grouped.values())) == len(reps)
        folder = OUT/stem
        folder.mkdir()
        records, accepted = [], {}
        for tile in report['tiles']:
            candidates = grouped[tile['tile_id']]
            if not candidates:
                continue
            roi = tile['crop_xyxy_original']
            rgb = np.asarray(photo.crop(roi))
            gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            with torch.inference_mode():
                sam.predictor.set_image(rgb)
                for c in candidates:
                    pid = reps[c['id']]
                    box = np.array(c['box_xyxy_local'],dtype=np.float32)
                    case = dict(id=pid,raw_id=c['id'],tile=tile['tile_id'],roi=roi,box=box.tolist(),prompts=[])
                    for i,(segment,length) in enumerate(select_segments(gray,box,detector),1):
                        points = segment[0]+FRACTIONS*(segment[1]-segment[0])
                        h,w=gray.shape
                        assert np.all((points >= 0) & (points < (w,h)))
                        assert np.all((points >= box[:2]) & (points <= box[2:]))
                        masks,scores,_ = sam.predictor.predict(box=box,point_coords=points,
                            point_labels=np.ones(3,dtype=np.int32),multimask_output=False)
                        assert masks.shape == (1,h,w) and np.size(scores)==1
                        mask = np.asarray(masks[0],dtype=bool)
                        area = int(mask.sum())
                        radius = POINT_TOL
                        supported = []
                        for x,y in np.rint(points).astype(int):
                            y0,y1=max(0,y-radius),min(h,y+radius+1)
                            x0,x1=max(0,x-radius),min(w,x+radius+1)
                            yy,xx=np.ogrid[y0:y1,x0:x1]
                            supported.append(bool((mask[y0:y1,x0:x1] & ((xx-x)**2+(yy-y)**2<=radius**2)).any()))
                        kept = all(supported) and area <= MAX_AREA_PER_LENGTH*length
                        key=f'{pid}_{i}'
                        if all(supported):
                            accepted[key]=mask
                        case['prompts'].append(dict(key=key,segment=segment.tolist(),length=length,
                            points=points.tolist(),sam_score=float(np.ravel(scores)[0]),area=area,
                            supported=supported,kept=kept))
                    records.append(case)
        save_json(folder/'prompts.json', records)
        np.savez_compressed(folder/'accepted_masks.npz',**accepted)
        print('INFER',stem[:8],len(records),sum(len(c['prompts']) for c in records),len(accepted),flush=True)
    torch.cuda.synchronize()
    save_json(OUT/'runtime.json',dict(seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    print('INFERENCE_COMPLETE',flush=True)

if __name__ == '__main__':
    infer()
