import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
import zipfile
from pathlib import Path

OUT = Path('results/mixed_sam_repair_20261107_v1')
TRANSFER = Path('results/dev_transfer_20261107_v1')
ABLATION = Path('results/skeleton_order_ablation_20261107_v1')
CHECKPOINT = Path('.cache/torch/hub/checkpoints/sam2.1_hiera_small.pt')
CHECKPOINT_SHA = '6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
# Coordinates below are relative to the candidate context crop. They were chosen
# from unannotated source pixels and locked before SAM or reference evaluation.
PROMPTS = [
    dict(image='PUYL5573_frame_000165', candidate='P15', origin=[1383,640],
         crop=[370,405,1090,723], box=[375,422,1080,711],
         positive=[[430,449],[600,521],[790,597],[1020,683]],
         negative=[[610,548],[730,470],[565,574]]),
    dict(image='KZIV5445_frame_000488', candidate='P03', origin=[2172,0],
         crop=[115,440,435,900], box=[120,445,430,895],
         positive=[[397,490],[326,600],[246,720],[154,850]],
         negative=[[225,600],[270,775],[208,720]]),
    dict(image='KZIV5445_frame_000488', candidate='P21', origin=[1383,615],
         crop=[80,711,480,1210], box=[87,715,472,1205],
         positive=[[442,750],[365,850],[251,1000],[135,1150]],
         negative=[[275,1000],[385,900],[125,1030]]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def key(prompt):
    return prompt['image'].split('_')[0]+'_'+prompt['candidate']


def infer():
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from lang_sam.models.sam import SAM
    from run_selective_sam_round1 import overlay

    OUT.mkdir(exist_ok=False)
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    save_json(OUT/'run_config.json', {
        'model': 'sam2.1_hiera_small', 'checkpoint_sha256': CHECKPOINT_SHA,
        'cases_reference_informed': True, 'point_source': 'assistant visually inspected unannotated original pixels',
        'human_assisted': True, 'reference_used_for_prompting': False,
        'scope': 'one visible strand segment per specified crop',
        'selection': 'highest predicted SAM score among all multimask outputs, before reference evaluation',
        'point_gate': 'all positive points inside and zero negative points inside; this is only a prompt check',
        'versions': {p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','Pillow']},
        'prompts_context_xy': PROMPTS,
    })
    started = time.perf_counter()
    sam = SAM()
    sam.build_model('sam2.1_hiera_small', ckpt_path=str(CHECKPOINT))
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    print('SAM_READY',load_seconds,flush=True)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for prompt in PROMPTS:
        name, ident = prompt['image'], prompt['candidate']
        src = TRANSFER/name/'candidates'
        data = read_json(src/'candidates.json')
        candidate = next(c for c in data['candidates'] if c['id']==ident)
        ox, oy = prompt['origin']
        assert candidate['context_xyxy_original'][:2] == [ox,oy]
        image_path = Path(data['image'])
        assert sha(image_path) == data['image_sha256']
        image = Image.open(image_path).convert('RGB')
        offset = np.array([ox,oy,ox,oy])
        roi = (np.array(prompt['crop'])+offset).tolist()
        x0,y0,x1,y1 = roi
        crop = image.crop(roi)
        assert crop.size == (x1-x0,y1-y0)
        xy = np.array(prompt['positive']+prompt['negative'],dtype=np.float32)
        xy -= np.array(prompt['crop'][:2],dtype=np.float32)
        labels = np.array([1]*len(prompt['positive'])+[0]*len(prompt['negative']),dtype=np.int32)
        box = np.array(prompt['box'],dtype=np.float32)-np.array([*prompt['crop'][:2]]*2,dtype=np.float32)
        assert ((xy>=0)&(xy<np.array(crop.size))).all()
        assert (box[:2]>=0).all() and (box[2:]<=np.array(crop.size)).all()
        mask_path = src/candidate['mask']
        mask_sha = sha(mask_path)
        old = np.asarray(Image.open(mask_path).convert('L').crop(roi))>0
        start = time.perf_counter()
        with torch.inference_mode():
            sam.predictor.set_image(np.asarray(crop))
            masks,scores,logits = sam.predictor.predict(point_coords=xy,point_labels=labels,box=box,multimask_output=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        masks,scores = np.asarray(masks,dtype=bool),np.asarray(scores)
        assert len(masks)==len(scores)==3 and masks.shape[1:]==old.shape
        choice = int(scores.argmax())
        new = masks[choice]
        folder = OUT/key(prompt)
        folder.mkdir()
        np.savez_compressed(folder/'sam_alternatives.npz',masks=masks,scores=scores,logits=logits)
        for label,mask in [('before',old),('after',new)]:
            Image.fromarray(mask.astype('uint8')*255).save(folder/(label+'_mask.png'))
        crop.save(folder/'original.png')
        panel = crop.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle(tuple(box),outline='yellow',width=2)
        for (x,y),label in zip(xy,labels):
            draw.ellipse((x-4,y-4,x+4,y+4),fill='lime' if label else 'red')
        panel.save(folder/'prompts.png')
        tiles = [panel,overlay(crop,old,(255,0,0)),overlay(crop,new,(0,220,255))]
        titles = ['Prompt points and box','Original candidate mask','Selected SAM mask']
        board = Image.new('RGB',(crop.width*3,crop.height+32),'white')
        draw = ImageDraw.Draw(board)
        for index,(tile,title) in enumerate(zip(tiles,titles,strict=True)):
            board.paste(tile,(index*crop.width,32))
            draw.text((index*crop.width+5,5),title,fill='black')
        board.save(folder/'comparison.jpg',quality=95)
        hit = np.array([bool(new[int(y),int(x)]) for x,y in xy])
        record = {
            'image':name,'candidate':ident,'key':key(prompt),'roi_original_xyxy':roi,
            'prompt_box_original_xyxy':(np.array(prompt['box'])+offset).tolist(),
            'positive_original_xy':(np.array(prompt['positive'])+[ox,oy]).tolist(),
            'negative_original_xy':(np.array(prompt['negative'])+[ox,oy]).tolist(),
            'source_mask_sha256':mask_sha,'source_image_sha256':data['image_sha256'],
            'before_area':int(old.sum()),'after_area':int(new.sum()),
            'positive_hits':int(hit[labels==1].sum()),'negative_hits':int(hit[labels==0].sum()),
            'positive_count':int((labels==1).sum()),'negative_count':int((labels==0).sum()),
            'all_prompt_constraints_met':bool(hit[labels==1].all() and not hit[labels==0].any()),
            'selected_index':choice,'scores':scores.tolist(),'inference_seconds':elapsed,
        }
        assert sha(mask_path)==mask_sha
        records.append(record)
        save_json(folder/'inference.json',record)
        print('SAM_CASE',record['key'],record['before_area'],record['after_area'],record['positive_hits'],record['negative_hits'],flush=True)
    save_json(OUT/'inference.json',{'cases':records,'model_load_seconds':load_seconds,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})


def evaluate():
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    from evaluate_first_image_v1 import metrics, readmask, trace

    summary = read_json(OUT/'inference.json')
    rows = []
    for case in summary['cases']:
        name, ident = case['image'],case['candidate']
        folder = OUT/case['key']
        x0,y0,x1,y1 = case['roi_original_xyxy']
        region = np.s_[y0:y1,x0:x1]
        src = TRANSFER/name/'candidates'
        candidate = next(c for c in read_json(src/'candidates.json')['candidates'] if c['id']==ident)
        assert sha(src/candidate['mask'])==case['source_mask_sha256']
        old = readmask(folder/'before_mask.png')
        new = readmask(folder/'after_mask.png')
        assert np.array_equal(old,readmask(src/candidate['mask'])[region])
        before,after = skeletonize(old),skeletonize(new)
        assert before.shape == after.shape == (y1-y0,x1-x0)
        reference = readmask(TRANSFER/name/'evaluation/reference_centerlines.png')
        local_ref = reference[region]
        assert local_ref.any()
        local_dist = distance_transform_edt(~local_ref)
        global_dist = distance_transform_edt(~reference)
        baseline = readmask(ABLATION/name/'B_vlm_keep_only/skeleton_then_union/centerline.png')
        raw_add = baseline.copy()
        raw_add[region] |= before
        repaired_add = baseline.copy()
        repaired_add[region] |= after
        comparisons = {
            'local_before':metrics(before,local_ref,local_dist)[0],
            'local_after':metrics(after,local_ref,local_dist)[0],
            'global_strict_keep':metrics(baseline,reference,global_dist)[0],
            'global_add_original_crop':metrics(raw_add,reference,global_dist)[0],
            'global_add_repaired_crop':metrics(repaired_add,reference,global_dist)[0],
        }
        for condition,values in comparisons.items():
            rows.extend(dict(r,case=case['key'],condition=condition) for r in values)
        case['metrics'] = comparisons
        for label,line in [('before',before),('after',after)]:
            Image.fromarray(line.astype('uint8')*255).save(folder/(label+'_centerline.png'))
            paths = trace(line)
            for path in paths:
                path['points_xy'] = [[x+x0,y+y0] for x,y in path['points_xy']]
            save_json(folder/(label+'_centerline_original_xy.json'),{'roi_original_xyxy':case['roi_original_xyxy'],'paths':paths})
        print('EVAL_CASE',case['key'],{k:[round(v[1][f'centerline_{m}'],4) for m in ['precision','recall','f1']] for k,v in comparisons.items()},flush=True)
    with (OUT/'metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary.update(status='MIXED_REPAIR_DIAGNOSTIC_COMPLETE',heldout_used=False,applied_to_pipeline=False,
                   comparison_scope='Local scores use all annotated strands inside the same crop. Global addbacks are independent hypothetical comparisons against the fixed VLM keep subset. Each addback is confined to the same crop.')
    save_json(OUT/'summary.json',summary)
    with zipfile.ZipFile('mixed_sam_repair_20261107_v1.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.rglob('*'):
            if path.is_file():
                archive.write(path,path.relative_to(OUT.parent))
        archive.write(__file__,Path(__file__).name)
    print('MIXED_REPAIR_DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['infer','evaluate'])
    (infer if parser.parse_args().phase=='infer' else evaluate)()
