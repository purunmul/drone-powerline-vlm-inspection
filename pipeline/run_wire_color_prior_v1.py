from repository_paths import image_path
import argparse
import base64
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from run_vlm_zoom_recheck_round1_v1 import BASE, STEMS, baseline_path, digest, input_paths, read_json

NAME = 'wire_color_prior_20261108_v3'
OUT = BASE / NAME
PALETTE = {
    'brown': (111, 77, 56), 'black': (42, 42, 42), 'gray': (133, 133, 133),
    'blue': (51, 96, 153), 'green': (58, 123, 70), 'yellow': (194, 165, 63),
}
STANDARD = ('Use this background information before judging: Korean conductor identification '
            'colors include brown, black, gray, blue and green-yellow. They can look lighter, '
            'darker or less saturated in a photo, and on bare wire color marking can be at ends. '
            'Treat this as a weak hint. ')
QUESTION = ('Compare the two panels. The right panel marks one short path with two cyan circles. '
            'Decide whether the path between those markers is an overhead electrical wire. '
            'Use the original left panel to check the object. Answer with exactly one word. '
            'WIRE means clear overhead wire. BACKGROUND means a clearly different object. '
            'UNSURE means the marked path cannot be identified. EMPTY means there are no cyan circles.')
POLICIES = ('baseline', 'expanded_standard', 'photo_auto', 'standard_and_photo_auto',
            'photo_qwen', 'standard_and_photo_qwen')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def inputs(stem):
    _, folder = input_paths(stem)
    record = read_json(folder / 'candidates.json')
    source = image_path(record['image'])
    assert digest(source) == record['image_sha256'], stem
    return source, [folder / item['mask'] for item in record['candidates']]


def segments(mask, index):
    import cv2
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    from skimage.morphology import skeletonize

    x0, y0, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w == 0 or h == 0:
        return []
    crop = mask[y0:y0+h, x0:x0+w]
    line = skeletonize(crop)
    lines = cv2.HoughLinesP(line.astype(np.uint8), 1, np.pi/180, 32,
                            minLineLength=55, maxLineGap=5)
    if lines is None:
        return []
    points = np.argwhere(line)
    tree = cKDTree(points)
    widths = distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1]
    candidates = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = float(np.hypot(x2-x1, y2-y1))
        count = max(25, round(length/3))
        sampled = np.column_stack((np.linspace(y1, y2, count), np.linspace(x1, x2, count)))
        distance, indices = tree.query(sampled)
        near = points[indices]
        if np.mean(distance <= 2) < .90 or np.mean(widths[tuple(near.T)] <= 5) < .80:
            continue
        near = near[distance <= 2]
        midpoint = near[len(near)//2]
        # Retain only the central 140 px for the review crop and color sampling.
        local = near[np.linalg.norm(near-midpoint, axis=1) <= 70] + (y0, x0)
        candidates.append(dict(mask_index=index, length=length, points=local.tolist(),
                               center=(midpoint+(y0,x0)).tolist()))
    return candidates


def prepare():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from evaluate_first_image_v1 import readmask

    OUT.mkdir(exist_ok=False)
    boards = OUT / 'review_inputs'
    boards.mkdir()
    manifest = []
    for stem in STEMS:
        source, masks = inputs(stem)
        image = Image.open(source).convert('RGB')
        pool = [entry for index, path in enumerate(masks) for entry in segments(readmask(path), index)]
        pool.sort(key=lambda entry: -min(entry['length'], 200))
        picked = []
        count_by_mask = {}
        for item in pool:
            center = np.asarray(item['center'])
            index = item['mask_index']
            if count_by_mask.get(index, 0) >= 2:
                continue
            if any(np.linalg.norm(center-other['center']) < 145 for other in picked):
                continue
            picked.append(item)
            count_by_mask[index] = count_by_mask.get(index, 0) + 1
            if len(picked) == 8:
                break
        assert len(picked) >= 2, stem
        for index, item in enumerate(picked + [dict(center=picked[0]['center'], points=[])], start=1):
            target_id = 'CTRL' if not item['points'] else f'S{index:02d}'
            cy, cx = item['center']
            x0 = max(0, min(cx-144, image.width-288))
            y0 = max(0, min(cy-120, image.height-240))
            crop = image.crop((x0,y0,x0+288,y0+240))
            marked = crop.copy()
            draw = ImageDraw.Draw(marked)
            for y, x in item['points'][::max(1,len(item['points'])-1)]:
                u, v = x-x0, y-y0
                draw.ellipse((u-5,v-5,u+5,v+5),outline=(0,245,245),width=2)
            board = Image.new('RGB',(576,264),(22,26,32))
            board.paste(crop,(0,24))
            board.paste(marked,(288,24))
            draw = ImageDraw.Draw(board)
            font = ImageFont.load_default(size=15)
            draw.text((8,2),'ORIGINAL',fill='white',font=font)
            draw.text((296,2),'TARGET',fill='white',font=font)
            path = boards / f'{stem}_{target_id}.png'
            board.save(path)
            manifest.append(dict(stem=stem, id=target_id, path=str(path), **item))
        print('PREPARED', stem, len(picked), 'anchors + control', flush=True)
    write_json(OUT / 'anchors.json', manifest)


def review():
    from vllm import LLM, SamplingParams

    target = OUT / 'qwen_responses.json'
    if target.exists():
        raise FileExistsError(target)
    manifest = read_json(OUT / 'anchors.json')
    settings = read_json(BASE / 'vlm_indexed_context_neutral_20261108_v1' / 'run_config.json')['settings']
    settings = dict(settings)
    settings.pop('quantization', None)
    settings.pop('load_format', None)
    settings.update(dtype='float16', gpu_memory_utilization=.65, max_num_seqs=2)
    settings['mm_processor_kwargs'] = dict(min_pixels=784, max_pixels=200704)
    options = dict(temperature=0, max_tokens=12)
    write_json(OUT / 'qwen_config.json', dict(settings=settings, options=options, plain=QUESTION,
                                             prior=STANDARD + QUESTION))
    llm = LLM(**settings)
    params = SamplingParams(**options)
    jobs = [(mode, item) for mode in ('photo', 'standard') for item in manifest]
    responses = []
    for start in range(0, len(jobs), 4):
        batch = jobs[start:start+4]
        messages = []
        for mode, item in batch:
            uri = 'data:image/png;base64,' + base64.b64encode(Path(item['path']).read_bytes()).decode('ascii')
            question = (STANDARD if mode == 'standard' else '') + QUESTION
            messages.append([{'role':'user','content':[{'type':'image_url','image_url':{'url':uri}},
                                                       {'type':'text','text':question}]}])
        for (mode, item), result in zip(batch, llm.chat(messages, params, use_tqdm=False), strict=True):
            raw = result.outputs[0].text
            label = {'WIRE':'wire','BACKGROUND':'background','UNSURE':'uncertain',
                     'EMPTY':'no_target'}.get(raw.strip().rstrip('.').upper())
            responses.append(dict(mode=mode, stem=item['stem'], id=item['id'], raw=raw, label=label))
        print('QWEN', len(responses), '/', len(jobs), flush=True)
    write_json(target, responses)


def palette_profiles():
    import cv2
    import numpy as np

    samples = np.array(list(PALETTE.values()), dtype=np.float32).reshape(1,-1,3)/255
    centers = cv2.cvtColor(samples, cv2.COLOR_RGB2LAB)[0]
    # Lab radii cover shade/brightening and lower or higher chroma than the swatch.
    radii = [(33,24,28), (34,18,18), (27,21,21), (35,24,30), (35,29,26), (35,25,32)]
    return [(center, np.array(radius, dtype=np.float32)) for center, radius in zip(centers,radii,strict=True)]


def photo_profiles(stem, mode, lab, anchors, responses):
    import numpy as np

    labels = {item['id']: item['label'] for item in responses
              if item['stem'] == stem and item['mode'] == mode}
    selected = [item for item in anchors if item['stem'] == stem and labels.get(item['id']) == 'wire'
                and item['id'] != 'CTRL']
    # An unmarked crop described as a wire, or insufficient anchors, disables photo pruning.
    usable = labels.get('CTRL') == 'no_target' and len(selected) >= 2
    profiles = profile_from_anchors(lab, selected) if usable else []
    audit = dict(control=labels.get('CTRL'), usable=usable, anchors=[item['id'] for item in selected],
                 profiles=[dict(center=center.tolist(),radius=radius.tolist()) for center,radius in profiles])
    return profiles, audit


def profile_from_anchors(lab, selected):
    import numpy as np

    profiles = []
    for anchor in selected:
        y,x = np.unique(anchor['points'],axis=0).T
        pixels = lab[y,x]
        center = np.median(pixels,axis=0)
        mad = np.median(np.abs(pixels-center),axis=0)
        radius = np.clip(3.0*1.4826*mad,(16,9,9),(34,22,22)).astype(np.float32)
        profiles.append((center,radius))
    return profiles


def auto_profiles(stem, lab, anchors):
    import numpy as np

    scored = []
    for item in anchors:
        if item['stem'] != stem or item['id'] == 'CTRL' or item['length'] < 75:
            continue
        points = np.asarray(item['points'])
        delta = points[-1]-points[0]
        norm = np.linalg.norm(delta)
        if norm < 30:
            continue
        normal = np.rint(5*np.array([-delta[1],delta[0]])/norm).astype(int)
        valid = ((points >= 5) & (points < np.array(lab.shape[:2])-5)).all(axis=1)
        y,x = points[valid].T
        if len(y) < 12:
            continue
        left = lab[y+normal[0],x+normal[1],0]-lab[y,x,0]
        right = lab[y-normal[0],x-normal[1],0]-lab[y,x,0]
        support = float(np.mean(((left >= 4) & (right >= 4)) | ((left <= -4) & (right <= -4))))
        if support >= .55:
            scored.append((support,item))
    selected = []
    masks = set()
    for support,item in sorted(scored,key=lambda entry:-entry[0]):
        if item['mask_index'] in masks:
            continue
        masks.add(item['mask_index'])
        selected.append(item)
        if len(selected) == 4:
            break
    profiles = profile_from_anchors(lab, selected) if len(selected) >= 2 else []
    return profiles, dict(anchors=[x['id'] for x in selected], usable=bool(profiles),
                          support={item['id']:score for score,item in scored},
                          profiles=[dict(center=c.tolist(),radius=r.tolist()) for c,r in profiles])


def rejected_pixels(lab, profiles):
    import cv2
    import numpy as np

    if not profiles:
        return np.zeros(lab.shape[:2], dtype=bool)
    best = np.full(lab.shape[:2], np.inf, dtype=np.float32)
    for center, radius in profiles:
        delta = (lab-center)/radius
        squared = np.sum(delta*delta, axis=2)
        np.minimum(best,squared,out=best)
    outside = best > 1
    return outside & (cv2.blur(outside.astype(np.float32), (5,5)) >= .80)


def overlay(photo, layers, width=5):
    import cv2
    from PIL import Image

    result = photo.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width,width))
    for line,color in layers:
        result[cv2.dilate(line.astype('uint8'),kernel) > 0] = color
    return Image.fromarray(result)


def save_views(stem, photo, variants):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    folder = OUT / stem
    folder.mkdir(exist_ok=True)
    before, after = variants['baseline'], variants['standard_and_photo_auto']
    cyan, magenta, orange = (0,252,231), (254,38,158), (255,155,35)
    overlay(photo, [(after,cyan)]).save(folder/'standard_and_photo_auto_on_original.jpg', quality=92)
    photo_only = variants['photo_auto']
    overlay(photo, [(photo_only,cyan)]).save(folder/'photo_auto_on_original.jpg', quality=92)
    overlay(photo, [(before & ~photo_only,magenta),(photo_only & ~before,orange)],7).resize(
        (1600,900),Image.Resampling.LANCZOS).save(folder/'photo_auto_changes.jpg',quality=93)
    panels = [
        ('Initial LangSAM', overlay(photo,[(before,cyan)],7)),
        ('Expanded standard only', overlay(photo,[(variants['expanded_standard'],cyan)],7)),
        ('Photo only (automatic anchors)', overlay(photo,[(variants['photo_auto'],cyan)],7)),
        ('Expanded standard + photo (automatic anchors)', overlay(photo,[(after,cyan)],7)),
    ]
    width, height, header = 760, 428, 28
    board = Image.new('RGB',(width*2,(height+header)*2),(21,25,31))
    font = ImageFont.load_default(size=18)
    draw = ImageDraw.Draw(board)
    for index, (title, image) in enumerate(panels):
        x,y = index%2*width,index//2*(height+header)
        draw.text((x+9,y+4),title,fill='white',font=font)
        board.paste(image.resize((width,height),Image.Resampling.LANCZOS),(x,y+header))
    board.save(folder/'comparison.jpg',quality=93)
    overlay(photo, [(before & ~after,magenta),(after & ~before,orange)],7).resize(
        (1600,900),Image.Resampling.LANCZOS).save(folder/'changes_magenta_removed_orange_added.jpg',quality=93)
    for method,line in variants.items():
        Image.fromarray(line.astype(np.uint8)*255).save(folder/f'{method}_centerline.png')


def predict(stem, anchors, responses):
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.morphology import skeletonize
    from evaluate_first_image_v1 import readmask

    source, paths = inputs(stem)
    photo = np.asarray(Image.open(source).convert('RGB'))
    lab = cv2.cvtColor(photo.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    palette = palette_profiles()
    plain, plain_audit = photo_profiles(stem,'photo',lab,anchors,responses)
    prior, prior_audit = photo_profiles(stem,'standard',lab,anchors,responses)
    automatic, auto_audit = auto_profiles(stem,lab,anchors)
    # Combined condition allows either expanded standard or photo-derived range.
    profiles = dict(baseline=[],expanded_standard=palette,photo_qwen=plain,
                    standard_and_photo_qwen=palette+prior,photo_auto=automatic,
                    standard_and_photo_auto=palette+automatic)
    rejects = {key: rejected_pixels(lab,value) for key,value in profiles.items()}
    variants = {key: np.zeros(photo.shape[:2],dtype=bool) for key in POLICIES}
    before_area = 0
    removed_area = dict.fromkeys(POLICIES,0)
    for path in paths:
        mask = readmask(path)
        before_area += int(mask.sum())
        for name, reject in rejects.items():
            refined = mask & ~reject
            removed_area[name] += int((mask & reject).sum())
            variants[name] |= skeletonize(refined)
    assert np.array_equal(variants['baseline'], readmask(baseline_path(stem))), stem
    save_views(stem, photo, variants)
    print('PREDICTED', stem, 'Qwen anchors', len(plain), len(prior), 'removed mask pixels', removed_area, flush=True)
    return variants, dict(photo=plain_audit, standard=prior_audit, automatic=auto_audit, mask_area=before_area,
                          removed_mask_pixels=removed_area)


def evaluate():
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from evaluate_first_image_v1 import metrics, readmask

    if (OUT/'summary.json').exists():
        raise FileExistsError(OUT/'summary.json')
    anchors = read_json(OUT/'anchors.json')
    responses = read_json(OUT/'qwen_responses.json')
    assert len(responses) == len(anchors)*2
    predictions = {}
    audit = {}
    for stem in STEMS:
        predictions[stem],audit[stem] = predict(stem,anchors,responses)
    # All candidate filtering and rendering is complete before reading annotation-derived rasters.
    rows = []
    for stem,variants in predictions.items():
        reference_folder = (BASE/'diagnostic_eval'/f'{stem}_v1' if stem == STEMS[0]
                            else BASE/'dev_transfer_20261107_v1'/stem/'evaluation')
        ref = readmask(reference_folder/'reference_centerlines.png')
        refdist = distance_transform_edt(~ref)
        before = variants['baseline']
        base_covered = distance_transform_edt(~before)[ref] <= 5
        for method,line in variants.items():
            scores,dist = metrics(line,ref,refdist)
            score = next(row for row in scores if row['tolerance_pixels'] == 5)
            removed = before & ~line
            added = line & ~before
            covered = dist[ref] <= 5 if dist is not None else np.zeros_like(base_covered)
            rows.append(dict(stem=stem,method=method,**score,
                             removed_off_reference=int((removed & (refdist>5)).sum()),
                             removed_near_reference=int((removed & (refdist<=5)).sum()),
                             added_centerline=int(added.sum()),
                             lost_reference=int((base_covered & ~covered).sum()),
                             gained_reference=int((~base_covered & covered).sum())))
    with (OUT/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ('centerline_precision','centerline_recall','centerline_f1')
    macro = {method:{k:float(np.mean([row[k] for row in rows if row['method']==method])) for k in keys}
             for method in POLICIES}
    write_json(OUT/'summary.json',dict(macro=macro,audit=audit,palette_srgb=PALETTE,
                                      tolerance=5,stage='first LangSAM masks; no zoom'))
    (OUT/'README.txt').write_text(
        'Three development frames. First LangSAM masks only; no zoom added.\n'
        'Palette sRGB values are illustrative engineering choices, not KEC numeric swatches.\n'
        'Qwen reviews identical original/marked crop pairs with and without KEC color guidance.\n'
        'At least two Qwen wire anchors and a passing unmarked control are needed for Qwen profiles.\n'
        'Automatic anchors use narrow segments with consistent contrast on both sides, from at least two masks.\n'
        'Combined acceptance is the union of expanded standard and photo-specific Lab ranges.\n'
        'Mask pixels are removed only outside all applicable ranges with >=80% agreement in a 5x5 neighborhood.\n'
        'The refined masks are re-skeletonized. Magenta marks removed centerline pixels and orange marks new ones.\n'
        'Annotation-derived raster used for evaluation only. No training or manual point prompts.\n')
    archive_path = BASE / f'{NAME}.zip'
    with ZipFile(archive_path,'x',ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and 'review_inputs' not in path.parts:
                archive.write(path,Path(NAME)/path.relative_to(OUT))
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    for method,values in macro.items():
        print(method,' '.join(f'{100*value:.2f}' for value in values.values()))
    print('VERIFIED', archive_path,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','review','evaluate'))
    {'prepare':prepare,'review':review,'evaluate':evaluate}[parser.parse_args().stage]()
