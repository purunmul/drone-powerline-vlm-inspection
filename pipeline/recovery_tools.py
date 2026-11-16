"""Geometry and local detection tools used by the Qwen recovery stage."""
import argparse
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from inspection_core import save_prediction
from inspection_io import ROOT, digest, open_image, read_json, write_json
import run_wire_long_trace_v1 as tracking
import run_wire_span_continuity_v1 as span


def read_mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L')) > 0


def crop_box(points, shape, padding):
    low = points.min(axis=0) - padding
    high = points.max(axis=0) + padding + 1
    y0, x0 = np.maximum(low, 0)
    y1, x1 = np.minimum(high, shape)
    return [int(v) for v in (x0, y0, x1, y1)]


def evidence(points, light, distance):
    """Keep proposals with nearby novel pixels and a thin ridge in the source."""
    if len(points) < 12:
        return None
    direction = points[-1] - points[0]
    chord = float(np.linalg.norm(direction))
    arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if chord < 32 or arc > 1.35 * chord:
        return None
    novelty = int(np.count_nonzero(distance[tuple(points.T)] > 2))
    if novelty < 14 or novelty < .2 * len(points):
        return None
    ridge = span.line_evidence(light, points, span.unit(direction))
    if ridge < .42:
        return None
    return dict(yx=points.tolist(), ridge=ridge, novel_samples=novelty, length=arc)


def plan(run_dir, photo):
    old = read_mask(run_dir / 'recovery/before.png')
    trace = read_mask(run_dir / 'centerlines/trace_only.png')
    raw = read_json(run_dir / 'geometry.json')['paths']
    lab = cv2.cvtColor(np.asarray(photo, dtype=np.float32) / 255, cv2.COLOR_RGB2LAB)
    distance = distance_transform_edt(~old) if old.any() else np.full(old.shape, np.inf)
    candidates = []
    # Recover dropped paths and revisit interrupted endpoints with visible support.
    for points in chain(span.trace_without_corner_edges(trace), (np.asarray(p['points']) for p in raw)):
        candidate = evidence(points, lab, distance)
        if candidate is not None:
            candidates.append(dict(candidate, kind='fragment'))

    near_mask = read_mask(run_dir / 'baseline_centerline.png')
    nearby = distance_transform_edt(~near_mask) if near_mask.any() else distance
    for item in raw:
        pts = np.asarray(item['points'])
        if len(pts) < 28 or item['ridge'] < .55:
            continue
        for ordered in (pts, pts[::-1]):
            y, x = ordered[-1]
            if distance[y, x] > 5 or min(y, x, old.shape[0] - 1 - y, old.shape[1] - 1 - x) < 16:
                continue
            extension, _ = tracking.walk_end(ordered, lab[:, :, 0], nearby, [], max_extension=160)
            if extension:
                candidate = evidence(np.concatenate((ordered[-1:], extension)), lab, distance)
                if candidate is not None:
                    candidates.append(dict(candidate, kind='endpoint'))

    selected = []
    limit = read_json(run_dir / 'run.json')['config']['recovery_regions']
    for candidate in sorted(candidates, key=lambda c: -c['ridge'] * min(c['length'], 240)):
        points = np.asarray(candidate['yx'])
        center = points[len(points) // 2]
        if any(np.linalg.norm(center - s) < 65 for s, _ in selected):
            continue
        selected.append((center, candidate))
        if len(selected) == limit:
            break
    return [dict(c, id=f'R{i:02d}', box=crop_box(np.asarray(c['yx']), old.shape, 75))
            for i, (_, c) in enumerate(selected, 1)]


def model_masks(action, crop, seed):
    import torch
    from lang_sam import LangSAM
    from lang_sam.models.sam import SAM

    checkpoint = str(ROOT / '.cache/torch/hub/checkpoints/sam2.1_hiera_small.pt')
    with torch.inference_mode():
        if action == 'REDETECT':
            model = LangSAM(sam_type='sam2.1_hiera_small', sam_ckpt_path=checkpoint)
            result = model.predict([crop], ['power line.'], box_threshold=.22, text_threshold=.20)[0]
            return list(result['masks'])
        sam = SAM()
        sam.build_model('sam2.1_hiera_small', checkpoint)
        sam.predictor.set_image(np.array(crop, copy=True))
        positive = seed[np.linspace(0, len(seed) - 1, 5, dtype=int), ::-1]
        masks, _, _ = sam.predictor.predict(point_coords=positive, point_labels=np.ones(len(positive)),
                                           multimask_output=True)
        return masks


def propose(run_dir, photo, request):
    old = read_mask(run_dir / 'recovery/before.png')
    x0, y0, x1, y1 = request['box']
    origin = np.array([y0, x0])
    seed = np.asarray(request['region']['yx']) - origin
    crop = photo.crop((x0, y0, x1, y1))
    previous = old[y0:y1, x0:x1]
    distance = distance_transform_edt(~previous) if previous.any() else np.full(previous.shape, np.inf)
    lab = cv2.cvtColor(np.asarray(crop, dtype=np.float32) / 255, cv2.COLOR_RGB2LAB)
    choices = []
    if request['action'] == 'TRACE':
        cached = read_mask(run_dir / 'baseline_centerline.png')[y0:y1, x0:x1]
        supported = distance_transform_edt(~cached) if cached.any() else distance
        head, _ = tracking.walk_end(seed[::-1], lab[:, :, 0], supported, [], max_extension=120)
        tail, _ = tracking.walk_end(seed, lab[:, :, 0], supported, [], max_extension=120)
        candidate = np.concatenate([np.asarray(head, dtype=int)[::-1].reshape(-1, 2), seed,
                                    np.asarray(tail, dtype=int).reshape(-1, 2)])
        if head or tail:
            choices.append(candidate)
    else:
        for mask in model_masks(request['action'], crop, seed):
            choices.extend(span.trace_without_corner_edges(skeletonize(np.asarray(mask, dtype=bool))))
    results = []
    for points in choices:
        item = evidence(points, lab, distance)
        if item is None:
            continue
        close = np.min(np.linalg.norm(points[::4, None] - seed[None, ::4], axis=2))
        if close > 75:
            continue
        item['yx'] = (points + origin).tolist()
        results.append(item)
    return sorted(results, key=lambda r: -r['ridge'] * min(r['length'], 240))[:2]


def finalize(run_dir, photo):
    recovery = run_dir / 'recovery'
    report = read_json(recovery / 'report.json')
    old = read_mask(recovery / 'before.png')
    additions = np.zeros(old.shape, dtype=np.uint8)
    accepted = [np.asarray(c['yx'], dtype=np.int32)[:, ::-1] for c in report['accepted']]
    if accepted:
        cv2.polylines(additions, accepted, False, 1, 1, lineType=cv2.LINE_8)
    added = (additions > 0) & ~old
    final = old | added
    Image.fromarray(added.astype(np.uint8) * 255).save(recovery / 'added_centerline.png')
    Image.fromarray(final.astype(np.uint8) * 255).save(run_dir / 'centerlines/recovered.png')
    summary = read_json(run_dir / 'summary.json')
    geometry = read_json(run_dir / 'geometry.json')
    if added.any():
        summary['graph_path_count'] = save_prediction(
            run_dir, np.asarray(photo), final, read_mask(run_dir / 'baseline_centerline.png'),
            geometry, summary['audit'], 'Multi-turn Qwen recovery')
    summary.update(selected='qwen_recovery', final_centerline_pixels=int(final.sum()),
                   recovery=dict(regions=len(report['regions']), turns=len(report['turns']),
                                 accepted_paths=len(accepted), added_centerline_pixels=int(added.sum()),
                                 report='recovery/report.json'))
    write_json(run_dir / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'propose', 'finalize'])
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    run = read_json(args.run_dir / 'run.json')
    if digest(run['image']) != run['image_sha256']:
        raise ValueError('Input image changed before wire recovery')
    photo = open_image(run['image'])
    recovery = args.run_dir / 'recovery'
    if args.action == 'plan':
        write_json(recovery / 'regions.json', plan(args.run_dir, photo))
    elif args.action == 'propose':
        request = read_json(args.request)
        write_json(args.request.with_suffix('.result.json'), propose(args.run_dir, photo, request))
    else:
        finalize(args.run_dir, photo)


if __name__ == '__main__':
    main()
