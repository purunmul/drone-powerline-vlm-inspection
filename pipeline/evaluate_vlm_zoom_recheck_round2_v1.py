import argparse
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, find_objects, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from evaluate_first_image_v1 import metrics, readmask
from repository_paths import image_path
from run_vlm_zoom_recheck_round1_v1 import (
    BASE, OUT as ROUND1, STEMS, baseline_path, digest, input_paths, read_json,
)

NAME = 'vlm_zoom_recheck_round2_v1'
DEFAULT_OUTPUT = BASE / 'reproductions' / NAME
DISTANCE = 5
MAX_ANGLE = 25
WINDOW = 13
KEYS = ('centerline_precision', 'centerline_recall', 'centerline_f1')


def tangents(line):
    axis = np.arange(-(WINDOW // 2), WINDOW // 2 + 1, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    image = line.astype(np.float32)
    points = np.nonzero(line)
    sums = [cv2.filter2D(image, -1, k, borderType=cv2.BORDER_CONSTANT)[points]
            for k in (np.ones_like(x), x, y, x*x, y*y, x*y)]
    n, sx, sy, sxx, syy, sxy = sums
    vx, vy = sxx - sx*sx/n, syy - sy*sy/n
    xy = 2 * (sxy - sx*sy/n)
    strength = np.hypot(vx-vy, xy)
    valid = (n >= 5) & (strength >= 0.6 * (vx+vy)) & (strength > 0)
    unit = np.column_stack((vx-vy, xy)) / np.maximum(strength[:, None], 1e-6)
    return np.column_stack(points)[valid], unit[valid]


def agreement(baseline, zoom):
    base_points, base_directions = tangents(baseline)
    zoom_points, zoom_directions = tangents(zoom)
    result = np.zeros_like(baseline)
    if not len(base_points) or not len(zoom_points):
        return result
    cos_limit = np.cos(np.deg2rad(2 * MAX_ANGLE))
    neighbors = cKDTree(zoom_points).query_ball_point(base_points, DISTANCE)
    for (y, x), direction, indices in zip(base_points, base_directions, neighbors):
        if indices and np.any(zoom_directions[indices] @ direction >= cos_limit):
            result[y, x] = True
    return result


def novel_lines(mask, line, far_from_baseline):
    components, count = label(line & far_from_baseline, structure=np.ones((3, 3)))
    accepted = np.zeros_like(line)
    if not count:
        return accepted
    half_width = distance_transform_edt(mask)
    for index, bounds in enumerate(find_objects(components), start=1):
        component = components[bounds] == index
        y, x = np.nonzero(component)
        if len(y) < 48:
            continue
        points = np.column_stack((x, y)).astype(float)
        centered = points - points.mean(axis=0)
        eigenvalues, axes = np.linalg.eigh(centered.T @ centered)
        span = np.ptp(centered @ axes[:, -1])
        if span < 48 or len(y) > 1.6 * span or eigenvalues[0] > 0.12 * eigenvalues[1]:
            continue
        if np.mean(half_width[bounds][component] <= 6) >= 0.8:
            accepted[bounds] |= component
    return accepted


def predict(stem, selected, regions):
    base = readmask(baseline_path(stem))
    strict, fixed = base.copy(), base.copy()
    novel = np.zeros_like(base)
    for ident in selected:
        entry = regions[(stem, ident)]
        x0, y0, x1, y1 = entry['core']
        a, b, c, d = entry['zoom']
        region = np.s_[y0:y1, x0:x1]
        core = np.s_[y0-b:y1-b, x0-a:x1-a]
        cropped_base = base[b:d, a:c]
        zoom_line = np.zeros_like(cropped_base)
        novel_crop = np.zeros_like(cropped_base)
        far = distance_transform_edt(~cropped_base) > DISTANCE if cropped_base.any() else ~cropped_base
        with np.load(ROUND1 / stem / f'{ident}_zoom.npz') as data:
            masks = data['masks']
            assert masks.shape[1:] == cropped_base.shape
            for mask in masks:
                skeleton = skeletonize(mask, method='zhang')
                zoom_line |= skeleton
                novel_crop |= novel_lines(mask, skeleton, far)
        fixed[region] |= zoom_line[core]
        strict[region] = agreement(cropped_base, zoom_line)[core]
        novel[region] = novel_crop[core]
    saved = readmask(ROUND1 / 'review' / stem / 'fixed_centerline.png')
    assert np.array_equal(saved, fixed), f'Round 1 reproduction failed for {stem}'
    return dict(baseline=base, round1_fixed=fixed, agreement_only=strict,
                agreement_plus_novel=strict | novel, baseline_plus_novel=base | novel)


def verify_direction():
    a = np.zeros((64, 64), dtype=bool)
    b = np.zeros_like(a)
    a[30, 10:55] = True
    b[34, 10:55] = True
    assert agreement(a, b)[30, 30]
    b[:] = False
    b[10:55, 30] = True
    assert not agreement(a, b).any()


def evaluate(predictions):
    rows = []
    for stem, variants in predictions.items():
        source = (BASE / 'diagnostic_eval' / f'{stem}_v1' if stem == STEMS[0]
                  else BASE / 'dev_transfer_20261107_v1' / stem / 'evaluation')
        ref = readmask(source / 'reference_centerlines.png')
        refdist = distance_transform_edt(~ref)
        base = variants['baseline']
        base_covered = distance_transform_edt(~base)[ref] <= DISTANCE
        for name, pred in variants.items():
            scores, distance = metrics(pred, ref, refdist)
            score = next(row for row in scores if row['tolerance_pixels'] == DISTANCE)
            covered = (distance[ref] <= DISTANCE) if distance is not None else np.zeros_like(base_covered)
            added, removed = pred & ~base, base & ~pred
            rows.append(dict(image=stem, method=name, **score,
                             added_pred=int(added.sum()), removed_pred=int(removed.sum()),
                             added_matched=int((refdist[added] <= DISTANCE).sum()),
                             removed_matched=int((refdist[removed] <= DISTANCE).sum()),
                             recovered_reference=int((covered & ~base_covered).sum()),
                             lost_reference=int((base_covered & ~covered).sum())))
    return rows


def render(output, stem, line):
    folder = output / stem
    folder.mkdir(exist_ok=True)
    info = read_json(input_paths(stem)[0] / 'result.json')
    source = image_path(info['image'])
    assert digest(source) == info['sha256']
    with Image.open(source) as image:
        photo = np.array(image.convert('RGB'))
    assert line.shape == photo.shape[:2] == (2160, 3840)
    pixels = line.astype(np.uint8)
    halo = cv2.dilate(pixels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    visible = cv2.dilate(pixels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    photo[halo] = (photo[halo] * 0.35).astype(np.uint8)
    photo[visible] = (30, 255, 234)
    overlay = Image.fromarray(photo)
    overlay.save(folder / 'prediction_on_original.jpg', quality=95, subsampling=0)
    Image.fromarray(pixels * 255).save(folder / 'prediction_centerline.png')
    return overlay


def main(output):
    archive_path = output.parent / f'{output.name}.zip'
    if output.exists() or archive_path.exists():
        raise FileExistsError(f'Choose a new --output-dir: {output}')
    verify_direction()
    regions = {(r['stem'], r['id']): r for r in read_json(ROUND1 / 'manifest.json') if not r['control']}
    choices = read_json(ROUND1 / 'choices.json')
    predictions = {}
    for stem in STEMS:
        predictions[stem] = predict(stem, choices[stem]['fixed'], regions)
        print('REPRODUCED', stem, flush=True)
    rows = evaluate(predictions)
    methods = tuple(predictions[STEMS[0]])
    summary = {name: {k: float(np.mean([row[k] for row in rows if row['method'] == name]))
                      for k in KEYS} for name in methods}
    selected = max(methods, key=lambda name: summary[name]['centerline_f1'])
    print('SELECTED', selected, flush=True)
    for name, values in summary.items():
        print(name, ' '.join(f'{100*v:.2f}' for v in values.values()), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(dict(selected=selected, tolerance_pixels=DISTANCE,
        selection='highest mean per-image F1 on the three development images', methods=summary), indent=2)+'\n')
    (output / 'README.txt').write_text(
        'Original-resolution model predictions on the three development images.\n'
        'Cyan lines are visualized at 5 px. The PNG centerline is the exact evaluation input.\n'
        'Baseline is the original LangSAM prediction. Round 1 fixed adds every zoom centerline.\n'
        'Agreement checks existing lines inside the 4 fixed-selected regions per image.\n'
        'Agreement uses a 13 px orientation window, 5 px distance and a 25 degree angle limit.\n'
        'Novel checks zoom-only components at >5 px from baseline, with span >=48 px,\n'
        'eigenvalue ratio <=0.12, density <=1.6, and at least 80% within 6 px mask radius.\n'
        'Reference annotations are used only for evaluation and comparing the methods.\n'
        'Chosen output is a development result and is not an independent test score.\n')
    preview = max((row for row in rows if row['method'] == selected), key=lambda row: row['centerline_f1'])['image']
    for stem in STEMS:
        overlay = render(output, stem, predictions[stem][selected])
        if stem == preview:
            overlay.resize((1600, 900), Image.Resampling.LANCZOS).save(output / 'preview.jpg', quality=92)
    with ZipFile(archive_path, 'x', ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, Path(output.name) / path.relative_to(output))
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 10
    print('VERIFIED', archive_path, 'PREVIEW', preview, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce the development-set overlays from saved LangSAM outputs.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output_dir)
