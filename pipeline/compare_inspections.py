"""Compare model reviews and final centerlines for an identical inspection input."""
import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from evaluate_first_image_v1 import metrics
from inspection_core import overlay
from inspection_io import digest, open_image, read_json, write_json


def load_run(folder):
    folder = Path(folder).resolve()
    run, summary = [read_json(folder / name) for name in ('run.json', 'summary.json')]
    if run['status'] != 'COMPLETE' or summary['status'] != 'COMPLETE':
        raise ValueError(f'Inspection is incomplete: {folder}')
    with Image.open(folder / 'prediction_centerline.png') as image:
        line = np.asarray(image.convert('L')) > 0
    return dict(folder=folder, run=run, summary=summary, line=line,
                review=read_json(folder / 'qwen_review.json'),
                jobs=read_json(folder / 'review_jobs.json'),
                geometry=read_json(folder / 'geometry.json'))


def annotation_mask(xml, source_image, expected_digest, shape):
    """A renamed input must match the annotated source image byte-for-byte."""
    if digest(source_image) != expected_digest:
        raise ValueError('Annotation source image does not match the inspected image SHA-256')
    root = ET.parse(xml).getroot()
    matches = [item for item in root.findall('image')
               if Path(item.attrib['name']).name == Path(source_image).name]
    if len(matches) != 1:
        raise ValueError('Expected exactly one annotated image')
    item = matches[0]
    if (int(item.attrib['height']), int(item.attrib['width'])) != shape:
        raise ValueError('Annotation image dimensions do not match the centerline')
    mask = np.zeros(shape, dtype=np.uint8)
    count = 0
    for line in item.findall('polyline'):
        if line.attrib['label'] not in ('powerline', 'polyline'):
            continue
        points = np.asarray([[float(v) for v in pair.split(',')]
                             for pair in line.attrib['points'].split(';')])
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
        cv2.polylines(mask, [points], False, 1, 1, lineType=cv2.LINE_8)
        count += 1
    if not count:
        raise ValueError('No powerline polylines found')
    return mask.astype(bool), dict(xml=str(Path(xml).resolve()), xml_sha256=digest(xml),
                                  annotated_image=str(Path(source_image).resolve()),
                                  source_image_sha256=expected_digest, polyline_count=count)


def review_rows(reference, candidate):
    a = {r['id']: r for r in reference['review']['responses']}
    b = {r['id']: r for r in candidate['review']['responses']}
    cases = {r['id']: r for r in reference['geometry']['contact_cases']}
    expected = dict(SCENE_CTRL='NONE', COLOR_CTRL='EMPTY', CTRL_EMPTY='NONE', CTRL_WRONG='NONE')
    result = []
    for job in reference['jobs']:
        ident = job['id']
        options = cases.get(ident, {}).get('options', {})
        row = dict(id=ident, reference_choice=a[ident]['choice'], candidate_choice=b[ident]['choice'],
                   changed=a[ident]['choice'] != b[ident]['choice'],
                   expected_control=expected.get(ident, ''),
                   geometric_pole_option=','.join(k for k, v in options.items() if 'pole' in v),
                   reference_raw=a[ident]['raw'], candidate_raw=b[ident]['raw'])
        for key, model in [('reference', reference), ('candidate', candidate)]:
            choice = row[f'{key}_choice']
            trusted = model['summary']['audit']['review']['contact_control_pass']
            row[f'{key}_qwen_contact_applied'] = bool(trusted and choice in options and 'pole' in options[choice])
        result.append(row)
    return result


def render(output, photo, reference, candidate, rows):
    a, b = reference['line'], candidate['line']
    shared, removed, added = a & b, a & ~b, b & ~a
    diff = (photo * .5).astype(np.uint8)
    for mask, color in [(shared, (175, 175, 175)), (removed, (255, 60, 180)), (added, (70, 255, 90))]:
        diff[cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = color
    Image.fromarray(diff).save(output / 'difference_on_original.jpg', quality=95, subsampling=0)
    panels = [('3B', overlay(photo, a)), ('7B', overlay(photo, b)),
              ('Difference: 3B only = pink | 7B only = green', Image.fromarray(diff))]
    board = Image.new('RGB', (2400, 500), (20, 24, 30))
    draw, font = ImageDraw.Draw(board), ImageFont.load_default(size=17)
    for index, (title, image) in enumerate(panels):
        draw.text((index * 800 + 8, 8), title, fill='white', font=font)
        panel = ImageOps.contain(image, (800, 450), Image.Resampling.LANCZOS)
        board.paste(panel, (index * 800 + (800 - panel.width) // 2, 40 + (450 - panel.height) // 2))
    board.save(output / 'comparison.jpg', quality=95)
    contact_rows = [row for row in rows if row['id'].startswith(('C0', 'CTRL_'))]
    if contact_rows:
        sheet = Image.new('RGB', (1280, ((len(contact_rows) + 1) // 2) * 400), (20, 24, 30))
        draw = ImageDraw.Draw(sheet)
        jobs = {job['id']: job for job in reference['jobs']}
        for index, row in enumerate(contact_rows):
            x, y = (index % 2) * 640, (index // 2) * 400
            title = f'{row["id"]}: 3B={row["reference_choice"]} | 7B={row["candidate_choice"]}'
            if row['expected_control']:
                title += f' | expected={row["expected_control"]}'
            draw.text((x + 5, y + 4), title, fill='white', font=font)
            image = open_image(reference['folder'] / jobs[row['id']]['image'])
            panel = ImageOps.contain(image, (640, 360), Image.Resampling.LANCZOS)
            sheet.paste(panel, (x + (640 - panel.width) // 2, y + 35))
        sheet.save(output / 'contact_reviews.jpg', quality=95)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--annotation-image', type=Path)
    args = p.parse_args()
    if bool(args.annotations) != bool(args.annotation_image):
        p.error('Provide both --annotations and --annotation-image')
    reference, candidate = load_run(args.reference), load_run(args.candidate)
    sha = reference['run']['image_sha256']
    if sha != candidate['run']['image_sha256']:
        raise ValueError('Images differ')
    for model in (reference, candidate):
        if digest(model['run']['image']) != sha:
            raise ValueError('Input image changed since inference')
    if reference['jobs'] != candidate['jobs'] or reference['geometry'] != candidate['geometry']:
        raise ValueError('Review inputs or geometry differ; comparison requires identical inputs')
    for model in (reference, candidate):
        for job in model['jobs']:
            if digest(model['folder'] / job['image']) != job['image_sha256']:
                raise ValueError('A review board changed')
    a, b = reference['line'], candidate['line']
    if a.shape != b.shape:
        raise ValueError('Centerline image dimensions differ')
    changed_config = {key: [reference['run']['config'].get(key), candidate['run']['config'].get(key)]
                      for key in reference['run']['config']
                      if reference['run']['config'].get(key) != candidate['run']['config'].get(key)}
    if set(changed_config) != {'qwen_model'}:
        raise ValueError(f'Expected only qwen_model to change, got {list(changed_config)}')
    rows = review_rows(reference, candidate)
    intersection, union = int((a & b).sum()), int((a | b).sum())
    result = dict(image_sha256=sha, identical_review_inputs=True, identical_geometry=True,
                  changed_config=changed_config, reference_model=reference['review']['model'],
                  candidate_model=candidate['review']['model'],
                  reference_run=str(reference['folder']), candidate_run=str(candidate['folder']),
                  reviews=rows, shared_pixels=intersection, reference_only_pixels=int((a & ~b).sum()),
                  candidate_only_pixels=int((b & ~a).sum()), changed_pixels=int((a ^ b).sum()),
                  prediction_agreement_iou=intersection / union if union else 1.0,
                  accuracy_evaluated=False)
    for key, model in [('reference', reference), ('candidate', candidate)]:
        s, r = model['summary'], model['run']
        result[key] = dict(candidate_count=s['candidate_count'], final_centerline_pixels=int(model['line'].sum()),
                           graph_path_count=s['graph_path_count'],
                           retained_tracking_paths=s['audit']['poles']['kept'],
                           applied_contacts=s['audit']['contacts'], review=s['audit']['review'],
                           color_choice=s['audit']['color_choice'], color_control_pass=s['audit']['color_control_pass'],
                           scene_control_pass=s['audit']['scene_control_pass'],
                           selected_link_count=len(s['audit']['poles']['policy']['links']),
                           pole_stop_count=len(s['audit']['poles']['stops']),
                           qwen_usage=s['qwen_usage'], total_seconds=r['seconds'],
                           review_seconds=next(stage['seconds'] for stage in r['stages'] if stage['name'] == 'review'))
    if args.annotations:
        from scipy.ndimage import distance_transform_edt
        gt, provenance = annotation_mask(args.annotations, args.annotation_image, sha, a.shape)
        distance = distance_transform_edt(~gt)
        for key, line in [('reference', a), ('candidate', b)]:
            scores, _ = metrics(line, gt, distance)
            result[key]['metrics_by_tolerance'] = scores
            result[key]['metrics_5px'] = next(row for row in scores if row['tolerance_pixels'] == 5)
        result.update(accuracy_evaluated=True, annotation=provenance,
                      evaluation_scope='one image; not an independent benchmark', evaluation_primary_tolerance_px=5)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'comparison.json', result)
    with (output / 'review_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render(output, np.asarray(open_image(reference['run']['image'])), reference, candidate, rows)
    print(f'COMPARISON_COMPLETE changed_pixels={result["changed_pixels"]} output={output}', flush=True)


if __name__ == '__main__':
    main()
