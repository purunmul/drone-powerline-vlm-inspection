"""Continuity inference on fresh masks; never reads development results or XML."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

import run_wire_span_continuity_v1 as span
import run_wire_contacts_v1 as prior
import run_wire_four_rules_v1 as four
import run_wire_long_trace_v1 as tracking
from inspection_io import SCENE_PROMPT, digest, open_image, read_json, write_json


def extract_fragments(mask_records, shape, lab):
    """Each mask has a local raster and its (x, y) origin in the input image."""
    baseline = np.zeros(shape, dtype=bool)
    paths = []
    for mask, (ox, oy) in mask_records:
        mask = np.asarray(mask, dtype=bool)
        if (mask.ndim != 2 or min(ox, oy) < 0 or
                oy + mask.shape[0] > shape[0] or ox + mask.shape[1] > shape[1]):
            raise ValueError('Candidate mask lies outside original image coordinates')
        x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
        if not width or not height:
            continue
        # Padding supplies the background that existed outside a cropped mask.
        crop = np.pad(mask[y:y + height, x:x + width], 1)
        line = skeletonize(crop)
        baseline[oy + y:oy + y + height, ox + x:ox + x + width] |= line[1:-1, 1:-1]
        half_width = distance_transform_edt(crop)
        for piece in span.trace_without_corner_edges(line):
            if len(piece) < span.MIN_LENGTH:
                continue
            displacement = piece[-1] - piece[0]
            chord = float(np.linalg.norm(displacement))
            arc = float(np.linalg.norm(np.diff(piece, axis=0), axis=1).sum())
            if chord < span.MIN_LENGTH - 2 or arc > 1.28 * chord:
                continue
            direction = span.unit(displacement)
            normal = np.array([-direction[1], direction[0]])
            if np.quantile(np.abs((piece - piece[0]) @ normal), .90) > max(3, min(11, chord * .035)):
                continue
            narrow = float(np.mean(half_width[tuple(piece.T)] <= 6))
            points = piece + (oy + y - 1, ox + x - 1)
            ridge = span.line_evidence(lab, points, direction)
            if narrow >= .67 or ridge >= .38:
                paths.append(dict(points=points, arc=arc, narrow=narrow, ridge=ridge))
    return baseline, paths


def normalize_contact_sides(cases, paths):
    """Use side 0=start and 1=end throughout selection AND tracing."""
    for case in cases:
        for option in case['options'].values():
            if 'path' not in option:
                continue
            points = paths[option['path']]['points']
            if np.array_equal(points[0], option['yx']):
                option['side'] = 0
            elif np.array_equal(points[-1], option['yx']):
                option['side'] = 1
            else:
                raise ValueError('Contact must reference a fragment endpoint')


def prepare(run_dir):
    run_dir = Path(run_dir)
    run = read_json(run_dir / 'run.json')
    detection = read_json(run_dir / 'detection/result.json')
    if digest(run['image']) != run['image_sha256']:
        raise ValueError('Input image changed during inspection')
    photo = np.asarray(open_image(run['image']))
    lab = cv2.cvtColor(photo.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)

    def masks():
        for candidate in detection['candidates']:
            with Image.open(run_dir / candidate['mask']) as image:
                yield np.asarray(image.convert('L')) > 0, candidate['mask_origin_xy']

    baseline, paths = extract_fragments(masks(), photo.shape[:2], lab)
    restored = prior.restore_fragments(baseline, paths, lab)
    paths.extend(restored)
    links = span.gap_candidates(paths, lab, baseline)
    poles = (span.terminal_zones(detection['poles'], paths) if paths else
             [dict(p, contacts=[]) for p in detection['poles']])
    signatures = [four.signature(lab, item['points']) for item in paths]
    ranked = sorted(range(len(paths)), key=lambda i: -(paths[i]['ridge'] * min(paths[i]['arc'], 140)))
    seeds = []
    for i in ranked:
        if paths[i]['ridge'] < .60 or paths[i]['arc'] < 42 or signatures[i]['width'] is None:
            continue
        middle = paths[i]['points'][len(paths[i]['points']) // 2]
        if any(np.linalg.norm(middle - paths[j]['points'][len(paths[j]['points']) // 2]) < 130 for j in seeds):
            continue
        seeds.append(i)
        if len(seeds) == 4:
            break
    Image.fromarray(baseline.astype(np.uint8) * 255).save(run_dir / 'baseline_centerline.png')
    boards = run_dir / 'review_inputs'
    boards.mkdir()
    scene = Image.fromarray(photo)
    scene.thumbnail((1280, 1280))
    scene.save(boards / 'scene.jpg', quality=94)
    Image.new('RGB', (640, 360), (150, 150, 150)).save(boards / 'scene_control.jpg')
    jobs = [dict(id='SCENE', image='review_inputs/scene.jpg', prompt=SCENE_PROMPT,
                 allowed=['WIRE', 'NONE', 'UNSURE']),
            dict(id='SCENE_CTRL', image='review_inputs/scene_control.jpg', prompt=SCENE_PROMPT,
                 allowed=['WIRE', 'NONE', 'UNSURE'])]
    if seeds:
        four.color_board(photo, paths, seeds, boards / 'color.jpg')
        Image.new('RGB', (640, 360), (150, 150, 150)).save(boards / 'color_control.jpg')
        prompt = four.COLOR_PROMPT + ' If no numbered crops are visible, reply EMPTY instead.'
        for ident, filename in [('COLOR', 'color.jpg'), ('COLOR_CTRL', 'color_control.jpg')]:
            jobs.append(dict(id=ident, image=f'review_inputs/{filename}', prompt=prompt,
                             allowed=['WIDE', 'MEDIUM', 'EMPTY']))
    cases = prior.review_cases('input', photo, paths, poles, baseline, output_dir=boards,
                               max_cases=run['config']['max_review_cases'])
    normalize_contact_sides(cases, paths)
    for case in cases:
        case['image'] = str(Path(case['image']).relative_to(run_dir))
        options = [row for row in case['options'].values() if 'path' in row]
        width = next((signatures[row['path']]['width'] for row in options
                      if signatures[row['path']]['width'] is not None), None)
        measurement = f' Estimated visible wire width: {width:.1f} pixels.' if width else ''
        jobs.append(dict(id=case['id'], image=case['image'], prompt=four.CONTACT_PROMPT + measurement,
                         allowed=['A', 'B', 'NONE', 'UNSURE']))
    for job in jobs:
        job['image_sha256'] = digest(run_dir / job['image'])
    write_json(run_dir / 'review_jobs.json', jobs)
    serial_paths = [dict(p, points=p['points'].tolist()) for p in paths]
    write_json(run_dir / 'geometry.json', dict(paths=serial_paths, links=links, poles=poles,
               signatures=signatures, seeds=seeds, contact_cases=cases,
               restored_fragments=len(restored), coordinate_order='yx'))
    print(f'GEOMETRY_COMPLETE paths={len(paths)} links={len(links)} reviews={len(jobs)}', flush=True)


def select_contacts(geometry, responses):
    poles = {p['id']: p for p in geometry['poles']}
    answers = {r['id']: r['choice'] for r in responses}
    controls = [answers.get(k) for k in ('CTRL_EMPTY', 'CTRL_WRONG')]
    trusted = controls == ['NONE', 'NONE']
    contacts = []
    for case in geometry['contact_cases']:
        if case['id'].startswith('CTRL_'):
            continue
        for letter, row in case['options'].items():
            if 'pole' not in row:
                continue
            path = geometry['paths'][row['path']]
            pole = poles[row['pole']]
            supported = pole['score'] >= .34 and path['arc'] >= 36 and path['ridge'] >= .50
            confirmed = trusted and answers.get(case['id']) == letter
            if supported or confirmed:
                contacts.append(dict(row, vlm=confirmed, geometry_supported=supported))
    return contacts, dict(contact_control_pass=trusted, contact_controls=controls,
                          geometry_contacts=sum(p['geometry_supported'] for p in contacts),
                          qwen_confirmed_contacts=sum(p['vlm'] for p in contacts))


def predict(geometry, baseline, photo, responses, config):
    """Reuse latest four-rule selection and v2 tracing with explicit run inputs."""
    for row in geometry['paths']:
        row['points'] = np.asarray(row['points'], dtype=int)
    paths = geometry['paths']
    contacts, review_audit = select_contacts(geometry, responses)
    answers = {r['id']: r['choice'] for r in responses}
    color_trusted = answers.get('COLOR_CTRL') == 'EMPTY'
    color_choice = answers.get('COLOR') if color_trusted else None
    scale = 1.30 if color_choice == 'WIDE' else 1.12
    signatures = geometry['signatures']
    standard = ([np.asarray(signatures[i]['color']) for i in geometry['seeds']],
                [(c, r * scale) for c, r in four.palette_profiles()])
    observed = np.zeros_like(baseline)
    for row in paths:
        observed[tuple(row['points'].T)] = True
    lab = cv2.cvtColor(photo.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[:, :, 0]
    distance = distance_transform_edt(~baseline)
    variants = {'baseline': baseline}
    audits = {}
    for mode, stops in [('no_poles', []), ('poles', contacts)]:
        joined, policy = four.select(observed, geometry['links'], stops, signatures, standard, True, True)
        chains = tracking.chain_paths(paths, geometry['links'], {r['id'] for r in policy['links']}, stops)
        raw, filtered, record = tracking.render_paths(
            chains, lab, distance, stops, min_length=config['min_length'],
            min_terminal_length=config['min_terminal_length'], max_extension=config['max_extension'])
        audits[mode] = dict(record, chains=len(chains), policy=policy)
        if mode == 'no_poles':
            variants.update(trace_only=raw, trace_min_length=filtered)
        else:
            variants.update(four_rules=joined, final=filtered)
    return variants, dict(audits, contacts=contacts, review=review_audit,
                          color_choice=color_choice, color_choice_raw=answers.get('COLOR'),
                          color_control_pass=color_trusted, color_scale=scale,
                          scene_choice=answers.get('SCENE'),
                          scene_control_pass=answers.get('SCENE_CTRL') == 'NONE',
                          invalid_qwen_responses=sum(r['choice'] is None for r in responses))


def overlay(photo, line):
    pixels = line.astype(np.uint8)
    result = photo.copy()
    halo = cv2.dilate(pixels, np.ones((9, 9), np.uint8)) > 0
    visible = cv2.dilate(pixels, np.ones((5, 5), np.uint8)) > 0
    result[halo] = (result[halo] * .35).astype(np.uint8)
    result[visible] = (4, 244, 229)
    return Image.fromarray(result)


def finish(run_dir):
    run_dir = Path(run_dir)
    run = read_json(run_dir / 'run.json')
    if digest(run['image']) != run['image_sha256']:
        raise ValueError('Input image changed during inspection')
    photo = np.asarray(open_image(run['image']))
    geometry = read_json(run_dir / 'geometry.json')
    review = read_json(run_dir / 'qwen_review.json')
    jobs = read_json(run_dir / 'review_jobs.json')
    if [r['id'] for r in review['responses']] != [j['id'] for j in jobs]:
        raise ValueError('Qwen response IDs do not match review jobs')
    with Image.open(run_dir / 'baseline_centerline.png') as image:
        baseline = np.asarray(image.convert('L')) > 0
    variants, audit = predict(geometry, baseline, photo, review['responses'], run['config'])
    intermediate = run_dir / 'centerlines'
    intermediate.mkdir()
    for name, line in variants.items():
        Image.fromarray(line.astype(np.uint8) * 255).save(intermediate / f'{name}.png')
    final = variants['final']
    Image.fromarray(final.astype(np.uint8) * 255).save(run_dir / 'prediction_centerline.png')
    image = overlay(photo, final)
    image.save(run_dir / 'prediction_on_original.jpg', quality=95, subsampling=0)
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    preview.save(run_dir / 'preview.jpg', quality=93)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for pole in geometry['poles']:
        draw.rectangle(pole['box'], outline=(255, 154, 44), width=3)
        draw.text(tuple(pole['box'][:2]), pole['id'], fill='white', stroke_width=1, stroke_fill='black')
    for stop in audit['poles']['stops']:
        y, x = stop['yx']
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 154, 44))
    marked.save(run_dir / 'poles_and_stop_points.jpg', quality=94)
    comparison = Image.new('RGB', (1600, 490), (20, 24, 30))
    draw = ImageDraw.Draw(comparison)
    font = ImageFont.load_default(size=18)
    for index, name in enumerate(('baseline', 'final')):
        draw.text((800 * index + 8, 6), 'Initial LangSAM' if index == 0 else 'Continuity + Qwen + tracking',
                  fill='white', font=font)
        panel = ImageOps.contain(overlay(photo, variants[name]), (800, 450), Image.Resampling.LANCZOS)
        comparison.paste(panel, (800 * index + (800 - panel.width) // 2, 35 + (450 - panel.height) // 2))
    comparison.save(run_dir / 'comparison.jpg', quality=93)
    graph_paths = [p[:, ::-1].tolist() for p in span.trace_without_corner_edges(final)]
    write_json(run_dir / 'line_coordinates.json', dict(image_size_wh=[photo.shape[1], photo.shape[0]],
               coordinate_order='xy', unit='EXIF-oriented input image pixel', physical_wire_instances=False,
               paths=graph_paths))
    detection = read_json(run_dir / 'detection/result.json')
    write_json(run_dir / 'summary.json', dict(status='COMPLETE', image=run['image'],
               image_sha256=run['image_sha256'], image_size_wh=run['image_size_wh'],
               selected='trace_min_length_poles', ground_truth_used=False, accuracy_evaluated=False,
               candidate_count_includes_tile_duplicates=True, candidate_count=len(detection['candidates']),
               tile_count=len(detection['tiles']), pole_count=len(detection['poles']),
               fragment_count=len(geometry['paths']), link_candidate_count=len(geometry['links']),
               final_centerline_pixels=int(final.sum()), graph_path_count=len(graph_paths),
               qwen_model=review['model'], qwen_usage=review['usage'], audit=audit,
               parameters=run['config']))
    print(f'TRACE_COMPLETE pixels={int(final.sum())} graph_paths={len(graph_paths)}', flush=True)
