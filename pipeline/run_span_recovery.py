"""Recover long missing wire spans from completed inspections, without annotations."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from inspection_io import digest, open_image, read_json, write_json
from inspection_worker import ask_images, qwen_settings
from mask_overlap import rasterize
from span_candidates import generate, unit


def draw_lines(image, lines, color, width):
    draw = ImageDraw.Draw(image)
    for line in lines:
        draw.line(line, fill=color, width=width)


def oriented_strip(photo, old, points_xy):
    a, b = points_xy[[0, -1]].astype(float)
    direction = unit(b-a)
    normal = np.array([-direction[1], direction[0]])
    s = (points_xy-a) @ direction
    n = (points_xy-a) @ normal
    u0, u1 = min(0, s.min())-38, max(s.max(), 0)+38
    v0, v1 = n.min()-43, n.max()+43
    width = int(np.ceil(u1-u0))+1
    height = int(np.ceil(v1-v0))+1
    grid = a + ((u0+np.arange(width))[None, :, None]*direction +
                (v0+np.arange(height))[:, None, None]*normal)
    grid = grid.astype(np.float32)
    def remap(image, mode):
        return cv2.remap(image, grid[..., 0], grid[..., 1], mode, borderMode=cv2.BORDER_REPLICATE)
    original = Image.fromarray(remap(np.asarray(photo), cv2.INTER_LINEAR))
    marked = original.copy()
    existing = cv2.dilate(remap(old, cv2.INTER_NEAREST), np.ones((3,3), np.uint8))
    marked.paste((0, 230, 222), mask=Image.fromarray(existing*255))
    line = list(zip(s-u0, n-v0))
    draw_lines(marked, [line], '#ffdc47', 3)
    return original, marked, line


def board(photo, old, proposal, poles, path):
    points = np.asarray(proposal['xy'])
    orig, marked, _ = oriented_strip(photo, old, points)
    context = photo.copy()
    draw_lines(context, [list(map(tuple, points))], '#ffdc47', 12)
    draw = ImageDraw.Draw(context)
    font = ImageFont.load_default(size=50)
    for pole in poles:
        if pole['score'] >= .28:
            x0,y0,x1,y1 = pole['box']
            rect = (x0, y0, x1, y0+min(110,(y1-y0)*.22))
            draw.rectangle(rect, outline='#ff853c', width=7)
            draw.text(rect[:2], pole['id'], fill='#ffb064', font=font, stroke_width=2)
    out = Image.new('RGB', (1050, 640), '#161e29')
    heading = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=22)
    def place(panel, x, y, w, h):
        panel = ImageOps.contain(panel, (w, h), Image.Resampling.LANCZOS)
        out.paste(panel, (x+(w-panel.width)//2, y+(h-panel.height)//2))
    heading.text((18,4), 'Context | Orange = possible pole top', fill='white', font=font)
    place(context, 0, 32, 1050, 226)
    heading.text((18,261), 'Original along the proposed route', fill='white', font=font)
    place(orig, 0, 293, 1050, 151)
    heading.text((18,446), 'Same strip | Cyan = existing | Yellow = proposal', fill='white', font=font)
    place(marked, 0, 478, 1050, 151)
    out.save(path, quality=94, subsampling=0)


def prepare(folder, output, limit, cached=None):
    run = read_json(folder/'run.json')
    if run['status'] != 'COMPLETE' or run['config']['prompt'].strip().lower().rstrip('.') != 'power line':
        raise ValueError(f'Expected completed power-line inspection: {folder}')
    if digest(run['image']) != run['image_sha256']:
        raise ValueError('Source image differs from the completed inspection')
    coords = read_json(folder/'line_coordinates.json')
    photo = open_image(run['image'])
    if (coords['coordinate_order'] != 'xy' or coords['image_size_wh'] != list(photo.size)):
        raise ValueError('Saved centerlines do not match the image dimensions')
    with Image.open(folder/'prediction_centerline.png') as image:
        old = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
    if old.shape != (photo.height, photo.width):
        raise ValueError('Saved prediction does not match the image dimensions')
    poles = read_json(folder/'geometry.json')['poles']
    if cached is None:
        proposals, audit = generate(photo, old > 0, coords['paths'], poles, limit)
    else:
        planned = read_json(cached/'candidates.json')
        if (planned['image_sha256'] != run['image_sha256'] or Path(planned['source']) != folder):
            raise ValueError('Saved candidates refer to a different inspection')
        proposals = planned['proposals']
        if len(proposals) > limit:
            raise ValueError('Saved candidates exceed the candidate limit')
        audit = {k:planned[k] for k in ('seeds','corridors','ridge_candidates')}
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for item in proposals:
        path = output/(item['id']+'.jpg')
        board(photo, old, item, poles, path)
        files[item['id']] = path
    write_json(output/'candidates.json', dict(source=str(folder), image=run['image'],
               image_sha256=run['image_sha256'], image_size_wh=list(photo.size),
               ground_truth_used=False, proposals=proposals, **audit))
    print(f'{output.name}: {len(proposals)} proposals from {audit["corridors"]} search corridors', flush=True)
    return dict(run=run, coords=coords, photo=photo, old=old,
                output=output, proposals=proposals, files=files)


def make_job(item, image, second=False):
    context = (
        'Review an overhead electricity wire on a drone image. The top panel shows context, the '
        'middle shows the unmarked original along the proposed route, the bottom adds cyan existing '
        'wires and a yellow proposed route. A wire often runs continuously from a pole or out of the '
        'frame, and parallel wires in the same span can run at similar spacing. A pole can be outside '
        'the image. Use these only as hints and check the visible thin line in the original. '
    )
    if second:
        question = (
            'Compare the two ends and interior of the yellow route against the original above. '
            'Answer SWITCH if the route jumps to a different parallel cable or follows a roof, branch, '
            'pole shaft, or road marking in any appreciable part. Answer UNCERTAIN if the original '
            'does not let you decide. Answer SAME only if the path follows the same visible cable.'
        )
        allowed = ['SWITCH', 'UNCERTAIN', 'SAME']
    else:
        question = (
            'Does the majority of the yellow route correspond to a visible overhead cable in the '
            'original, including areas away from the cyan? Answer BACKGROUND for a non-wire feature '
            'or a route unsupported by the original, UNCERTAIN for indistinct detail, or WIRE for a cable.'
        )
        allowed = ['BACKGROUND', 'UNCERTAIN', 'WIRE']
    return dict(id=item['id'], image=image, image_sha256=digest(image), allowed=allowed,
                prompt=context+question+' Reply with one label only.')


def contrast_jobs(prepared, accepted, swap):
    folder = prepared['output']
    jobs = []
    for item in accepted:
        points = np.asarray(item['xy'])
        original, _, line = oriented_strip(prepared['photo'], prepared['old'], points)
        shift = (16 if int(item['id'][1:]) % 2 else -16)
        shifted = [(x, y+shift) for x,y in line]
        paths = (shifted, line) if swap else (line, shifted)
        canvas = Image.new('RGB', (1040, 540), '#161e29')
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=22)
        for i,(label, route) in enumerate(zip(('Unmarked original', 'A', 'B'), (None, *paths))):
            panel = original.copy()
            if route is not None:
                # Small markers leave the proposed wire visible between samples.
                marks = ImageDraw.Draw(panel)
                for x,y in route[::5]:
                    marks.line((x,y-8,x,y-4),fill='#ffd747',width=2)
                    marks.line((x,y+4,x,y+8),fill='#ffd747',width=2)
            panel = ImageOps.contain(panel, (1020,145), Image.Resampling.LANCZOS)
            draw.text((12,i*180+3), label, fill='white', font=font)
            canvas.paste(panel, ((1040-panel.width)//2,i*180+29+(145-panel.height)//2))
        image = folder/(item['id']+('_BA.jpg' if swap else '_AB.jpg'))
        canvas.save(image, quality=94, subsampling=0)
        prompt = (
            'All three panels contain the identical crop of a drone photograph. A and B mark two '
            'different nearby tracks in this crop. The yellow pairs of ticks straddle each track. '
            'Look at the original on top. Which track lies on a visible thin overhead power cable '
            'between the ticks for most of the strip? Distinguish the cable from branches, road '
            'markings or roof edges. Reply A or B. Reply NEITHER for background, or UNSURE if you '
            'cannot distinguish the tracks. Give exactly one label.'
        )
        jobs.append(dict(id=item['id'],image=image,image_sha256=digest(image),prompt=prompt,
                         allowed=['A','B','NEITHER','UNSURE']))
    return jobs


def validate(prepared, llm, contrast_check=True):
    proposals, files = prepared['proposals'], prepared['files']
    jobs = [make_job(item, files[item['id']]) for item in proposals]
    first = {a['id']:a for a in ask_images(llm, jobs)} if jobs else {}
    confirming = [make_job(item, files[item['id']], second=True) for item in proposals
                  if first[item['id']]['choice']=='WIRE']
    second = {a['id']:a for a in ask_images(llm, confirming)} if confirming else {}
    accepted = [p for p in proposals if second.get(p['id'], {}).get('choice')=='SAME']
    semantic_accepted = [p['id'] for p in accepted]
    comparisons = {}
    if accepted and contrast_check:
        for swap in (False, True):
            comparisons['BA' if swap else 'AB'] = {
                a['id']:a for a in ask_images(llm, contrast_jobs(prepared, accepted, swap))
            }
        accepted = [p for p in accepted
                    if comparisons['AB'][p['id']]['choice']=='A' and comparisons['BA'][p['id']]['choice']=='B']
    output = prepared['output']
    write_json(output/'qwen_selection.json', dict(first=first, second=second,
               semantic_accepted=semantic_accepted, comparison=comparisons,
               accepted=[p['id'] for p in accepted], ground_truth_used=False))
    coords = prepared['coords']
    final_paths = coords['paths'] + [p['xy'] for p in accepted]
    write_json(output/'line_coordinates.json', dict(coords, paths=final_paths))
    photo = prepared['photo']
    shape = (photo.height,photo.width)
    canvas = rasterize(final_paths, shape, 1)
    Image.fromarray(canvas*255).save(output/'prediction_centerline.png')
    addition = rasterize([p['xy'] for p in accepted],shape,5)
    preview = photo.copy()
    guide = cv2.dilate(canvas, np.ones((5,5), np.uint8))
    preview.paste((0,230,222), mask=Image.fromarray(guide*255))
    preview.paste((255,220,65), mask=Image.fromarray(addition*255))
    preview.thumbnail((1600,1600))
    preview.save(output/'preview.jpg', quality=94)
    print(output.name, 'accepted', len(accepted), dict(Counter(a['choice'] for a in first.values())),
          dict(Counter(a['choice'] for a in second.values())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-candidates', type=Path,
                        help='Review saved candidates from an earlier run of the same inspections')
    parser.add_argument('--skip-contrast-check', action='store_true',
                        help='Use the two semantic questions only, for comparison')
    parser.add_argument('--candidates', type=int, default=18)
    parser.add_argument('--gpu-mem', type=float, default=.30)
    args = parser.parse_args()
    if args.candidates < 1 or not 0 < args.gpu_mem < 1:
        parser.error('Use a positive candidate limit and GPU memory fraction in (0, 1)')
    inspections = [p.resolve() for p in args.inspection]
    names = [Path(read_json(p/'run.json')['image']).stem for p in inspections]
    if len(set(names)) != len(names):
        parser.error('Input image names must be unique within one run')
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = [prepare(p, args.output/name, args.candidates,
                        args.reuse_candidates/name if args.reuse_candidates else None)
                for p,name in zip(inspections,names)]
    models = {p['run']['config']['qwen_model'] for p in prepared}
    if len(models) != 1:
        parser.error('Use the same Qwen model for all input inspections')
    llm = None
    if any(p['proposals'] for p in prepared):
        from vllm import LLM

        llm = LLM(**qwen_settings(dict(qwen_model=models.pop(), gpu_mem=args.gpu_mem), max_pixels=602112))
    for p in prepared:
        validate(p,llm,contrast_check=not args.skip_contrast_check)
    write_json(args.output/'run.json', dict(status='COMPLETE', images=names, ground_truth_used=False))


if __name__ == '__main__':
    main()
