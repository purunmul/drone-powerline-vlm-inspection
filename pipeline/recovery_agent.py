"""Let Qwen select a local tool and inspect its proposals before adding a wire."""
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from inspection_io import ROOT, digest, open_image, read_json, write_json
from inspection_worker import ask_images, qwen_settings

TOOLS = ('TRACE', 'REDETECT', 'REFINE')
COLORS = ((255, 212, 56), (240, 102, 246))


def run_tool(run_dir, action, request=None, needs_model=False):
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    if needs_model:
        env['PYTHONPATH'] = str(ROOT / 'vendor/lang-segment-anything')
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    environment = '.venv-langsam' if needs_model else '.venv-eval'
    command = [str(ROOT / environment / 'bin/python'), str(ROOT / 'pipeline/recovery_tools.py'),
               action, str(run_dir)]
    if request is not None:
        command.extend(('--request', str(request)))
    subprocess.run(command, env=env, cwd=ROOT, check=True, timeout=180)


def make_board(photo, current, box, paths, out):
    x0, y0, x1, y1 = box
    original = photo.crop(box)
    marked = original.copy()
    existing = cv2.dilate(current[y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8))
    marked.paste((0, 235, 220), mask=Image.fromarray(existing * 255))
    draw = ImageDraw.Draw(marked)
    for color, path in zip(COLORS, paths):
        points = [(x - x0, y - y0) for y, x in path['yx']]
        draw.line(points, fill=color, width=3)
    board = Image.new('RGB', (1024, 390), '#151d27')
    heading = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=17)
    titles = ['Original crop', 'Cyan = existing | Yellow = candidate 1']
    if len(paths) == 2:
        titles[1] += ' | Magenta = candidate 2'
    for offset, title, panel in zip((0, 512), titles, (original, marked)):
        heading.text((offset + 10, 5), title, font=font, fill='white')
        panel = ImageOps.contain(panel, (510, 359), Image.Resampling.LANCZOS)
        board.paste(panel, (offset + (512 - panel.width) // 2, 29 + (359 - panel.height) // 2))
    board.save(out, quality=94, subsampling=0)


def choose_job(photo, current, folder, track, turn, last_turn):
    options = [] if last_turn else [action for action in TOOLS if action not in track['tried']]
    proposals = track['proposals']
    ident = f'{track["region"]["id"]}_T{turn:02d}'
    board = folder / f'{ident}.jpg'
    make_board(photo, current, track['box'], proposals or [track['region']], board)
    prompt = (
        'The panels show the same drone crop. Cyan lines are wires already in the output. '
        'Inspect the unmarked original on the left for a thin, continuous overhead utility cable. '
        'A road marking, roof edge, branch, or shadow is not an overhead wire. '
    )
    allowed = ['SKIP']
    if proposals:
        allowed += [f'ADD{i}' for i in range(1, len(proposals) + 1)]
        prompt += ('The colored candidates on the right were produced by ' + track['last_tool'] +
                   '. Reply ADD1 for a real missing wire along yellow; ' +
                   ('reply ADD2 for one along magenta; ' if len(proposals) == 2 else '') +
                   'reply SKIP if none is a visible missing wire. ')
    else:
        if track['last_tool']:
            prompt += f"The last {track['last_tool']} request returned no usable candidate. "
        prompt += ('The yellow line marks a suspected gap or weakly detected wire. '
                   'Request a tool if a missing cable is plausible. Reply SKIP if the marked '
                   'feature is clearly background or already covered. ')
    if options:
        meanings = dict(TRACE='follow the visible thin line through the marked gap',
                        REDETECT='search the crop again for missing wires using LangSAM',
                        REFINE='use SAM2 with points on the suspected wire')
        allowed += options
        prompt += 'Tools you can request: ' + '; '.join(f'{a} = {meanings[a]}' for a in options) + '. '
    prompt += 'Reply with exactly one of: ' + ', '.join(allowed) + '.'
    return dict(id=ident, image=board, image_sha256=digest(board), prompt=prompt, allowed=allowed)


def recover(run_dir):
    from vllm import LLM

    run = read_json(run_dir / 'run.json')
    config = run['config']
    photo = open_image(run['image'])
    folder = run_dir / 'recovery'
    folder.mkdir()
    shutil.copyfile(run_dir / 'prediction_centerline.png', folder / 'before.png')
    shutil.copyfile(run_dir / 'line_coordinates.json', folder / 'before_coordinates.json')
    run_tool(run_dir, 'plan')
    regions = read_json(folder / 'regions.json')
    current = np.array(Image.open(folder / 'before.png').convert('L')) > 0
    active = [dict(region=r, box=r['box'], proposals=[], tried=[], last_tool=None) for r in regions]
    turns, accepted = [], []
    if active:
        llm = LLM(**qwen_settings(config, max_pixels=401408))
        for turn in range(1, config['recovery_turns'] + 1):
            last = turn == config['recovery_turns']
            jobs = [choose_job(photo, current, folder, track, turn, last) for track in active]
            following = []
            for track, job, answer in zip(active, jobs, ask_images(llm, jobs)):
                choice = answer['choice']
                event = dict(answer, board=job['image'].name, region=track['region']['id'],
                             turn=turn, available_choices=job['allowed'],
                             tool_from_previous_turn=track['last_tool'])
                turns.append(event)
                if choice in TOOLS:
                    request = folder / f'{job["id"]}.json'
                    # Narrow subsequent tool calls around the last proposal when available.
                    focus = track['proposals'][0] if track['proposals'] else track['region']
                    pts = np.asarray(focus['yx'])
                    padding = 80 if choice == 'REDETECT' else 44
                    low = np.maximum(pts.min(axis=0) - padding, 0)
                    high = np.minimum(pts.max(axis=0) + padding + 1, (photo.height, photo.width))
                    y0, x0 = map(int, low)
                    y1, x1 = map(int, high)
                    track['box'] = [x0, y0, x1, y1]
                    write_json(request, dict(region=focus, box=track['box'], action=choice))
                    run_tool(run_dir, 'propose', request, needs_model=choice != 'TRACE')
                    track['proposals'] = read_json(request.with_suffix('.result.json'))
                    track['tried'].append(choice)
                    track['last_tool'] = choice
                    event['returned_candidates'] = len(track['proposals'])
                    if track['proposals'] or (not last and len(track['tried']) < len(TOOLS)):
                        following.append(track)
                elif choice in ('ADD1', 'ADD2'):
                    proposal = track['proposals'][int(choice[-1]) - 1]
                    mask = np.zeros(current.shape, dtype=np.uint8)
                    cv2.polylines(mask, [np.asarray(proposal['yx'], dtype=np.int32)[:, ::-1]],
                                  False, 1, 1, lineType=cv2.LINE_8)
                    new_pixels = int(np.count_nonzero((mask > 0) & ~current))
                    diameter = cv2.dilate(current.astype(np.uint8), np.ones((5, 5), np.uint8))
                    if np.count_nonzero((mask > 0) & (diameter == 0)) >= 12:
                        current |= mask > 0
                        accepted.append(dict(proposal, region=track['region']['id'],
                                             action=track['last_tool'], new_pixels=new_pixels))
                        event['added_pixels'] = new_pixels
                    else:
                        event['result'] = 'duplicate'
            active = following
            if not active:
                break
    write_json(folder / 'report.json', dict(regions=regions, turns=turns, accepted=accepted,
               max_turns_per_region=config['recovery_turns'], ground_truth_used=False))
    run_tool(run_dir, 'finalize')
    print(f'RECOVERY_COMPLETE regions={len(regions)} turns={len(turns)} accepted={len(accepted)}', flush=True)
