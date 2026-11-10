"""Separate processes keep LangSAM and vLLM dependencies and GPU allocations apart."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from inspection_io import digest, open_image, parse_choice, read_json, tile_boxes, write_json


def detect(run_dir):
    import importlib.metadata
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from lang_sam import LangSAM

    run = read_json(run_dir / 'run.json')
    config = run['config']
    if digest(run['image']) != run['image_sha256']:
        raise ValueError('Input image changed during inspection')
    image = open_image(run['image'])
    width, height = image.size
    if not torch.cuda.is_available():
        raise RuntimeError('LangSAM requires an available CUDA GPU')
    output = run_dir / 'detection'
    (output / 'masks').mkdir(parents=True)
    started = time.perf_counter()
    model = LangSAM(sam_type='sam2.1_hiera_small')
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    union = np.zeros((height, width), dtype=bool)
    candidates, tiles = [], []
    regions = tile_boxes(width, height, config['tile_size'], config['overlap'])
    for index, (x0, y0, x1, y1) in enumerate(regions, 1):
        crop = image.crop((x0, y0, x1, y1))
        start = time.perf_counter()
        with torch.inference_mode():
            result = model.predict([crop], [config['prompt']],
                                   box_threshold=config['box_threshold'],
                                   text_threshold=config['text_threshold'])[0]
        torch.cuda.synchronize()
        boxes = np.asarray(result['boxes'], dtype=np.float32).reshape(-1, 4)
        masks = np.asarray(result['masks'], dtype=bool).reshape(len(boxes), crop.height, crop.width)
        scores = np.asarray(result['scores'], dtype=np.float32).reshape(-1)
        mask_scores = np.asarray(result['mask_scores'], dtype=np.float32).reshape(-1)
        if not len(boxes) == len(scores) == len(mask_scores):
            raise ValueError('LangSAM output lengths do not match')
        for box, mask, score, mask_score in zip(boxes, masks, scores, mask_scores):
            ident = f'C{len(candidates) + 1:05d}'
            union[y0:y1, x0:x1] |= mask
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            stored = mask[y:y + h, x:x + w] if w and h else np.zeros((1, 1), dtype=bool)
            filename = f'detection/masks/{ident}.png'
            Image.fromarray(stored.astype(np.uint8) * 255).save(run_dir / filename)
            global_box = box + np.asarray([x0, y0, x0, y0])
            global_box[::2] = np.clip(global_box[::2], 0, width)
            global_box[1::2] = np.clip(global_box[1::2], 0, height)
            candidates.append(dict(id=ident, tile=index, box_xyxy=global_box.tolist(),
                                   detector_score=float(score), sam_mask_score=float(mask_score),
                                   mask=filename, mask_origin_xy=[x0 + x, y0 + y],
                                   mask_area_pixels=int(mask.sum())))
        tiles.append(dict(id=index, box_xyxy=[x0, y0, x1, y1], candidate_count=len(boxes),
                          seconds=round(time.perf_counter() - start, 4)))
        print(f'TILE {index}/{len(regions)} candidates={len(boxes)}', flush=True)

    # Same pole model and geometry filters as the continuity experiment, with a
    # dynamically generated grid that also supports small and portrait images.
    views = list(dict.fromkeys([(0, 0, width, height)] + tile_boxes(width, height, 1280, 256)))
    proposals = []
    for index, (x0, y0, x1, y1) in enumerate(views, 1):
        crop = image.crop((x0, y0, x1, y1))
        with torch.inference_mode():
            result = model.gdino.predict([crop], ['utility pole. electricity pole. power pole.'], .23, .23)[0]
        for box, score in zip(result['boxes'].tolist(), result['scores'].tolist()):
            a, b, c, d = box
            if d - b < 45 or (d - b) / max(1, c - a) < 1.15:
                continue
            if (y0 and b < 9) or (y1 < height and d > crop.height - 9):
                continue
            bounds = [max(x0, round(a + x0)), max(y0, round(b + y0)),
                      min(x1, round(c + x0)), min(y1, round(d + y0))]
            if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                proposals.append(dict(box=bounds, score=score))
        print(f'POLE_VIEW {index}/{len(views)}', flush=True)

    def overlap(a, b):
        x, y = max(a[0], b[0]), max(a[1], b[1])
        right, bottom = min(a[2], b[2]), min(a[3], b[3])
        area = max(0, right - x) * max(0, bottom - y)
        return area / max(1, min((a[2] - a[0]) * (a[3] - a[1]),
                                (b[2] - b[0]) * (b[3] - b[1])))

    poles = []
    for item in sorted(proposals, key=lambda p: -p['score']):
        if any(overlap(item['box'], kept['box']) > .58 for kept in poles):
            continue
        poles.append(dict(id=f'P{len(poles) + 1:02d}', **item))
        if len(poles) == 8:
            break
    Image.fromarray(union.astype(np.uint8) * 255).save(output / 'mask_union.png')
    write_json(output / 'result.json', dict(status='COMPLETE', image=run['image'],
               image_sha256=run['image_sha256'], image_size_wh=[width, height],
               sam_type='sam2.1_hiera_small', grounding_dino='IDEA-Research/grounding-dino-base',
               versions={name: importlib.metadata.version(name) for name in
                         ['torch', 'torchvision', 'transformers', 'lang-sam', 'sam-2']},
               model_load_seconds=load_seconds, seconds=time.perf_counter() - started,
               gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
               tiles=tiles, candidates=candidates, poles=poles,
               ground_truth_used=False, candidate_count_includes_tile_duplicates=True))
    print(f'DETECTION_COMPLETE candidates={len(candidates)} poles={len(poles)}', flush=True)


def review(run_dir):
    import base64
    import importlib.metadata
    from vllm import LLM, SamplingParams

    run = read_json(run_dir / 'run.json')
    jobs = read_json(run_dir / 'review_jobs.json')
    for job in jobs:
        if digest(run_dir / job['image']) != job['image_sha256']:
            raise ValueError(f'Review image changed: {job["id"]}')
    model = run['config']['qwen_model']
    settings = dict(model=model, max_model_len=2048, max_num_batched_tokens=2048,
                    gpu_memory_utilization=run['config']['gpu_mem'], max_num_seqs=2,
                    limit_mm_per_prompt={'image': 1, 'video': 0}, enforce_eager=True, seed=0,
                    mm_processor_kwargs={'min_pixels': 784, 'max_pixels': 200704})
    llm = LLM(**settings)
    sampling = SamplingParams(temperature=0, max_tokens=16)
    answers = []
    for start in range(0, len(jobs), 2):
        batch = jobs[start:start + 2]
        messages = []
        for job in batch:
            data = base64.b64encode((run_dir / job['image']).read_bytes()).decode('ascii')
            messages.append([dict(role='user', content=[
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + data)),
                dict(type='text', text=job['prompt'])])])
        outputs = llm.chat(messages, sampling_params=sampling, use_tqdm=False)
        if len(outputs) != len(batch):
            raise ValueError('Qwen returned the wrong number of responses')
        for job, result in zip(batch, outputs):
            value = result.outputs[0]
            choice = parse_choice(value.text, value.finish_reason, job['allowed'])
            answers.append(dict(id=job['id'], raw=value.text, choice=choice,
                                finish_reason=value.finish_reason,
                                prompt_tokens=len(result.prompt_token_ids or []),
                                output_tokens=len(value.token_ids),
                                image_sha256=job['image_sha256']))
            print(f'QWEN {job["id"]}: {choice!r}', flush=True)
    usage = {key: sum(item[key] for item in answers) for key in ('prompt_tokens', 'output_tokens')}
    write_json(run_dir / 'qwen_review.json', dict(status='COMPLETE', model=model, settings=settings,
               timestamp_utc=datetime.now(timezone.utc).isoformat(),
               versions={name: importlib.metadata.version(name) for name in ['vllm', 'torch', 'transformers']},
               responses=answers, usage=usage))
    print(f'QWEN_COMPLETE reviews={len(answers)}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['detect', 'prepare', 'review', 'trace'])
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    if args.stage in ('prepare', 'trace'):
        from inspection_core import finish, prepare
        function = prepare if args.stage == 'prepare' else finish
    else:
        function = detect if args.stage == 'detect' else review
    function(args.run_dir.resolve())


if __name__ == '__main__':
    main()
