"""Resumable, supervised dataset inference and centerline precision/recall."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from inspection_io import ROOT, default_qwen_model, digest, read_json

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
FIELDS = ['image', 'precision', 'recall', 'status', 'attempts', 'output', 'error']


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def discover_dataset(dataset):
    """Match actual images to CVAT records inside their own annotation group."""
    from PIL import Image
    dataset = Path(dataset).resolve()
    images = sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f'No images found: {dataset}')
    matched = {}
    for xml in sorted(dataset.rglob('*.xml')):
        tree = ET.parse(xml).getroot()
        if tree.tag != 'annotations':
            continue
        group_images = [p for p in images if p.is_relative_to(xml.parent)]
        xml_hash = digest(xml)
        for item in tree.findall('image'):
            filename = Path(item.attrib['name']).name
            candidates = [p for p in group_images if p.name == filename]
            direct = xml.parent / item.attrib['name']
            if direct in candidates:
                candidates = [direct]
            if len(candidates) != 1:
                raise ValueError(f'Annotation has {len(candidates)} matching images: {xml}: {filename}')
            image = candidates[0]
            if image in matched:
                raise ValueError(f'Multiple annotations match: {image}')
            width, height = int(item.attrib['width']), int(item.attrib['height'])
            with Image.open(image) as source:
                if source.size != (width, height) or source.getexif().get(274, 1) != 1:
                    raise ValueError(f'Annotation dimensions/EXIF orientation need review: {image}')
            polylines = []
            for line in item.findall('polyline'):
                if line.attrib.get('label') not in ('powerline', 'polyline'):
                    continue
                points = [[float(v) for v in pair.split(',')] for pair in line.attrib['points'].split(';')]
                if len(points) < 2:
                    raise ValueError(f'Invalid polyline: {xml}: {filename}')
                polylines.append(points)
            key = image.relative_to(dataset).as_posix()
            matched[image] = dict(key=key, image=str(image), image_sha256=digest(image),
                                  group=xml.parent.relative_to(dataset).as_posix(),
                                  width=width, height=height, polylines=polylines,
                                  annotation=str(xml), annotation_sha256=xml_hash,
                                  annotation_image_name=item.attrib['name'])
    missing = [str(p) for p in images if p not in matched]
    if missing:
        raise ValueError(f'{len(missing)} images have no annotation: {missing[:5]}')
    return [matched[p] for p in images]


def evaluate_prediction(path, task, tolerance):
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    with Image.open(path) as image:
        prediction = np.asarray(image.convert('L')) > 0
    if prediction.shape != (task['height'], task['width']):
        raise ValueError('Prediction and annotation dimensions differ')
    gt = np.zeros(prediction.shape, dtype=np.uint8)
    for points in task['polylines']:
        raster = np.rint(np.asarray(points)).astype(np.int32)
        raster[:, 0] = np.clip(raster[:, 0], 0, task['width'] - 1)
        raster[:, 1] = np.clip(raster[:, 1], 0, task['height'] - 1)
        cv2.polylines(gt, [raster], False, 1, 1, lineType=cv2.LINE_8)
    gt = gt.astype(bool)
    n, g = int(prediction.sum()), int(gt.sum())
    matched = int((distance_transform_edt(~gt)[prediction] <= tolerance).sum()) if n and g else 0
    covered = int((distance_transform_edt(~prediction)[gt] <= tolerance).sum()) if n and g else 0
    return dict(precision=matched / n if n else 0.0, recall=covered / g if g else 0.0,
                predicted_pixels=n, reference_pixels=g, matched_predicted_pixels=matched,
                covered_reference_pixels=covered, prediction_sha256=digest(path))


def publish(output, state):
    state['updated_at_utc'] = now()
    records = state['records']
    completed = [r for r in records if r['status'] == 'COMPLETE']
    counts = Counter(r['status'] for r in records)
    predicted = sum(r.get('predicted_pixels', 0) for r in completed)
    reference = sum(r.get('reference_pixels', 0) for r in completed)
    summary = dict(status=state['status'], updated_at_utc=state['updated_at_utc'],
                   total=len(records), completed=len(completed), failed=counts['FAILED'],
                   pending=counts['PENDING'], running=counts['RUNNING'],
                   current_image=state.get('current_image'), current_stage=state.get('current_stage'),
                   tolerance_px=state['config']['tolerance_px'],
                   qwen_model=state['config']['qwen_model'], value_range='0 to 1',
                   evaluation_complete=len(completed) == len(records),
                   macro_precision=sum(r['precision'] for r in completed) / len(completed) if completed else None,
                   macro_recall=sum(r['recall'] for r in completed) / len(completed) if completed else None,
                   micro_precision=(sum(r['matched_predicted_pixels'] for r in completed) / predicted
                                    if predicted else 0.0) if completed else None,
                   micro_recall=(sum(r['covered_reference_pixels'] for r in completed) / reference
                                 if reference else 0.0) if completed else None,
                   aggregation_scope='completed images only; see coverage counts',
                   groups={group: dict(total=sum(r['group'] == group for r in records),
                                       completed=sum(r['group'] == group for r in completed))
                           for group in sorted({r['group'] for r in records})})
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction='ignore')
    writer.writeheader()
    for record in records:
        row = {key: record.get(key, '') for key in FIELDS}
        if record['status'] != 'COMPLETE':
            row['precision'] = row['recall'] = ''
        writer.writerow(row)
    # The durable state is authoritative; the CSV and summary can be regenerated.
    save_json(output / 'state.json', state)
    atomic_text(output / 'metrics.csv', '\ufeff' + buffer.getvalue())
    save_json(output / 'summary.json', summary)


def freeze_source(output):
    source = output / 'source'
    source.mkdir()
    shutil.copytree(ROOT / 'pipeline', source / 'pipeline', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    vendor = source / 'vendor/lang-segment-anything'
    vendor.mkdir(parents=True)
    shutil.copytree(ROOT / 'vendor/lang-segment-anything/lang_sam', vendor / 'lang_sam',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('.venv-eval', '.venv-langsam', '.venv-vlm', '.cache'):
        (source / name).symlink_to((ROOT / name).resolve(), target_is_directory=True)
    hashes = {p.relative_to(source).as_posix(): digest(p) for folder in (source / 'pipeline', source / 'vendor')
              for p in folder.rglob('*') if p.is_file()}
    save_json(output / 'source_manifest.json', dict(created_at_utc=now(), files=hashes))


def prepare(args):
    output = args.output.expanduser().resolve()
    config = dict(dataset=str(args.dataset.expanduser().resolve()), qwen_model=args.qwen_model,
                  tolerance_px=args.tolerance_px, gpu_mem=args.gpu_mem,
                  max_attempts=args.max_attempts, timeout_seconds=args.timeout_seconds,
                  retry_delay=args.retry_delay, max_gpu_used_mib=args.max_gpu_used_mib)
    model = Path(config['qwen_model']).expanduser()
    if not model.is_dir():
        raise ValueError('Use a complete local Qwen model snapshot; download the model before starting the batch')
    config['qwen_model'] = str(model.resolve())
    index = read_json(model / 'model.safetensors.index.json')
    if not all((model / name).is_file() for name in set(index['weight_map'].values())):
        raise ValueError('Qwen weights are incomplete')
    if (output / 'state.json').exists():
        previous = read_json(output / 'state.json')['config']
        if config != previous:
            raise ValueError('Resume settings differ from the saved batch settings; use the original settings')
        return output
    if output.exists():
        raise ValueError(f'Output exists without a state checkpoint: {output}')
    tasks = discover_dataset(args.dataset)
    if args.expected_count is not None and len(tasks) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} images, found {len(tasks)}')
    if args.priority_image:
        matches = [t for t in tasks if t['key'] == args.priority_image or Path(t['key']).name == args.priority_image]
        if len(matches) != 1:
            raise ValueError('Priority image must identify exactly one dataset image')
        tasks = matches + [t for t in tasks if t is not matches[0]]
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    (output / 'logs').mkdir()
    save_json(output / 'tasks.json', tasks)
    freeze_source(output)
    records = [dict(image=t['key'], group=t['group'], status='PENDING', attempts=0,
                    output='', error='') for t in tasks]
    state = dict(status='STARTING', created_at_utc=now(), config=config, records=records,
                 current_image=None, current_stage=None)
    publish(output, state)
    print(f'BATCH_PREPARED {len(tasks)} images -> {output}', flush=True)
    return output


def lock_file(path):
    stream = Path(path).open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f'Another batch process holds {path}')
    return stream


def gpu_used():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                   text=True, timeout=15)
    return max(int(line.strip()) for line in text.splitlines() if line.strip())


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def process_start_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def cleanup_recorded_processes(state):
    """Only terminate our recorded image process groups, never reused PIDs."""
    for row in state['records']:
        pid, ticks = row.get('process_pid'), row.get('process_start_ticks')
        if not pid or ticks is None or process_start_ticks(pid) != ticks:
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(50):
            if process_start_ticks(pid) != ticks:
                break
            time.sleep(.2)
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_image(output, state, row, task):
    config = state['config']
    # Do not compete with unrelated GPU work, and never kill another user's job.
    while gpu_used() > config['max_gpu_used_mib']:
        state['status'], state['current_stage'] = 'WAITING_GPU', 'waiting for an idle GPU'
        publish(output, state)
        time.sleep(15)
    if digest(task['image']) != task['image_sha256']:
        raise ValueError('Dataset image changed since the batch manifest was created')
    row['attempts'] += 1
    key = hashlib.sha256(task['key'].encode()).hexdigest()[:10]
    folder = output / 'images' / task['group'] / f'{Path(task["image"]).stem}_{key}'
    folder.mkdir(parents=True, exist_ok=True)
    attempt = folder / f'attempt_{row["attempts"]:03d}'
    if attempt.exists():
        attempt = folder / f'attempt_{row["attempts"]:03d}_{time.time_ns()}'
    row.update(status='RUNNING', output=str(attempt), error='')
    state.update(status='RUNNING', current_stage='starting inference')
    publish(output, state)
    script = output / 'source/pipeline/inspect_image.py'
    command = [str(output / 'source/.venv-eval/bin/python'), '-u', str(script),
               '--image', task['image'], '--output', str(attempt), '--qwen-model', config['qwen_model'],
               '--gpu-mem', str(config['gpu_mem'])]
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    started = time.monotonic()
    with (folder / f'attempt_{row["attempts"]:03d}.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=output / 'source', env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        row.update(process_pid=process.pid, process_start_ticks=process_start_ticks(process.pid))
        publish(output, state)
        try:
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started > config['timeout_seconds']:
                        raise TimeoutError(f'Image exceeded {config["timeout_seconds"]} seconds')
                    run = attempt / 'run.json'
                    if run.exists():
                        data = read_json(run)
                        state['current_stage'] = data['stages'][-1]['name'] if data.get('stages') else 'starting'
                    publish(output, state)
            if code:
                raise RuntimeError(f'Inference exited with code {code}; see {folder}/attempt_{row["attempts"]:03d}.log')
        finally:
            terminate_group(process)
    row['inference_seconds'] = round(time.monotonic() - started, 3)


def collect_result(row, task, config):
    attempt = Path(row['output'])
    run = read_json(attempt / 'run.json')
    if run['status'] != 'COMPLETE' or run['image_sha256'] != task['image_sha256']:
        raise ValueError('Incomplete run or mismatched input hash')
    if str(Path(run['config']['qwen_model']).resolve()) != config['qwen_model']:
        raise ValueError('The model used for this image differs from the batch model')
    scores = evaluate_prediction(attempt / 'prediction_centerline.png', task, config['tolerance_px'])
    row.update(scores, status='COMPLETE', error='', completed_at_utc=now())


def worker(output):
    lock = lock_file(output / '.worker.lock')
    state = read_json(output / 'state.json')
    tasks = read_json(output / 'tasks.json')
    config = state['config']
    if len(tasks) != len(state['records']) or any(r['image'] != t['key'] for r, t in zip(state['records'], tasks)):
        raise ValueError('Batch state and task manifest do not match')
    cleanup_recorded_processes(state)
    for row, task in zip(state['records'], tasks):
        state['current_image'] = row['image']
        if row['status'] == 'COMPLETE':
            continue
        # Recover a run that finished just before a controller restart.
        if row.get('output') and (Path(row['output']) / 'run.json').exists():
            try:
                collect_result(row, task, config)
                publish(output, state)
                continue
            except (ValueError, OSError, KeyError):
                pass
        if row['attempts'] >= config['max_attempts']:
            row.update(status='FAILED', error=row.get('error') or 'Retry limit reached after interruption')
            publish(output, state)
            continue
        while row['attempts'] < config['max_attempts']:
            previous_attempts = row['attempts']
            print(f'IMAGE_START {row["image"]} attempt={row["attempts"]+1}', flush=True)
            try:
                run_image(output, state, row, task)
                state['current_stage'] = 'evaluating'
                collect_result(row, task, config)
                publish(output, state)
                print(f'IMAGE_COMPLETE {row["image"]} precision={row["precision"]:.6f} recall={row["recall"]:.6f}', flush=True)
                break
            except Exception as exc:
                if row['attempts'] == previous_attempts:
                    row['attempts'] += 1
                row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
                publish(output, state)
                print(f'IMAGE_FAILED {row["image"]}: {row["error"]}', flush=True)
                if row['attempts'] < config['max_attempts']:
                    time.sleep(config['retry_delay'])
    failures = sum(row['status'] != 'COMPLETE' for row in state['records'])
    state.update(status='COMPLETE_WITH_ERRORS' if failures else 'COMPLETE',
                 current_image=None, current_stage=None, finished_at_utc=now())
    publish(output, state)
    print(f'BATCH_FINISHED status={state["status"]} failures={failures}', flush=True)
    lock.close()
    return 2 if failures else 0


def supervise(output):
    lock = lock_file(output / '.supervisor.lock')
    save_json(output / 'supervisor.json', dict(pid=os.getpid(), started_at_utc=now()))
    command = [str(output / 'source/.venv-eval/bin/python'), '-u',
               str(output / 'source/pipeline/run_dataset.py'), '--worker', '--output', str(output)]
    while True:
        print(f'SUPERVISOR_START_WORKER {now()}', flush=True)
        process = subprocess.Popen(command, start_new_session=True)
        try:
            code = process.wait()
        finally:
            terminate_group(process)
            cleanup_recorded_processes(read_json(output / 'state.json'))
        state = read_json(output / 'state.json')
        if state['status'] in ('COMPLETE', 'COMPLETE_WITH_ERRORS'):
            print(f'SUPERVISOR_FINISHED {state["status"]} code={code}', flush=True)
            lock.close()
            return code
        print(f'SUPERVISOR_RESTART code={code}; resuming checkpoint in 30 seconds', flush=True)
        time.sleep(30)


def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qwen-model', default=default_qwen_model())
    parser.add_argument('--tolerance-px', type=int, default=5)
    parser.add_argument('--gpu-mem', type=float, default=.35)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--retry-delay', type=int, default=30)
    parser.add_argument('--max-gpu-used-mib', type=int, default=1024)
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--priority-image')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.output.resolve())
    if (args.tolerance_px < 0 or not 0 < args.gpu_mem < 1 or args.max_attempts < 1 or
            args.timeout_seconds < 1 or args.retry_delay < 0):
        parser.error('Invalid tolerance, memory fraction, attempt count, timeout or retry delay')
    output = prepare(args)
    return 0 if args.prepare_only else supervise(output)


if __name__ == '__main__':
    sys.exit(main())
