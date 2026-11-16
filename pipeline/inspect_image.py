"""Run fresh detection, Qwen review and continuity tracing for one image."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from inspection_io import ROOT, default_qwen_model, digest, open_image, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, help='New output directory; relative to caller directory')
    p.add_argument('--prompt', default='power line.')
    p.add_argument('--box-threshold', type=float, default=.30)
    p.add_argument('--text-threshold', type=float, default=.25)
    p.add_argument('--tile-size', type=int, default=1024)
    p.add_argument('--overlap', type=int, default=256)
    p.add_argument('--qwen-model', default=default_qwen_model())
    p.add_argument('--gpu-mem', type=float, default=.35)
    p.add_argument('--max-review-cases', type=int, default=4)
    p.add_argument('--min-length', type=int, default=160)
    p.add_argument('--min-terminal-length', type=int, default=80)
    p.add_argument('--max-extension', type=int, default=192)
    p.add_argument('--recovery-turns', type=int, default=0,
                   help='Enable Qwen wire recovery with this many turns per region (0 disables it)')
    p.add_argument('--recovery-regions', type=int, default=12,
                   help='Maximum number of suspected gaps or missing fragments to review')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.image = args.image.expanduser()
    if not args.image.is_file():
        p.error(f'Image does not exist: {args.image}')
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        p.error('Require tile-size > 0 and 0 <= overlap < tile-size')
    if not 0 < args.gpu_mem < 1 or args.max_review_cases < 1:
        p.error('Require 0 < gpu-mem < 1 and max-review-cases >= 1')
    if not 0 <= args.box_threshold <= 1 or not 0 <= args.text_threshold <= 1:
        p.error('Detection thresholds must be in [0, 1]')
    if not 0 < args.min_terminal_length <= args.min_length or args.max_extension < 0:
        p.error('Require 0 < min-terminal-length <= min-length and max-extension >= 0')
    if not args.prompt.strip():
        p.error('Prompt must not be empty')
    if args.recovery_turns == 1 or args.recovery_turns < 0 or args.recovery_regions < 1:
        p.error('Recovery needs at least two turns and a positive region limit, or zero turns to disable')
    if args.recovery_turns and args.prompt.strip().lower().rstrip('.') != 'power line':
        p.error('Recovery supports the power line prompt only')
    image = args.image.resolve()
    model_path = Path(args.qwen_model).expanduser()
    if model_path.is_dir():
        args.qwen_model = str(model_path.resolve())
    try:
        size = open_image(image).size
    except Exception as exc:
        p.error(f'Cannot read image: {exc}')
    interpreters = {stage: ROOT / env / 'bin/python' for stage, env in
                    [('detect', '.venv-langsam'), ('prepare', '.venv-eval'),
                     ('review', '.venv-vlm'), ('trace', '.venv-eval')]}
    if args.recovery_turns:
        interpreters['recover'] = ROOT / '.venv-vlm/bin/python'
    for executable in set(interpreters.values()):
        if not os.access(executable, os.X_OK):
            p.error(f'Missing environment interpreter: {executable}')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output or ROOT / 'results/inspections' / f'{image.stem}_{stamp}').expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        p.error(f'Output already exists; choose a new directory: {output}')
    (output / 'logs').mkdir()
    config = {k: v for k, v in vars(args).items() if k not in ('image', 'output')}
    run = dict(status='RUNNING', started_at_utc=datetime.now(timezone.utc).isoformat(),
               image=str(image), image_sha256=digest(image), image_size_wh=list(size),
               image_orientation='EXIF-transposed', output=str(output), config=config,
               ground_truth_used=False, stages=[])
    manifest = output / 'run.json'
    write_json(manifest, run)
    started = time.perf_counter()
    labels = dict(detect='LangSAM tiles + pole detection', prepare='Continuity geometry + review inputs',
                  review='Qwen image review', trace='Centerline tracing + visualization',
                  recover='Multi-turn Qwen wire recovery')
    try:
        for index, (stage, executable) in enumerate(interpreters.items(), 1):
            log = output / 'logs' / f'{stage}.log'
            record = dict(name=stage, status='RUNNING', log=str(log.relative_to(output)))
            run['stages'].append(record)
            write_json(manifest, run)
            print(f'[{index}/{len(interpreters)}] {labels[stage]}', flush=True)
            env = os.environ.copy()
            for key in ('PYTHONPATH', 'PYTHONHOME'):
                env.pop(key, None)
            env.update(PYTHONNOUSERSITE='1', HF_HOME=str(ROOT / '.cache/huggingface'),
                       HF_HUB_DISABLE_XET='1', TORCH_HOME=str(ROOT / '.cache/torch'),
                       TORCH_EXTENSIONS_DIR=str(ROOT / '.cache/torch_extensions'))
            if stage == 'detect':
                env['PYTHONPATH'] = str(ROOT / 'vendor/lang-segment-anything')
            stage_start = time.perf_counter()
            with log.open('w', encoding='utf-8') as stream:
                process = subprocess.run([str(executable), '-u', str(ROOT / 'pipeline/inspection_worker.py'),
                                          stage, '--run-dir', str(output)],
                                         cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record.update(seconds=round(time.perf_counter() - stage_start, 3),
                          returncode=process.returncode,
                          status='COMPLETE' if process.returncode == 0 else 'FAILED')
            write_json(manifest, run)
            if process.returncode:
                tail = '\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-18:])
                raise RuntimeError(f'{stage} failed; see {log}\n{tail}')
            print(f'      complete ({record["seconds"]:.1f}s)', flush=True)
        run.update(status='COMPLETE', seconds=round(time.perf_counter() - started, 3),
                   finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest, run)
        print(f'COMPLETE {output}\nPreview: {output / "preview.jpg"}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if run['stages'] and run['stages'][-1]['status'] == 'RUNNING':
            run['stages'][-1]['status'] = 'FAILED'
        run.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                   error=str(exc), seconds=round(time.perf_counter() - started, 3))
        write_json(manifest, run)
        print(f'{run["status"]}: {exc}', file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    sys.exit(main())
