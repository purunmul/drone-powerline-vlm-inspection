"""Shared run files and coordinate conventions; no model imports."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
QWEN_REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
SCENE_PROMPT = ('Does this aerial image visibly contain overhead electrical wires? '
                'Road markings, fences and building edges alone are not wires. '
                'Reply with exactly WIRE, NONE or UNSURE.')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def open_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('RGB')


def tile_boxes(width, height, size, overlap):
    if min(width, height, size) <= 0 or not 0 <= overlap < size:
        raise ValueError('Positive dimensions and 0 <= overlap < tile size are required')

    def origins(length):
        last = max(0, length - size)
        values = list(range(0, last + 1, size - overlap))
        if values[-1] != last:
            values.append(last)
        return values

    return [(x, y, min(x + size, width), min(y + size, height))
            for y in origins(height) for x in origins(width)]


def parse_choice(raw, finish_reason, allowed):
    choice = raw.strip().strip('.*!').upper()
    return choice if finish_reason == 'stop' and choice in allowed else None


def default_qwen_model():
    home = Path(os.environ.get('HF_HOME', ROOT / '.cache/huggingface'))
    snapshot = home / 'hub' / ('models--' + QWEN_ID.replace('/', '--')) / 'snapshots' / QWEN_REVISION
    index = snapshot / 'model.safetensors.index.json'
    if index.is_file() and (snapshot / 'config.json').is_file():
        weights = set(read_json(index)['weight_map'].values())
        if weights and all((snapshot / name).is_file() for name in weights):
            return str(snapshot)
    return QWEN_ID
