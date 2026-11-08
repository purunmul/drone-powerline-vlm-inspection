import argparse
import base64
import hashlib
import importlib.metadata
import json
import shutil
import time
import zipfile
from pathlib import Path

OUT = Path('results/semantic_rereview_20261107_v1')
TRANSFER = Path('results/dev_transfer_20261107_v1')
FIRST_REVIEW = Path('results/vlm_review/BEGO7553_frame_001098_review_v4')
ABLATION = Path('results/skeleton_order_ablation_20261107_v1')
CASES = [
    ('BEGO7553_frame_001098', 'P02', Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1'), FIRST_REVIEW, Path('results/diagnostic_eval/BEGO7553_frame_001098_v1')),
    ('PUYL5573_frame_000165', 'P01', TRANSFER/'PUYL5573_frame_000165/candidates', TRANSFER/'PUYL5573_frame_000165/vlm', TRANSFER/'PUYL5573_frame_000165/evaluation'),
]
PAIRED_PROMPT = (
    'This image has two aligned panels of one crop. The LEFT panel contains only the selected foreground pixels on a uniform gray background. '
    'The RIGHT panel may show the original crop as context or may be blank. Use it only to interpret pixels selected in the LEFT panel. '
    'Classify the visible selected foreground in the LEFT panel. Objects visible only in the RIGHT panel are outside the selection. '
    'If the LEFT panel has no visible foreground or its identity is unresolved, set both category and visibility to unclear. '
    'Return one JSON object with category, visibility, evidence. Category must be one of vehicle, road_or_ground, wire_or_cable, pole, building_edge, vegetation, other, unclear. '
    'Visibility must be clear or unclear. Evidence must briefly describe what is visible in the selected foreground.'
)
CATEGORIES = {'vehicle', 'road_or_ground', 'wire_or_cable', 'pole', 'building_edge', 'vegetation', 'other', 'unclear'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def infer():
    from PIL import Image
    from vllm import LLM, SamplingParams

    OUT.mkdir(parents=True, exist_ok=False)
    baseline = read_json(FIRST_REVIEW/'run_config.json')
    config = {
        'v4_settings': baseline['settings'], 'sampling': baseline['sampling'],
        'v4_prompt': baseline['prompt'], 'paired_prompt': PAIRED_PROMPT,
        'cases': [[name, ident] for name, ident, *_ in CASES],
        'selection': 'two preselected reference-informed development failure cases',
        'conditions': ['v4_replay', 'paired_no_context', 'paired_context', 'paired_empty_target'],
        'control_gate': 'the matched empty target must return unclear category and unclear visibility; case replay must reproduce original category and visibility',
        'context_effect_comparison': 'paired_no_context versus paired_context, identical target, geometry, prompt and sampling',
        'input_pixels': 'exact saved v4 JPEG for replay; decoded target pixels for paired PNGs',
        'ground_truth_in_inference': False,
        'versions': {p: importlib.metadata.version(p) for p in ['vllm', 'torch', 'transformers', 'Pillow']},
    }
    save_json(OUT/'run_config.json', config)
    started = time.perf_counter()
    model = LLM(**baseline['settings'])
    load_seconds = time.perf_counter() - started
    sample = SamplingParams(**baseline['sampling'])
    image_token = model.get_tokenizer().convert_tokens_to_ids('<|image_pad|>')

    def ask(path, prompt):
        mime = 'image/png' if path.suffix == '.png' else 'image/jpeg'
        uri = f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
        start = time.perf_counter()
        response = model.chat([{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': uri}},
            {'type': 'text', 'text': prompt},
        ]}], sampling_params=sample, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        output = response.outputs[0]
        token_count = response.prompt_token_ids.count(image_token)
        assert token_count > 0
        parsed, error = None, None
        try:
            value = json.JSONDecoder().raw_decode(output.text[output.text.index('{'):])[0]
            assert value['category'] in CATEGORIES
            assert value['visibility'] in {'clear', 'unclear'}
            assert isinstance(value['evidence'], str) and value['evidence'].strip()
            assert output.finish_reason == 'stop'
            parsed = value
        except (ValueError, KeyError, AssertionError, TypeError) as exc:
            error = repr(exc)
        return {'input': str(path.relative_to(OUT)), 'sha256': sha(path), 'image_tokens': token_count,
                'seconds': elapsed, 'raw': output.text, 'parsed': parsed,
                'parse_error': error, 'finish_reason': output.finish_reason}

    cases = []
    for name, ident, src, review, _ in CASES:
        folder = OUT/f'{name}_{ident}'
        folder.mkdir()
        candidate = next(c for c in read_json(src/'candidates.json')['candidates'] if c['id'] == ident)
        previous = next(r for r in read_json(review/'review.json')['results'] if r['candidate_id'] == ident)
        original_input = review/f'{ident}_isolated.jpg'
        assert sha(original_input) == previous['input_sha256']
        mask_path = src/candidate['mask']
        mask_sha = sha(mask_path)
        replay = folder/'v4_replay.jpg'
        shutil.copyfile(original_input, replay)
        target = Image.open(replay).convert('RGB')
        context = Image.open(src/candidate['original_context']).convert('RGB')
        assert target.size == context.size
        gray = Image.new('RGB', target.size, (150, 150, 150))
        width, height = target.size
        paired_inputs = {
            'paired_no_context': (target, gray),
            'paired_context': (target, context),
            'paired_empty_target': (gray, context),
        }
        results = {'v4_replay': ask(replay, baseline['prompt'])}
        for label, (left, right) in paired_inputs.items():
            board = Image.new('RGB', (2*width+12, height), 'white')
            board.paste(left, (0, 0))
            board.paste(right, (width+12, 0))
            path = folder/f'{label}.png'
            board.save(path)
            results[label] = ask(path, PAIRED_PROMPT)
        assert results['paired_no_context']['image_tokens'] == results['paired_context']['image_tokens'] == results['paired_empty_target']['image_tokens']
        old = previous['parsed']
        replayed = results['v4_replay']['parsed'] or {}
        with_context = results['paired_context']['parsed'] or {}
        empty = results['paired_empty_target']['parsed'] or {}
        replay_match = all(replayed.get(k) == old[k] for k in ('category', 'visibility'))
        control_pass = empty.get('category') == 'unclear' and empty.get('visibility') == 'unclear'
        eligible = replay_match and control_pass and with_context.get('category') == 'wire_or_cable' and with_context.get('visibility') == 'clear'
        assert sha(mask_path) == mask_sha
        case = {'image': name, 'candidate_id': ident, 'original': old, 'results': results,
                'baseline_replay_match': replay_match, 'empty_target_control_pass': control_pass,
                'eligible_for_diagnostic_addback': eligible, 'source_mask_sha256': mask_sha}
        cases.append(case)
        save_json(folder/'review.json', case)
        print('CASE', name, ident, {k: (v['parsed'] or {}).get('category') for k,v in results.items()}, 'CONTROL', control_pass, 'ADD', eligible, flush=True)
    save_json(OUT/'inference.json', {'cases': cases, 'model_load_seconds': load_seconds})


def evaluate():
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from evaluate_first_image_v1 import metrics, readmask

    summary = read_json(OUT/'inference.json')
    for case, (name, ident, src, _, ev) in zip(summary['cases'], CASES, strict=True):
        assert (case['image'], case['candidate_id']) == (name, ident)
        candidate = next(c for c in read_json(src/'candidates.json')['candidates'] if c['id'] == ident)
        assert sha(src/candidate['mask']) == case['source_mask_sha256']
        if not case['eligible_for_diagnostic_addback']:
            case['addback'] = {'status': 'not_eligible'}
            continue
        path = ABLATION/name
        old = readmask(path/'B_vlm_keep_only/skeleton_then_union/centerline.png')
        candidate_line = readmask(path/'candidate_skeletons'/f'{ident}.png')
        combined = old | candidate_line
        gt = readmask(ev/'reference_centerlines.png')
        refdist = distance_transform_edt(~gt)
        baseline_metrics, _ = metrics(old, gt, refdist)
        revised_metrics, _ = metrics(combined, gt, refdist)
        assert all(r['covered_reference_pixels'] >= b['covered_reference_pixels'] for b, r in zip(baseline_metrics, revised_metrics, strict=True))
        saved = read_json(path/'summary.json')
        expected = next(s['metrics'] for s in saved['conditions'] if s['condition']=='B_vlm_keep_only' and s['method']=='skeleton_then_union')
        assert baseline_metrics == [{k:r[k] for k in baseline_metrics[0]} for r in expected]
        case['addback'] = {'status': 'development_counterfactual', 'baseline': baseline_metrics, 'revised': revised_metrics, 'new_centerline_pixels': int(np.count_nonzero(combined & ~old))}
        Image.fromarray(combined.astype('uint8')*255).save(OUT/f'{name}_{ident}'/'diagnostic_addback_centerline.png')
        print('ADDBACK', name, baseline_metrics[1]['centerline_f1'], revised_metrics[1]['centerline_f1'], flush=True)
    summary.update(status='SEMANTIC_REREVIEW_COMPLETE', selection_is_reference_informed=True, evaluation_split='development', decisions_applied_to_pipeline=False)
    save_json(OUT/'summary.json', summary)
    with zipfile.ZipFile('semantic_rereview_20261107_v1.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in OUT.rglob('*'):
            if file.is_file():
                archive.write(file, file.relative_to(OUT.parent))
        archive.write(__file__, Path(__file__).name)
    print('SEMANTIC_REREVIEW_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['infer', 'evaluate'])
    phase = parser.parse_args().phase
    (infer if phase == 'infer' else evaluate)()
