"""Compare two completed batches over identical images and annotations."""
import argparse
from math import fsum, isclose
from pathlib import Path
import time

from inspection_io import read_json
from mask_overlap import f1_score, overlap_counts
from run_dataset import atomic_text, save_json


def completed_macro(batch, state, tasks, expected_count):
    records = {row['image']: row for row in state['records']}
    if (state['status'] != 'COMPLETE' or len(tasks) != expected_count or
            len(state['records']) != expected_count or records.keys() != tasks.keys() or
            any(row['status'] != 'COMPLETE' for row in records.values())):
        raise ValueError(f'{batch} does not have {expected_count} completed images')
    scores = []
    for row in records.values():
        counts = [row[key] for key in ('tp', 'fp', 'fn')]
        score = overlap_counts(*counts)
        if any(not isclose(score[key], row[key], abs_tol=1e-12)
               for key in ('precision', 'recall', 'f1')):
            raise ValueError(f"Inconsistent pixel counts for {row['image']}")
        scores.append(score)
    precision = fsum(row['precision'] for row in scores) / len(scores)
    recall = fsum(row['recall'] for row in scores) / len(scores)
    return dict(precision=precision, recall=recall, f1=f1_score(precision, recall))


def compare(baseline, candidate, expected_count, wait):
    a = read_json(baseline / 'state.json')
    tasks_a = {task['key']: task for task in read_json(baseline / 'tasks.json')}
    before = completed_macro(baseline, a, tasks_a, expected_count)
    tasks_b = {task['key']: task for task in read_json(candidate / 'tasks.json')}
    if tasks_a != tasks_b:
        raise ValueError('Image or annotation manifests differ')
    while True:
        b = read_json(candidate / 'state.json')
        if not wait or b['status'] in ('COMPLETE', 'COMPLETE_WITH_ERRORS'):
            break
        time.sleep(30)
    for key in ('mask_thickness_px', 'qwen_model'):
        if not a['config'].get(key) or a['config'][key] != b['config'].get(key):
            raise ValueError(f'Batch settings differ for {key}')
    after = completed_macro(candidate, b, tasks_b, expected_count)
    return dict(
        images=expected_count, mask_thickness_px=a['config']['mask_thickness_px'],
        macro_f1_definition='harmonic mean of per-image mean precision and recall',
        baseline=before, recovery=after,
        delta_percentage_points={key: 100 * (after[key] - before[key]) for key in before},
        baseline_batch=str(baseline.resolve()), recovery_batch=str(candidate.resolve()),
        recovery_finished_at_utc=b['finished_at_utc'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=200)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.expected_count, args.wait)
    report = [
        '# 원본 해상도 {}px Macro 비교 ({}장)'.format(result['mask_thickness_px'], result['images']),
        '',
        '| 방식 | Precision | Recall | F1 |',
        '|---|---:|---:|---:|',
    ]
    for key, name in [('baseline', '기존'), ('recovery', 'Qwen 복구')]:
        report.append('| {} | {:.2%} | {:.2%} | {:.2%} |'.format(name, *result[key].values()))
    report.append('| 변화 | {precision:+.2f}%p | {recall:+.2f}%p | {f1:+.2f}%p |'.format(
        **result['delta_percentage_points']))
    report.extend(('', 'Macro F1은 사진별 평균 Precision과 Recall의 조화평균입니다.', ''))
    markdown = '\n'.join(report)
    save_json(args.candidate / 'macro_comparison.json', result)
    atomic_text(args.candidate / 'macro_comparison.md', markdown)
    print(markdown)


if __name__ == '__main__':
    main()
