"""Strict pixel overlap on equal-width prediction and annotation masks."""
from math import fsum
from pathlib import Path

import cv2
import numpy as np

from inspection_io import read_json


def rasterize(polylines, shape, thickness):
    height, width = shape
    lines = []
    for points in polylines:
        points = np.rint(np.asarray(points)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        lines.append(points)
    mask = np.zeros(shape, dtype=np.uint8)
    if lines:
        cv2.polylines(mask, lines, False, 1, thickness, lineType=cv2.LINE_8)
    return mask


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def overlap_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=precision, recall=recall, f1=f1_score(precision, recall),
                tp=tp, fp=fp, fn=fn)


def evaluate_masks(inspection, task, thickness):
    inspection = Path(inspection)
    coords = read_json(inspection / 'line_coordinates.json')
    width, height = task['width'], task['height']
    if coords['coordinate_order'] != 'xy' or coords['image_size_wh'] != [width, height]:
        raise ValueError('Prediction coordinates and annotation dimensions differ')
    shape = (height, width)
    pred = rasterize(coords['paths'], shape, thickness)
    gt = rasterize(task['polylines'], shape, thickness)
    tp = cv2.countNonZero(cv2.bitwise_and(pred, gt))
    fp = cv2.countNonZero(pred) - tp
    fn = cv2.countNonZero(gt) - tp
    return overlap_counts(tp, fp, fn)


def aggregate_overlap(rows):
    if not rows:
        return dict.fromkeys(('macro_precision', 'macro_recall', 'macro_f1',
                              'mean_image_f1', 'micro_precision', 'micro_recall',
                              'micro_f1', 'tp', 'fp', 'fn'))
    precision, recall, mean_f1 = (
        fsum(row[key] for row in rows) / len(rows)
        for key in ('precision', 'recall', 'f1')
    )
    micro = overlap_counts(*(sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn')))
    # The reference study computes its headline F1 from the two image averages.
    return dict(macro_precision=precision, macro_recall=recall,
                macro_f1=f1_score(precision, recall), mean_image_f1=mean_f1,
                micro_precision=micro['precision'], micro_recall=micro['recall'],
                micro_f1=micro['f1'], tp=micro['tp'], fp=micro['fp'], fn=micro['fn'])
