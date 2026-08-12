"""Evaluation of detections against ground truth (TP/FP/FN, precision, recall)
and export of per-detection results to CSV.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from annotations import Rect
from detection import Detection
from geometry import intersection_over_union


@dataclass
class EvaluationCounters:
    """Running TP/FP/FN counters accumulated over the whole dataset."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    total_signs: int = 0
    # One row per detection: (image_name, x, y, w, h, match_score, is_true_positive)
    csv_rows: list[tuple] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return 100.0 * self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return 100.0 * self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0


def evaluate_image(
    image_name: str,
    detections: list[Detection],
    ground_truth: list[Rect],
    counters: EvaluationCounters,
    iou_threshold: float,
) -> None:
    """Match detections against ground-truth boxes and update the counters.

    A detection is a true positive if its IoU with at least one *unmatched*
    ground-truth box exceeds `iou_threshold`; each ground-truth box can be
    matched at most once, so duplicate detections on the same box count as
    false positives. Ground-truth boxes never matched by any detection count
    as false negatives.
    """
    counters.total_signs += len(ground_truth)
    matched_gt: set[int] = set()

    for det in detections:
        hit = None
        for gt_idx, gt_rect in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if intersection_over_union(det.rect, gt_rect) > iou_threshold:
                hit = gt_idx
                break

        if hit is not None:
            counters.tp += 1
            matched_gt.add(hit)
        else:
            counters.fp += 1

        counters.csv_rows.append(
            (image_name, *det.rect, round(det.match_score, 6), int(hit is not None))
        )

    counters.fn += len(ground_truth) - len(matched_gt)


def save_results_csv(counters: EvaluationCounters, csv_path: Path) -> None:
    """Write one row per detection, plus a header, to `csv_path`."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x", "y", "width", "height", "match_score", "true_positive"])
        writer.writerows(counters.csv_rows)
