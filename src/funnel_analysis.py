"""Stage-by-stage funnel analysis: where does the pipeline lose the signs?

For every ground-truth sign of detectable size (area within the candidate
area limits), the script finds the deepest pipeline stage that still covers
it, using the detections previously exported to CSV by main.py:

  detected             the final detector output hits it (IoU > threshold)
  triangle/pictogram   a plate candidate survives every filter, but the
                       pictogram verification fails inside the crop
  rectangularity       covered by a mask contour of valid area only
  area filter          covered by a mask contour of the wrong size
  color mask           never covered by any contour of the candidate mask

A candidate covers a sign when its bounding box overlaps at least half of
the ground-truth box.

Usage (from the repository root, after a main.py run on the same folder):
    python src/funnel_analysis.py --data-dir data/raw/heldout-crossing \
        --csv results/heldout_method2.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from annotations import load_ground_truth
from config import DetectionParams
from geometry import intersection_over_union, is_rectangular, rect_intersection
from preprocessing import candidate_mask


def covers(candidate_rect, gt_rect) -> bool:
    inter = rect_intersection(candidate_rect, gt_rect)
    return inter[2] * inter[3] >= 0.5 * gt_rect[2] * gt_rect[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline funnel analysis")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True,
                        help="Per-detection CSV produced by main.py on the same folder")
    args = parser.parse_args()

    params = DetectionParams()

    detections = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            detections[row["image"]].append(
                (int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"]))
            )

    # lost_at[i] = signs lost entering stage i (see module docstring)
    lost_at = [0, 0, 0, 0]
    detected = total = 0

    image_paths = sorted(args.data_dir.glob("*.jpg"))
    for i, image_path in enumerate(image_paths, start=1):
        json_path = image_path.with_suffix(".json")
        gts = [
            r for r in (load_ground_truth(json_path) if json_path.exists() else [])
            if params.area_min <= r[2] * r[3] <= params.area_max
        ]
        if not gts:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        mask = candidate_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        stage_rects = [[], [], []]
        for c in contours:
            rect = cv2.boundingRect(c)
            stage_rects[0].append(rect)
            if not (params.area_min <= cv2.contourArea(c) <= params.area_max):
                continue
            stage_rects[1].append(rect)
            if is_rectangular(c, params.rectangularity_threshold):
                stage_rects[2].append(rect)

        dets = detections.get(image_path.name, [])
        for gt in gts:
            total += 1
            if any(intersection_over_union(d, gt) > params.iou_threshold for d in dets):
                detected += 1
            elif any(covers(r, gt) for r in stage_rects[2]):
                lost_at[3] += 1
            elif any(covers(r, gt) for r in stage_rects[1]):
                lost_at[2] += 1
            elif any(covers(r, gt) for r in stage_rects[0]):
                lost_at[1] += 1
            else:
                lost_at[0] += 1
        if i % 200 == 0:
            print(f"  {i}/{len(image_paths)}", flush=True)

    if not total:
        print("No detectable-size ground-truth signs found")
        return 1

    print(f"\nDetectable GT signs (area in [{params.area_min:.0f}, {params.area_max:.0f}]): {total}")
    print(f"Detected (final):                {detected:4d}  ({100 * detected / total:.1f}%)")
    print(f"Lost at color mask:              {lost_at[0]:4d}  ({100 * lost_at[0] / total:.1f}%)")
    print(f"Lost at area filter:             {lost_at[1]:4d}  ({100 * lost_at[1] / total:.1f}%)")
    print(f"Lost at rectangularity filter:   {lost_at[2]:4d}  ({100 * lost_at[2] / total:.1f}%)")
    print(f"Lost at triangle/pictogram step: {lost_at[3]:4d}  ({100 * lost_at[3] / total:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
