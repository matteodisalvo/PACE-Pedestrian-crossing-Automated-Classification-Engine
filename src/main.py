"""Entry point: detect pedestrian-crossing signs on the whole dataset and
report precision/recall against the JSON ground-truth annotations.

Usage (from the repository root):
    python src/main.py                 # method 2 (default) on the full dataset
    python src/main.py --method 1      # use method 1
    python src/main.py --limit 10      # quick run on the first 10 images
    python src/main.py --save-vis      # save annotated images to results/annotated/
"""

import argparse
import sys
from pathlib import Path

import cv2

from annotations import load_ground_truth
from config import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, DEFAULT_TEMPLATE, DetectionParams
from detection import detect_signs, load_triangle_contour
from evaluation import EvaluationCounters, evaluate_image, save_results_csv
from plots import save_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pedestrian-crossing sign detector")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing the .jpg images and their .json annotations",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Triangle template image used by cv2.matchShapes",
    )
    parser.add_argument(
        "--method",
        type=int,
        choices=(1, 2),
        default=2,
        help="Crop binarization method (1: grayscale top-hat, 2: saturation top-hat)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (useful for quick tests)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "results.csv",
        help="Output CSV with one row per detection",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "plots",
        help="Folder where evaluation plots (confusion matrix, PR curve, ...) are saved",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving the evaluation plots",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save annotated images (detections in black, ground truth in red)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each annotated image in a window (press any key to continue)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_paths = sorted(args.data_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No .jpg images found in {args.data_dir}", file=sys.stderr)
        return 1
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    triangle_contour = load_triangle_contour(args.template)
    params = DetectionParams()
    counters = EvaluationCounters()

    vis_dir = DEFAULT_RESULTS_DIR / "annotated"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(image_paths)} images (method {args.method})...")

    for i, image_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  Skipping unreadable image: {image_path.name}", file=sys.stderr)
            continue

        json_path = image_path.with_suffix(".json")
        ground_truth = load_ground_truth(json_path) if json_path.exists() else []

        detections = detect_signs(img, triangle_contour, params, method=args.method)
        evaluate_image(image_path.name, detections, ground_truth, counters, params.iou_threshold)

        print(
            f"[{i}/{len(image_paths)}] {image_path.name}: "
            f"{len(detections)} detection(s), {len(ground_truth)} ground-truth sign(s) | "
            f"precision {counters.precision:.2f}%  recall {counters.recall:.2f}%  "
            f"(TP {counters.tp}, FP {counters.fp}, FN {counters.fn})"
        )

        if args.save_vis or args.show:
            annotated = img.copy()
            for det in detections:
                x, y, w, h = det.rect
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 0), 10, cv2.LINE_AA)
            for x, y, w, h in ground_truth:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3, cv2.LINE_AA)
            if args.save_vis:
                cv2.imwrite(str(vis_dir / image_path.name), annotated)
            if args.show:
                cv2.imshow("Detections", cv2.resize(annotated, None, fx=0.25, fy=0.25))
                cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()

    save_results_csv(counters, args.csv)

    print("\n===== Final results =====")
    print(f"Ground-truth signs: {counters.total_signs}")
    print(f"True positives:  {counters.tp}")
    print(f"False positives: {counters.fp}")
    print(f"False negatives: {counters.fn}")
    print(f"Precision: {counters.precision:.2f}%")
    print(f"Recall:    {counters.recall:.2f}%")
    print(f"Per-detection results saved to {args.csv}")

    if not args.no_plots:
        aupr = save_all_plots(counters, args.plots_dir)
        print(f"AUPR: {aupr:.3f}")
        print(f"Plots saved to {args.plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
