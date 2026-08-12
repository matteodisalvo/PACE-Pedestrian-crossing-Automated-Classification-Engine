"""Regenerate every figure referenced by docs/paper/report.tex.

The pipeline figures are produced from one example image of the dataset,
keeping the exact file names used in the paper; the results figures
(running precision/recall and precision-recall curves) are produced by
running both detection methods on the whole dataset.

Usage (from the repository root):
    python src/paper_figures.py                  # all figures
    python src/paper_figures.py --skip-results   # only the pipeline figures
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from annotations import load_ground_truth
from config import DEFAULT_DATA_DIR, DEFAULT_TEMPLATE, PROJECT_ROOT, DetectionParams
from detection import detect_signs, load_triangle_contour
from evaluation import EvaluationCounters, evaluate_image
from geometry import is_rectangular, is_triangle
from preprocessing import binarize_crop_method1, binarize_crop_method2
from plots import (
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_LIGHT,
    SERIES_1,
    SERIES_2,
    _precision_recall_points,
    _style_axes,
    average_precision,
    plt,
)

PAPER_DIR = PROJECT_ROOT / "docs" / "paper"
DEFAULT_EXAMPLE = "00010042_ayVAxp8dvnLy-YWsCGjFuQ.jpg"


def save_image(path: Path, img: np.ndarray, max_width: int = 1600) -> None:
    """Save an image, downscaling to `max_width` to keep the paper light."""
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _rect_kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def generate_preprocessing_figures(img: np.ndarray) -> np.ndarray:
    """Save the full-image preprocessing figures; returns the final binary mask.

    Mirrors preprocessing.candidate_mask step by step so that every
    intermediate result can be exported.
    """
    save_image(PAPER_DIR / "originale.jpg", img)

    smoothed = cv2.GaussianBlur(img, (17, 27), 0)
    smoothed = cv2.medianBlur(smoothed, 9)
    save_image(PAPER_DIR / "image" / "rumore.jpg", smoothed)

    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    save_image(PAPER_DIR / "image" / "precl.jpg", v)
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(8, 8))
    v = clahe.apply(v)
    save_image(PAPER_DIR / "image" / "postcl.jpg", v)

    save_image(PAPER_DIR / "image" / "preth.jpg", s)
    s = cv2.morphologyEx(s, cv2.MORPH_TOPHAT, _rect_kernel(75))
    save_image(PAPER_DIR / "image" / "postth.jpg", s)

    enhanced_bgr = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    save_image(PAPER_DIR / "bgrpost.jpg", enhanced_bgr)

    lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2Lab)
    mask = cv2.inRange(lab, (20, 40, 50), (220, 140, 118))
    save_image(PAPER_DIR / "image" / "binar.jpg", mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _rect_kernel(30))
    save_image(PAPER_DIR / "close.jpg", mask)
    return mask


def find_detected_candidate(
    img: np.ndarray,
    mask: np.ndarray,
    triangle_contour: np.ndarray,
    params: DetectionParams,
) -> np.ndarray:
    """Return the candidate contour whose crop yields a method-2 detection
    (falling back to the first accepted candidate)."""
    candidates, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    accepted = [
        c
        for c in candidates
        if params.area_min <= cv2.contourArea(c) <= params.area_max
        and is_rectangular(c, params.rectangularity_threshold)
    ]
    if not accepted:
        raise RuntimeError("No candidate passes the filters on the chosen image")

    for candidate in accepted:
        x, y, w, h = cv2.boundingRect(candidate)
        binary = binarize_crop_method2(img[y : y + h, x : x + w])
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (params.crop_area_min <= area <= params.crop_area_max):
                continue
            score = cv2.matchShapes(contour, triangle_contour, cv2.CONTOURS_MATCH_I1, 0)
            if score <= params.match_threshold_method2 and is_triangle(
                contour, params.approx_max_error, params.angle_tolerance
            ):
                return candidate
    return accepted[0]


def generate_method1_figures(img: np.ndarray, candidate: np.ndarray) -> None:
    """Save the method-1 figures (crop with blue outline and every processing step)."""
    outlined = img.copy()
    cv2.drawContours(outlined, [candidate], -1, (255, 0, 0), 5, cv2.LINE_AA)
    x, y, w, h = cv2.boundingRect(candidate)
    crop = outlined[y : y + h, x : x + w]
    save_image(PAPER_DIR / "b7.jpg", crop)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hc, sc, vc = cv2.split(hsv)
    save_image(PAPER_DIR / "precl1.jpg", vc)
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(14, 14))
    vc = clahe.apply(vc)
    save_image(PAPER_DIR / "postcl1.jpg", vc)

    enhanced = cv2.cvtColor(cv2.merge([hc, sc, vc]), cv2.COLOR_HSV2BGR)
    save_image(PAPER_DIR / "a7.jpg", enhanced)

    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    enhanced = cv2.medianBlur(enhanced, 3)
    save_image(PAPER_DIR / "rumori1.jpg", enhanced)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    save_image(PAPER_DIR / "preth1.jpg", gray)
    gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _rect_kernel(140))
    save_image(PAPER_DIR / "postth1.jpg", gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    save_image(PAPER_DIR / "bin1.jpg", binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _rect_kernel(5))
    save_image(PAPER_DIR / "close1.jpg", binary)


def generate_method2_figures(img: np.ndarray, candidate: np.ndarray) -> None:
    """Save the method-2 figures (every processing step on the plain crop)."""
    x, y, w, h = cv2.boundingRect(candidate)
    crop = img[y : y + h, x : x + w]
    save_image(PAPER_DIR / "crop.jpg", crop)

    smoothed = cv2.GaussianBlur(crop, (15, 15), 0)
    smoothed = cv2.medianBlur(smoothed, 5)
    save_image(PAPER_DIR / "rumori2.jpg", smoothed)

    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    save_image(PAPER_DIR / "preth2.jpg", s)
    s = cv2.morphologyEx(s, cv2.MORPH_TOPHAT, _rect_kernel(140))
    save_image(PAPER_DIR / "postth2.jpg", s)

    _, binary = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)
    save_image(PAPER_DIR / "bin2.jpg", binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _rect_kernel(15))
    save_image(PAPER_DIR / "close2.jpg", binary)
    binary = cv2.bitwise_not(binary)
    save_image(PAPER_DIR / "invert7.jpg", binary)


def run_method(method: int, data_dir: Path, template: Path, params: DetectionParams):
    """Run one method on the whole dataset; returns counters and the
    per-image history of running (precision, recall)."""
    triangle_contour = load_triangle_contour(template)
    counters = EvaluationCounters()
    history = []
    image_paths = sorted(data_dir.glob("*.jpg"))
    for i, image_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        json_path = image_path.with_suffix(".json")
        ground_truth = load_ground_truth(json_path) if json_path.exists() else []
        detections = detect_signs(img, triangle_contour, params, method=method)
        evaluate_image(image_path.name, detections, ground_truth, counters, params.iou_threshold)
        history.append((counters.precision, counters.recall))
        print(f"  [method {method}] {i}/{len(image_paths)}", end="\r")
    print()
    return counters, history


def plot_running_metrics(history, out_path: Path) -> None:
    """Line chart of the running precision and recall over the dataset."""
    x = range(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _style_axes(ax)
    ax.plot(x, [p for p, _ in history], color=SERIES_1, linewidth=2, label="Precision", zorder=2)
    ax.plot(x, [r for _, r in history], color=SERIES_2, linewidth=2, label="Recall", zorder=2)
    ax.set_xlim(1, len(history))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Metric (%)")
    legend = ax.legend(frameon=False, fontsize=10, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curve(counters: EvaluationCounters, out_path: Path) -> float:
    """Precision-recall curve with the AUPR annotated inside the axes."""
    recalls, precisions = _precision_recall_points(counters)
    aupr = average_precision(counters)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _style_axes(ax)
    ax.fill_between(recalls, precisions, step="post", color=SEQUENTIAL_LIGHT, zorder=1)
    ax.step(recalls, precisions, where="post", color=SERIES_1, linewidth=2, zorder=2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.text(
        0.97, 0.95, f"AUPR = {aupr:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=12, color=INK_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return aupr


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the figures of the paper")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--image", default=DEFAULT_EXAMPLE, help="Dataset image used for the pipeline figures"
    )
    parser.add_argument(
        "--skip-results", action="store_true", help="Skip the dataset runs (results figures)"
    )
    args = parser.parse_args()

    params = DetectionParams()
    example_path = args.data_dir / args.image
    img = cv2.imread(str(example_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read example image: {example_path}")

    print(f"Generating pipeline figures from {args.image}...")
    mask = generate_preprocessing_figures(img)
    triangle_contour = load_triangle_contour(args.template)
    candidate = find_detected_candidate(img, mask, triangle_contour, params)
    generate_method2_figures(img, candidate)
    generate_method1_figures(img, candidate)

    if not args.skip_results:
        print("Running method 1 on the whole dataset...")
        counters1, history1 = run_method(1, args.data_dir, args.template, params)
        plot_running_metrics(history1, PAPER_DIR / "PRM1.png")
        aupr1 = plot_pr_curve(counters1, PAPER_DIR / "precision-recall-metodo1.png")
        print(
            f"  method 1: precision {counters1.precision:.2f}%  "
            f"recall {counters1.recall:.2f}%  AUPR {aupr1:.3f}"
        )

        print("Running method 2 on the whole dataset...")
        counters2, history2 = run_method(2, args.data_dir, args.template, params)
        plot_running_metrics(history2, PAPER_DIR / "PRM2.png")
        aupr2 = plot_pr_curve(counters2, PAPER_DIR / "precision-recall-metodo2.png")
        print(
            f"  method 2: precision {counters2.precision:.2f}%  "
            f"recall {counters2.recall:.2f}%  AUPR {aupr2:.3f}"
        )

    print(f"Figures saved to {PAPER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
