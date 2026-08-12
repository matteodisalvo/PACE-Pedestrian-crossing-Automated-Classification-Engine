"""Detection of pedestrian-crossing signs in a single image.

The detector extracts the connected components of the preprocessed mask,
keeps the ones that are large enough and roughly rectangular (the sign
plate), then looks inside each cropped candidate for a triangular shape
that matches a reference triangle template.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from annotations import Rect
from config import DetectionParams
from geometry import is_rectangular, is_triangle
from preprocessing import binarize_crop_method1, binarize_crop_method2, candidate_mask


@dataclass
class Detection:
    """A detected sign: bounding box in full-image coordinates and its shape-match score."""

    rect: Rect
    match_score: float


def load_triangle_contour(template_path: Path) -> np.ndarray:
    """Extract the reference triangle contour from the template image."""
    template = cv2.imread(str(template_path))
    if template is None:
        raise FileNotFoundError(f"Cannot read triangle template: {template_path}")

    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No contour found in triangle template: {template_path}")
    # The template contains a single triangle: take the largest contour
    return max(contours, key=cv2.contourArea)


def detect_signs(
    img_bgr: np.ndarray,
    triangle_contour: np.ndarray,
    params: DetectionParams,
    method: int = 2,
) -> list[Detection]:
    """Run the full detection pipeline on one BGR image."""
    binarize_crop = binarize_crop_method1 if method == 1 else binarize_crop_method2
    match_threshold = params.match_threshold(method)

    mask = candidate_mask(img_bgr)
    candidates, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Method 1 requires the candidate contours to be outlined in blue on the
    # image before cropping: the outline becomes a dark border in the grayscale
    # crop and helps isolate the triangle (drawContours in the original C++).
    work_img = img_bgr.copy() if method == 1 else img_bgr

    detections: list[Detection] = []
    for candidate in candidates:
        if method == 1:
            cv2.drawContours(work_img, [candidate], -1, (255, 0, 0), 5, cv2.LINE_AA)
        area = cv2.contourArea(candidate)
        if not (params.area_min <= area <= params.area_max):
            continue
        if not is_rectangular(candidate, params.rectangularity_threshold):
            continue

        x, y, w, h = cv2.boundingRect(candidate)
        crop = work_img[y : y + h, x : x + w]
        crop_binary = binarize_crop(crop)

        contours, _ = cv2.findContours(crop_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if not (params.crop_area_min <= contour_area <= params.crop_area_max):
                continue

            score = cv2.matchShapes(contour, triangle_contour, cv2.CONTOURS_MATCH_I1, 0)
            if score > match_threshold:
                continue
            if not is_triangle(contour, params.approx_max_error, params.angle_tolerance):
                continue

            # Bounding box of the triangle, moved to full-image coordinates
            tx, ty, tw, th = cv2.boundingRect(contour)
            detections.append(Detection(rect=(x + tx, y + ty, tw, th), match_score=score))

    return detections
