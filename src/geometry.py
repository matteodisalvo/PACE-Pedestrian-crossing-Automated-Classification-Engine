"""Geometric predicates and rectangle utilities used by the detector."""

import math

import cv2
import numpy as np

from annotations import Rect


def rect_intersection(r1: Rect, r2: Rect) -> Rect:
    """Intersection of two (x, y, w, h) rectangles; (0, 0, 0, 0) if disjoint."""
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[0] + r1[2], r2[0] + r2[2])
    y2 = min(r1[1] + r1[3], r2[1] + r2[3])
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2 - x1, y2 - y1)
    return (0, 0, 0, 0)


def intersection_over_union(r1: Rect, r2: Rect) -> float:
    """IoU of two (x, y, w, h) rectangles, in [0, 1]."""
    inter = rect_intersection(r1, r2)
    inter_area = inter[2] * inter[3]
    if inter_area == 0:
        return 0.0
    union_area = r1[2] * r1[3] + r2[2] * r2[3] - inter_area
    return inter_area / union_area


def corrected_min_area_rect(contour: np.ndarray):
    """cv2.minAreaRect with width/height swapped so that height >= width."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    if h < w:
        w, h = h, w
        angle += 90.0
    return (cx, cy), (w, h), angle


def is_rectangular(contour: np.ndarray, threshold: float) -> bool:
    """True if the contour fills at least `threshold` of its min-area rectangle."""
    _, (w, h), _ = corrected_min_area_rect(contour)
    rect_area = w * h
    if rect_area == 0:
        return False
    return cv2.contourArea(contour) / rect_area > threshold


def is_triangle(contour: np.ndarray, max_error: float, angle_tolerance: float) -> bool:
    """True if the contour approximates an (almost) equilateral triangle.

    The contour is simplified with approxPolyDP; it must reduce to exactly
    3 vertices, with similar side lengths and all internal angles within
    `angle_tolerance` degrees of 60.
    """
    approx = cv2.approxPolyDP(contour, max_error, True)
    if len(approx) != 3:
        return False

    pts = approx.reshape(3, 2).astype(np.float64)
    side1 = float(np.linalg.norm(pts[1] - pts[0]))
    side2 = float(np.linalg.norm(pts[2] - pts[1]))
    side3 = float(np.linalg.norm(pts[0] - pts[2]))
    if min(side1, side2, side3) == 0:
        return False

    tolerance = max_error * 2
    if not (
        abs(side1 - side2) < tolerance
        and abs(side2 - side3) < tolerance
        and abs(side3 - side1) < tolerance
    ):
        return False

    def angle_deg(a: float, b: float, opposite: float) -> float:
        # Law of cosines; clamp to acos domain against float rounding
        cos_val = (a * a + b * b - opposite * opposite) / (2 * a * b)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))

    angles = (
        angle_deg(side1, side3, side2),
        angle_deg(side2, side1, side3),
        angle_deg(side3, side2, side1),
    )
    return all(abs(a - 60.0) < angle_tolerance for a in angles)
