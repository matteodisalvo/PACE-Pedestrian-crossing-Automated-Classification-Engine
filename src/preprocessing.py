"""Image preprocessing: from a BGR frame to the binary mask of sign candidates,
and the two alternative binarization methods applied to each cropped candidate.
"""

import cv2
import numpy as np


def _rect_kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def candidate_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the regions that may contain a traffic sign.

    Pipeline: Gaussian + median smoothing, CLAHE on the HSV value channel,
    top-hat on the saturation channel, then thresholding in the Lab color
    space followed by a morphological closing.
    """
    smoothed = cv2.GaussianBlur(img_bgr, (17, 27), 0)
    smoothed = cv2.medianBlur(smoothed, 9)

    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(8, 8))
    v = clahe.apply(v)
    s = cv2.morphologyEx(s, cv2.MORPH_TOPHAT, _rect_kernel(75))

    enhanced_bgr = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2Lab)

    mask = cv2.inRange(lab, (20, 40, 50), (220, 140, 118))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _rect_kernel(30))


def binarize_crop_method1(crop_bgr: np.ndarray) -> np.ndarray:
    """Method 1: CLAHE on V, grayscale top-hat, Otsu threshold."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(14, 14))
    v = clahe.apply(v)

    enhanced = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    enhanced = cv2.medianBlur(enhanced, 3)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _rect_kernel(140))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _rect_kernel(5))


def binarize_crop_method2(crop_bgr: np.ndarray) -> np.ndarray:
    """Method 2: top-hat on the HSV saturation channel, Otsu threshold, inversion."""
    smoothed = cv2.GaussianBlur(crop_bgr, (15, 15), 0)
    smoothed = cv2.medianBlur(smoothed, 5)

    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)

    s = cv2.morphologyEx(s, cv2.MORPH_TOPHAT, _rect_kernel(140))
    _, binary = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _rect_kernel(15))
    return cv2.bitwise_not(binary)
