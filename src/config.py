"""Default paths and detection parameters for the pedestrian-crossing sign detector."""

from dataclasses import dataclass
from pathlib import Path

# Project root = repository root (this file lives in <root>/src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "subset-IPA-AIA-crossing"
DEFAULT_TEMPLATE = PROJECT_ROOT / "data" / "templates" / "triangle_template.png"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# Label of the sign to detect in the ground-truth annotations
TARGET_LABEL = "information--pedestrians-crossing--g1"


@dataclass
class DetectionParams:
    """Tunable parameters of the detection pipeline (values from the original C++ project)."""

    # Area limits for candidate connected components on the full image
    area_min: float = 2_000.0
    area_max: float = 100_000.0

    # Rectangularity index threshold (contour area / min-area-rect area)
    rectangularity_threshold: float = 0.4

    # Area limits for triangle candidates inside a crop
    crop_area_min: float = 500.0
    crop_area_max: float = 30_000.0

    # cv2.matchShapes similarity thresholds (method 1 / method 2)
    match_threshold_method1: float = 0.13
    match_threshold_method2: float = 0.2

    # Max approximation error (px) for cv2.approxPolyDP in the triangle test
    approx_max_error: float = 40.0

    # Angle tolerance (degrees) around 60 deg for the triangle test
    angle_tolerance: float = 28.0

    # Minimum Intersection-over-Union to count a detection as a true positive
    iou_threshold: float = 0.2

    def match_threshold(self, method: int) -> float:
        return self.match_threshold_method1 if method == 1 else self.match_threshold_method2
