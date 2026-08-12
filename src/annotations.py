"""Loading of ground-truth bounding boxes from the Mapillary-style JSON annotations.

The original C++ project parsed the JSON files line by line with substring
offsets; here we use proper JSON parsing, which yields the same boxes but is
robust to formatting changes.
"""

import json
from pathlib import Path

from config import TARGET_LABEL

# A rectangle is represented everywhere as (x, y, width, height) in pixels.
Rect = tuple[int, int, int, int]


def load_ground_truth(json_path: Path, label: str = TARGET_LABEL) -> list[Rect]:
    """Return the bounding boxes of all objects with the given label."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rects: list[Rect] = []
    for obj in data.get("objects", []):
        if obj.get("label") != label:
            continue
        bbox = obj["bbox"]
        x, y = int(bbox["xmin"]), int(bbox["ymin"])
        w = int(bbox["xmax"]) - x
        h = int(bbox["ymax"]) - y
        rects.append((x, y, w, h))
    return rects
