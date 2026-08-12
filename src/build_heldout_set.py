"""Build the held-out generalization set from the official MTSD v2 archives.

The set contains every MTSD image that (a) has at least one object with the
target label, (b) is not one of the 155 images of the course subset used to
develop and tune the pipeline, (c) is not a 360 panorama, and (d) has a width
in the same range as the development subset (1280-5344 px), so that the
pixel-based pipeline parameters remain comparable. The resulting 1188 image
keys are frozen in data/heldout-keys.txt for reproducibility.

Usage (from the repository root):
    python src/build_heldout_set.py --mtsd-dir /path/to/MTSD-zips \
        --out-dir data/raw/heldout-crossing

The MTSD directory must contain the official zip archives: the annotation
archive (mtsd_v2_fully_annotated) and the train/val image archives. Images
and their JSON annotations are written side by side, in the layout expected
by main.py --data-dir.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from config import PROJECT_ROOT

DEFAULT_KEYS_FILE = PROJECT_ROOT / "data" / "heldout-keys.txt"
ANNOTATION_PREFIX = "mtsd_v2_fully_annotated/annotations/"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the held-out evaluation set")
    parser.add_argument(
        "--mtsd-dir", type=Path, required=True,
        help="Folder containing the official MTSD v2 zip archives",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "heldout-crossing",
        help="Destination folder for the extracted images and annotations",
    )
    parser.add_argument(
        "--keys-file", type=Path, default=DEFAULT_KEYS_FILE,
        help="Text file with one MTSD image key per line",
    )
    args = parser.parse_args()

    wanted = set(args.keys_file.read_text().split())
    if not wanted:
        raise SystemExit(f"No keys found in {args.keys_file}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(args.mtsd_dir.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No zip archives found in {args.mtsd_dir}")

    images = annotations = 0
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                path = Path(name)
                if name.startswith("images/") and path.stem in wanted:
                    target = args.out_dir / f"{path.stem}.jpg"
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    images += 1
                elif name.startswith(ANNOTATION_PREFIX) and path.stem in wanted:
                    target = args.out_dir / f"{path.stem}.json"
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    annotations += 1
        print(f"{zip_path.name[:16]}...: {images} images, {annotations} annotations so far")

    missing = len(wanted) - images
    print(f"\nExtracted {images} images and {annotations} annotations to {args.out_dir}")
    if missing:
        print(f"WARNING: {missing} keys not found in the archives")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
