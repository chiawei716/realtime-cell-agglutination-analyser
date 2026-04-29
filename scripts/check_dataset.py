"""
Check dataset for image classification.
Responsibilities:
- scan image files
- verify readability with OpenCV
- summarize image size distribution
- summarize label distribution if labels exist
- save random sample images
- report failed files
"""

from pathlib import Path
import argparse
import random
import shutil
import pandas as pd
from collections import Counter
import numpy as np

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABELS = {"normal", "agglutination"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Check dataset for image classification.")
    parser.add_argument("--image-dir", type=str, required=True, help="Path to the raw image directory")
    parser.add_argument("--sample-dir", type=str, required=True, help="Path to the sample image directory")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of sample images to save")
    parser.add_argument("--processed-dir", type=str, required=False, default=None, help="Path to the processed image directory")
    return parser.parse_args()

def find_image_files(image_dir: Path) -> list[Path]:
    """Return all image files under image_dir with valid extensions."""
    return [path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def read_image_info(image_path: Path):
    """
    Read image with OpenCV.

    Return:
        (width, height) if readable
        None if failed
    """
    try:
        # Avoid chinese path problem for Windows
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)

        if image is None:
            return None

        height, width = image.shape[:2]
        return (width, height)
    except Exception as e:
        return None


def infer_label(image_path: Path, image_dir: Path):
    """
    Infer label from folder structure, if applicable.

    Example:
        data/raw/normal/xxx.jpg -> normal
        data/raw/agglutination/yyy.jpg -> agglutination
    """
    return image_path.parent.name if image_path.parent.name in LABELS else "other"
    


def save_sample_images(image_paths: list[Path], output_dir: Path, count: int):
    """Randomly copy sample images to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in random.sample(image_paths, count):
        shutil.copy(path, output_dir / path.name)
    return


def print_report(image_paths: list[Path], readable_files: list[Path], failed_files: list[Path], size_counter: Counter, label_counter: Counter, output_dir: Path, processed_dir: Path):
    """Print dataset summary."""
    print(f"Total images: {len(image_paths)}")
    print(f"Readable images: {len(readable_files)}")
    print(f"Failed images: {len(failed_files)}")
    print()

    print(f"Image size distribution:")
    for (size, count) in size_counter.most_common():
        print(f"{size}: {count}")
    print()

    print("Label distribution:")
    for (label, count) in label_counter.most_common():
        print(f"{label}: {count}")
    print()

    if (processed_dir is not None):
        print(f"Processed images saved to: {processed_dir}")

    print(f"Sample images saved to: {output_dir}")


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    sample_dir = Path(args.sample_dir)
    processed_dir = Path(args.processed_dir)
    num_samples = args.num_samples

    if (processed_dir is not None):
        processed_dir.mkdir(parents=True, exist_ok=True)
        for label in LABELS:
            (processed_dir / label).mkdir(parents=True, exist_ok=True)

    df = {"raw_image_path": [], "processed_image_path": [], "width": [], "height": [], "label": []}

    image_paths = find_image_files(image_dir)

    size_counter = Counter()
    label_counter = Counter()
    failed_files = []
    readable_files = []

    for image_path in image_paths:
        image_info = read_image_info(image_path)

        if image_info is None:
            failed_files.append(image_path)
            continue

        width, height = image_info
        size_counter[f"{width}x{height}"] += 1
        label = infer_label(image_path, image_dir)
        if label is not None:
            label_counter[label] += 1
        
        new_filename = f"{label}_{label_counter[label]:06d}{image_path.suffix.lower()}"

        readable_files.append(processed_dir / label / new_filename if processed_dir else image_path)

        if (processed_dir is not None):
            shutil.copy(image_path, processed_dir / label / new_filename)

        # Record information of path mapping, image size, and label
        df["raw_image_path"].append(image_path)
        df["processed_image_path"].append(processed_dir / label / new_filename if processed_dir else image_path)
        df["width"].append(width)
        df["height"].append(height)
        df["label"].append(label)   

    df = pd.DataFrame(df)
    df.to_csv("data/dataset_info.csv", index=False)

    save_sample_images(readable_files, sample_dir, num_samples)

    print_report(image_paths, readable_files, failed_files, size_counter, label_counter, sample_dir, processed_dir)


if __name__ == "__main__":
    main()