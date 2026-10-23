"""Copy one labeled dataset image per pose into the tutor's reference folder.

Use with an extracted image-classification dataset whose images are organized
under folders named after their pose labels (for example ``train/Padmasana``).
"""

import argparse
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def output_slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Import one sample image for each pose label in pose_data.csv."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Extracted classification dataset directory (class-named folders).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("pose_data.csv"),
        help="Pose dataset CSV (default: pose_data.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pose_references"),
        help="Tutor image directory (default: pose_references).",
    )
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset_dir}")
    if not args.csv.is_file():
        parser.error(f"Pose CSV does not exist: {args.csv}")

    import csv

    with args.csv.open(newline="", encoding="utf-8-sig") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            parser.error(f"{args.csv} must contain a 'label' column.")
        labels = sorted(
            {row["label"].strip() for row in reader if row.get("label", "").strip()},
            key=str.casefold,
        )

    class_dirs = {}
    for folder in args.dataset_dir.rglob("*"):
        if folder.is_dir():
            class_dirs.setdefault(normalize_label(folder.name), []).append(folder)

    args.output.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for label in labels:
        matches = class_dirs.get(normalize_label(label), [])
        images = sorted(
            (image for folder in matches for image in folder.rglob("*")
             if image.is_file() and image.suffix.casefold() in IMAGE_EXTENSIONS),
            key=lambda image: str(image).casefold(),
        )
        if not images:
            missing.append(label)
            continue

        source = images[0]
        destination = args.output / f"{output_slug(label)}{source.suffix.casefold()}"
        shutil.copy2(source, destination)
        copied.append((label, source, destination))

    print(f"Imported {len(copied)} of {len(labels)} pose images into {args.output}.")
    for label, source, destination in copied:
        print(f"{label}: {source} -> {destination.name}")
    if missing:
        print("No matching class images found for:")
        for label in missing:
            print(f"  - {label}")
        print("Check that the extracted dataset has a folder for each pose label.")


if __name__ == "__main__":
    main()
