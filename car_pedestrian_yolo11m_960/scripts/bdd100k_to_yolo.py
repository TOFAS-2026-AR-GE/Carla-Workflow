#!/usr/bin/env python3
"""BDD100K detection JSON etiketlerini 7 sınıflı YOLO biçimine dönüştürür."""

import argparse
import json
from collections import Counter
from pathlib import Path


CLASSES = ("car", "person", "truck", "bus", "bike", "rider", "motor")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = next((p for p in PACKAGE.parents if (p / "datasets").is_dir()), PACKAGE)


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024):
    """Büyük JSON dizisini belleğe tamamen almadan nesne nesne okur."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False

    with path.open("r", encoding="utf-8") as stream:
        while True:
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0
            if not eof and len(buffer) - position < chunk_size:
                data = stream.read(chunk_size)
                if data:
                    buffer += data
                else:
                    eof = True

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"Boş JSON dosyası: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"JSON kökü dizi değil: {path}")
                position += 1
                started = True
                continue

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                item, position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                continue
            yield item


def image_index(images_dir: Path) -> dict[str, Path]:
    index = {}
    duplicates = set()
    for image_path in images_dir.rglob("*.jpg"):
        if image_path.name in index:
            duplicates.add(image_path.name)
        index[image_path.name] = image_path
    if duplicates:
        raise ValueError(f"Aynı isimli görüntüler bulundu: {sorted(duplicates)[:5]}")
    return index


def yolo_line(label: dict):
    category = label.get("category")
    box = label.get("box2d")
    if category not in CLASS_TO_ID or not box:
        return None

    x1 = max(0.0, min(float(box["x1"]), IMAGE_WIDTH))
    y1 = max(0.0, min(float(box["y1"]), IMAGE_HEIGHT))
    x2 = max(0.0, min(float(box["x2"]), IMAGE_WIDTH))
    y2 = max(0.0, min(float(box["y2"]), IMAGE_HEIGHT))
    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2) / IMAGE_WIDTH
    cy = ((y1 + y2) / 2) / IMAGE_HEIGHT
    width = (x2 - x1) / IMAGE_WIDTH
    height = (y2 - y1) / IMAGE_HEIGHT
    return category, (
        f"{CLASS_TO_ID[category]} {cx:.6f} {cy:.6f} "
        f"{width:.6f} {height:.6f}"
    )


def convert_split(annotation_path: Path, images_dir: Path, labels_dir: Path):
    images = image_index(images_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    processed = 0
    missing = 0

    for frame in iter_json_array(annotation_path):
        image_path = images.get(frame["name"])
        if image_path is None:
            missing += 1
            continue
        relative = image_path.relative_to(images_dir).with_suffix(".txt")
        output_path = labels_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for label in frame.get("labels", []):
            converted = yolo_line(label)
            if converted:
                category, line = converted
                counts[category] += 1
                lines.append(line)
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        processed += 1
        if processed % 10_000 == 0:
            print(f"  {processed:,}/{len(images):,} görüntü işlendi")

    print(f"Tamamlandı: {processed:,} görüntü, {sum(counts.values()):,} kutu")
    if missing:
        print(f"Uyarı: {missing:,} annotation görüntüsü bulunamadı")
    for name in CLASSES:
        print(f"  {name}: {counts[name]:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir", type=Path, default=WORKSPACE
    )
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    dataset_dir = project_dir / "datasets" / "bdd100k" / "bdd100k"
    annotation_dir = project_dir / "datasets" / "bdd100k_labels_release" / "bdd100k" / "labels"

    for split in args.splits:
        print(f"\n{split.upper()} dönüştürülüyor")
        convert_split(
            annotation_dir / f"bdd100k_labels_images_{split}.json",
            dataset_dir / "images" / "100k" / split,
            dataset_dir / "labels" / "100k" / split,
        )


if __name__ == "__main__":
    main()
