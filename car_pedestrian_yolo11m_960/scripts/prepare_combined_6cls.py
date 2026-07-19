#!/usr/bin/env python3
"""BDD100K ve nuImages'i dengeli, 6 sınıflı YOLO veri setinde birleştirir."""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


NAMES = ("car", "person", "truck", "bus", "bicycle", "motorcycle")
PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = next((p for p in PACKAGE.parents if (p / "datasets").is_dir()), PACKAGE)
TARGET_COUNTS = {0: 50_000, 1: 50_000}
CAR_SOFT_LIMIT = 60_000
NU_CATEGORY_MAP = {
    "vehicle.car": 0,
    "vehicle.truck": 2,
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    "vehicle.bicycle": 4,
    "vehicle.motorcycle": 5,
}


@dataclass
class Record:
    source: Path
    labels: list[str]
    origin: str


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024):
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
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"Boş JSON: {path}")
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


def nu_category_ids(metadata_dir: Path) -> dict[str, int]:
    result = {}
    for category in iter_json_array(metadata_dir / "category.json"):
        name = category["name"]
        if name.startswith("human.pedestrian."):
            result[category["token"]] = 1
        elif name in NU_CATEGORY_MAP:
            result[category["token"]] = NU_CATEGORY_MAP[name]
    return result


def load_nu_records(nu_root: Path, split: str) -> list[Record]:
    metadata_dir = nu_root / f"v1.0-{split}"
    category_ids = nu_category_ids(metadata_dir)
    sample_data = {}
    print(f"nuImages {split}: key-frame görüntüleri indeksleniyor")
    for row in iter_json_array(metadata_dir / "sample_data.json"):
        if row.get("is_key_frame") and row["filename"].startswith("samples/"):
            sample_data[row["token"]] = (
                nu_root / row["filename"],
                int(row["width"]),
                int(row["height"]),
            )

    labels_by_token = defaultdict(list)
    print(f"nuImages {split}: kutular dönüştürülüyor")
    for ann in iter_json_array(metadata_dir / "object_ann.json"):
        class_id = category_ids.get(ann["category_token"])
        image_info = sample_data.get(ann["sample_data_token"])
        if class_id is None or image_info is None:
            continue
        _, image_width, image_height = image_info
        x1, y1, x2, y2 = map(float, ann["bbox"])
        x1, x2 = max(0.0, x1), min(float(image_width), x2)
        y1, y2 = max(0.0, y1), min(float(image_height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2) / image_width
        cy = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        labels_by_token[ann["sample_data_token"]].append(
            f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
        )

    records = []
    for token, (image_path, _, _) in sample_data.items():
        if image_path.exists():
            records.append(Record(image_path, labels_by_token.get(token, []), "nu"))
    print(f"nuImages {split}: {len(records):,} görüntü hazır")
    return records


def bdd_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_bdd_records(bdd_root: Path, split: str) -> list[Record]:
    records = []
    for image_path in sorted((bdd_root / "images" / "100k" / split).rglob("*.jpg")):
        source_label = bdd_label_path(image_path)
        labels = []
        if source_label.exists():
            for line in source_label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                old_id = int(fields[0])
                if old_id == 5:  # rider kullanılmıyor
                    continue
                new_id = 5 if old_id == 6 else old_id
                labels.append(" ".join((str(new_id), *fields[1:])))
        records.append(Record(image_path, labels, "bdd"))
    print(f"BDD100K {split}: {len(records):,} görüntü hazır")
    return records


def class_counts(records: list[Record]) -> Counter:
    counts = Counter()
    for record in records:
        counts.update(int(line.split(maxsplit=1)[0]) for line in record.labels)
    return counts


def select_train(records: list[Record], seed: int) -> list[Record]:
    rng = random.Random(seed)
    rare = [r for r in records if any(int(x.split(maxsplit=1)[0]) >= 2 for x in r.labels)]
    rng.shuffle(rare)

    def record_counts(record: Record) -> Counter:
        return Counter(int(line.split(maxsplit=1)[0]) for line in record.labels)

    def rare_score(record: Record) -> float:
        counts = record_counts(record)
        weighted_rare = counts[2] + 2 * counts[3] + 2 * counts[4] + 3 * counts[5]
        return weighted_rare / (1 + counts[0] + 0.25 * counts[1])

    rare.sort(key=rare_score, reverse=True)
    selected_ids = set()
    selected = []
    counts = Counter()
    for record in rare:
        current = record_counts(record)
        if current[0] and counts[0] + current[0] > CAR_SOFT_LIMIT:
            continue
        selected.append(record)
        selected_ids.add(id(record))
        counts.update(current)

    for target_class in (1, 0):
        candidates = [
            r for r in records
            if id(r) not in selected_ids
            and any(int(x.split(maxsplit=1)[0]) == target_class for x in r.labels)
        ]
        rng.shuffle(candidates)
        for record in candidates:
            if counts[target_class] >= TARGET_COUNTS[target_class]:
                break
            selected.append(record)
            selected_ids.add(id(record))
            counts.update(record_counts(record))

    rng.shuffle(selected)
    return selected


def select_val(bdd: list[Record], nu: list[Record], seed: int) -> list[Record]:
    rng = random.Random(seed)
    return rng.sample(bdd, min(5_000, len(bdd))) + rng.sample(nu, min(5_000, len(nu)))


def write_split(records: list[Record], output_root: Path, split: str):
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records):
        name = f"{record.origin}_{index:06d}_{record.source.name}"
        image_output = image_dir / name
        label_output = label_dir / Path(name).with_suffix(".txt")
        os.symlink(record.source.resolve(), image_output)
        label_output.write_text(
            "\n".join(record.labels) + ("\n" if record.labels else ""),
            encoding="utf-8",
        )


def print_stats(title: str, records: list[Record]):
    counts = class_counts(records)
    print(f"\n{title}: {len(records):,} görüntü, {sum(counts.values()):,} kutu")
    for class_id, name in enumerate(NAMES):
        print(f"  {name}: {counts[class_id]:,}")


def write_dataset_yaml(output_root: Path):
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in enumerate(NAMES))
    (output_root / "dataset.yaml").write_text(
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir", type=Path, default=WORKSPACE
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    bdd_root = project_dir / "datasets" / "bdd100k" / "bdd100k"
    nu_root = project_dir / "datasets" / "nuimages"
    output_root = project_dir / "datasets" / "combined_6cls"

    bdd_train = load_bdd_records(bdd_root, "train")
    bdd_val = load_bdd_records(bdd_root, "val")
    nu_train = load_nu_records(nu_root, "train")
    nu_val = load_nu_records(nu_root, "val")
    train = select_train(bdd_train + nu_train, args.seed)
    val = select_val(bdd_val, nu_val, args.seed)

    if output_root.exists():
        shutil.rmtree(output_root)
    write_split(train, output_root, "train")
    write_split(val, output_root, "val")
    write_dataset_yaml(output_root)
    print_stats("TRAIN", train)
    print_stats("VAL", val)
    print(f"\nÇıktı: {output_root}")


if __name__ == "__main__":
    main()
