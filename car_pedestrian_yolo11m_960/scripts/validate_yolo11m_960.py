from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = next((p for p in PACKAGE.parents if (p / "datasets").is_dir()), PACKAGE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO11m and generate metric plots.")
    parser.add_argument("--model", type=Path, default=PACKAGE / "model/yolo11m_960_best.pt")
    parser.add_argument("--data", type=Path, default=WORKSPACE / "datasets/combined_6cls/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "runs")
    parser.add_argument("--name", default="yolo11m_960_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model bulunamadı: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML bulunamadı: {args.data}")

    model = YOLO(str(args.model.resolve()), task="detect")
    model.val(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=str(args.output.resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
