from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = next((p for p in PACKAGE.parents if (p / "datasets").is_dir()), PACKAGE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11m on the combined six-class dataset.")
    parser.add_argument("--data", type=Path, default=WORKSPACE / "datasets/combined_6cls/dataset.yaml")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "runs")
    parser.add_argument("--name", default="bdd_nuimages_6cls_yolo11m_960")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML bulunamadı: {args.data}\n"
            "Önce prepare_combined_6cls.py çalıştırın veya --data ile yolu verin."
        )

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        amp=True,
        patience=args.patience,
        cos_lr=True,
        close_mosaic=10,
        optimizer="auto",
        seed=42,
        deterministic=False,
        project=str(args.output.resolve()),
        name=args.name,
        save=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
