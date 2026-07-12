from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pretrained CARLA YOLOv8n model")
    parser.add_argument("source", help="Image, video, directory, URL, or webcam index")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    weights = Path(__file__).with_name("carla_yolov8n_best.pt")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install/use a CUDA-enabled PyTorch build")

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(str(weights))
    model.predict(
        source=source,
        classes=[9],
        conf=args.conf,
        imgsz=960,
        device=0,
        save=True,
        show=args.show,
        project=str(Path(__file__).parent / "predictions"),
        name="carla_yolov8n",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

