from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = next((p for p in PACKAGE.parents if (p / "datasets").is_dir()), PACKAGE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect vehicles and people in an image, directory, video, webcam, or stream."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image/video path, directory, webcam index (0), or stream URL.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PACKAGE / "model/yolo11m_960_best.onnx",
        help="ONNX or PyTorch model path.",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--device", default="0", help="GPU index such as 0, or cpu.")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "runs/detector")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--show", action="store_true", help="Display results while processing.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format detections.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in txt output.")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Optional class IDs: 0 car, 1 person, 2 truck, 3 bus, 4 bicycle, 5 motorcycle.",
    )
    return parser.parse_args()


def normalize_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path), task="detect")
    results = model.predict(
        source=normalize_source(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=args.classes,
        stream=True,
        save=True,
        show=False,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(output),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    frames = 0
    detections = 0
    try:
        for result in results:
            frames += 1
            if result.boxes is not None:
                detections += len(result.boxes)
            if args.show:
                cv2.imshow("YOLO11m Detector - Q/Esc: quit", result.plot())
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Durdurma tuşuna basıldı.")
                    break
    except KeyboardInterrupt:
        print("Ctrl+C alındı, detector durduruluyor.")
    finally:
        close = getattr(results, "close", None)
        if close:
            close()
        cv2.destroyAllWindows()

    result_dir = output / args.name
    print(f"Processed frames/images: {frames}")
    print(f"Total detections: {detections}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
