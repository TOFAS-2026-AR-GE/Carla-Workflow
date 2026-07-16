import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Eski projeden model dosyalarini kopyalar."
    )
    parser.add_argument("source", type=Path, help="Eski Carla-Workflow klasoru")
    args = parser.parse_args()

    mappings = {
        args.source / "models/carla_yolov8n_best.pt": PROJECT_ROOT
        / "models/vehicle/carla_yolov8n_best.pt",
        args.source / "trafik_levhasi_onnx_paket/models/detector.onnx": PROJECT_ROOT
        / "models/signs/detector.onnx",
        args.source / "trafik_levhasi_onnx_paket/models/classifier.onnx": PROJECT_ROOT
        / "models/signs/classifier.onnx",
    }

    missing = [source for source in mappings if not source.is_file()]
    if missing:
        print("Eksik kaynak dosyalari:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)

    for source, target in mappings.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"[OK] {target.name}")


if __name__ == "__main__":
    main()
