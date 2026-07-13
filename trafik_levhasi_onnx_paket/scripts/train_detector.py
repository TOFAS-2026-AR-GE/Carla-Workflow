#!/usr/bin/env python3
"""Ultralytics YOLO detection modelini bir data.yaml ile eğitir.

Örnek kullanım:
    python3 train_detector.py --data datasets/gtsdb_belgium_tek_sinif/data.yaml \
      --model yolo11s.pt --epochs 50 --imgsz 512 --batch 16 \
      --proje_adi gtsdb_belgium_tek_sinif_s --run_adi egitim
"""

import argparse
from ultralytics import YOLO


def argumanlari_oku():
    """Detection eğitimi için komut satırı seçeneklerini okur."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="YOLO data.yaml yolu")
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Başlangıç ağırlığı: yolov8n.pt / yolov8s.pt / yolov8m.pt",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=512, help="Giriş boyutu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument(
        "--cache",
        default="disk",
        choices=["ram", "disk", "false"],
        help="Resimleri cache'le",
    )
    p.add_argument("--proje_adi", default="gtsdb_belgium_tek_sinif")
    p.add_argument("--run_adi", default="egitim")
    return p.parse_args()


def main():
    """YOLO modelini oluşturur, eğitimi başlatır ve ağırlık yollarını yazdırır."""
    args = argumanlari_oku()

    cache_degeri = args.cache if args.cache != "false" else False

    model = YOLO(args.model)

    sonuc = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/detect",
        name=f"{args.proje_adi}/{args.run_adi}",
        patience=args.patience,
        cache=cache_degeri,
        workers=args.workers,
    )

    print("\n[EĞİTİM TAMAMLANDI]")
    print(f"En iyi model ağırlıkları: {sonuc.save_dir}/weights/best.pt")
    print(f"Son model ağırlıkları: {sonuc.save_dir}/weights/last.pt")


if __name__ == "__main__":
    main()
