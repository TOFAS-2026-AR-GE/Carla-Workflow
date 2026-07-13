#!/usr/bin/env python3
"""Ultralytics YOLO classification modelini klasör tabanlı dataset ile eğitir.

Örnek kullanım:
    python3 train_classifier.py \
      --data datasets/gtsrb_belgium_78cls \
      --model yolo11n-cls.pt --epochs 30 --imgsz 96 --batch 32 \
      --run_adi gtsrb_belgium_78cls_no_flip_yolo11n

Trafik levhalarında sol/sağ anlamını bozmamak için yatay ve dikey çevirme
augmentation'ları bilinçli olarak kapalı tutulur.
"""

import argparse

from ultralytics import YOLO


def argumanlari_oku():
    """Classifier eğitimi için komut satırı seçeneklerini okur."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default="datasets/gtsrb_belgium_78cls",
        help="İçinde train/ ve val/ bulunan classification dataset kökü",
    )
    p.add_argument("--model", default="yolo11n-cls.pt", help="Başlangıç classification modeli")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=96, help="Classifier giriş boyutu")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    p.add_argument("--device", default=None, help="Örnek: 0, cpu; boşsa Ultralytics seçer")
    p.add_argument(
        "--cache",
        default="false",
        choices=["ram", "disk", "false"],
        help="Görüntü cache yöntemi",
    )
    p.add_argument("--project", default="runs/classify", help="Eğitim çıktılarının ana klasörü")
    p.add_argument(
        "--run_adi",
        default="gtsrb_belgium_78cls_no_flip_yolo11n",
        help="Bu eğitim koşusunun klasör adı",
    )
    return p.parse_args()


def main():
    """Classifier modelini yön-koruyan augmentation ayarlarıyla eğitir."""
    args = argumanlari_oku()
    cache_degeri = args.cache if args.cache != "false" else False

    model = YOLO(args.model)
    sonuc = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        cache=cache_degeri,
        project=args.project,
        name=args.run_adi,
        fliplr=0.0,
        flipud=0.0,
    )

    print("\n[CLASSIFIER EĞİTİMİ TAMAMLANDI]")
    print(f"En iyi model ağırlıkları: {sonuc.save_dir}/weights/best.pt")
    print(f"Son model ağırlıkları: {sonuc.save_dir}/weights/last.pt")


if __name__ == "__main__":
    main()
