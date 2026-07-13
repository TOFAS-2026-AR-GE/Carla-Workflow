#!/usr/bin/env python3
"""İki YOLO detection datasetini dosya adlarına a/b öneki ekleyerek birleştirir.

Örnek kullanım:
    python3 yolo_birlestir.py --a datasets/gtsdb/gtsdb_yolo \
      --b datasets/belgium/belgium_yolo --out datasets/gtsdb_belgium_tek_sinif
"""

import os
import shutil
import argparse


def argumanlari_oku():
    """İki kaynak dataset ve çıktı klasörü argümanlarını okur."""
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="1. YOLO dataset")
    p.add_argument("--b", required=True, help="2. YOLO dataset")
    p.add_argument("--out", default="datasets/birlesik_tek_sinif")
    return p.parse_args()


def split_kopyala(kaynak, hedef, prefix):
    """Bir datasetin splitlerini hedefe benzersiz önekle kopyalar."""
    for split in ["train", "val", "valid", "test"]:
        img_src = os.path.join(kaynak, split, "images")
        lbl_src = os.path.join(kaynak, split, "labels")

        if not os.path.isdir(img_src) or not os.path.isdir(lbl_src):
            continue

        split_hedef = "val" if split == "valid" else split

        img_dst = os.path.join(hedef, split_hedef, "images")
        lbl_dst = os.path.join(hedef, split_hedef, "labels")
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for f in os.listdir(img_src):
            src = os.path.join(img_src, f)
            if not os.path.isfile(src):
                continue
            yeni = f"{prefix}_{f}"
            shutil.copy2(src, os.path.join(img_dst, yeni))

        for f in os.listdir(lbl_src):
            src = os.path.join(lbl_src, f)
            if not os.path.isfile(src):
                continue
            yeni = f"{prefix}_{f}"
            shutil.copy2(src, os.path.join(lbl_dst, yeni))


def main():
    """Çıktıyı temizler, datasetleri birleştirir ve data.yaml oluşturur."""
    args = argumanlari_oku()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    split_kopyala(args.a, args.out, "a")
    split_kopyala(args.b, args.out, "b")

    data_yaml = os.path.join(args.out, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("nc: 1\n")
        f.write("names: ['traffic_sign']\n")

    print(f"[BAŞARILI] Birleşik dataset hazır: {args.out}")


if __name__ == "__main__":
    main()
