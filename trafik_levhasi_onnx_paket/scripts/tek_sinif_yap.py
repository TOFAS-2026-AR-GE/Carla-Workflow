#!/usr/bin/env python3
"""Çok sınıflı YOLO detection etiketlerini tek traffic_sign sınıfına indirger.

Örnek kullanım:
    python3 tek_sinif_yap.py --girdi datasets/gtsdb/gtsdb_yolo \
      --cikti datasets/gtsdb_tek_sinif --sinif_adi traffic_sign
"""

import os
import shutil
import argparse


def argumanlari_oku():
    """Kaynak, çıktı ve yeni tek sınıf adını okur."""
    p = argparse.ArgumentParser()
    p.add_argument("--girdi", required=True, help="Çok sınıflı YOLO dataset klasörü")
    p.add_argument("--cikti", default=None, help="Belirtilmezse <girdi>_tek_sinif kullanılır")
    p.add_argument("--sinif_adi", default="traffic_sign", help="Yeni tek sınıf adı")
    return p.parse_args()


def labels_donustur(kaynak_klasor, hedef_klasor):
    """Bir labels klasöründeki bütün sınıf ID'lerini 0 yapar."""
    os.makedirs(hedef_klasor, exist_ok=True)
    toplam_satir = 0

    for dosya_adi in os.listdir(kaynak_klasor):
        if not dosya_adi.endswith(".txt"):
            continue

        kaynak_yol = os.path.join(kaynak_klasor, dosya_adi)
        hedef_yol = os.path.join(hedef_klasor, dosya_adi)

        with open(kaynak_yol, "r", encoding="utf-8") as f:
            satirlar = [s.strip() for s in f.readlines() if s.strip()]

        yeni_satirlar = []
        for satir in satirlar:
            parcalar = satir.split()
            if len(parcalar) < 5:
                continue
            parcalar[0] = "0"
            yeni_satirlar.append(" ".join(parcalar))
            toplam_satir += 1

        with open(hedef_yol, "w", encoding="utf-8") as f:
            f.write("\n".join(yeni_satirlar) + ("\n" if yeni_satirlar else ""))

    return toplam_satir


def main():
    """Splitleri dönüştürür, görüntüleri kopyalar ve data.yaml yazar."""
    args = argumanlari_oku()
    cikti_klasoru = args.cikti or f"{args.girdi}_tek_sinif"

    toplam_genel = 0

    for split in ["train", "val", "test"]:
        kaynak_labels = os.path.join(args.girdi, split, "labels")
        kaynak_images = os.path.join(args.girdi, split, "images")

        if not os.path.isdir(kaynak_labels):
            print(f"[ATLANDI] {split}/labels bulunamadı.")
            continue

        hedef_labels = os.path.join(cikti_klasoru, split, "labels")
        hedef_images = os.path.join(cikti_klasoru, split, "images")

        satir_sayisi = labels_donustur(kaynak_labels, hedef_labels)
        toplam_genel += satir_sayisi

        if os.path.isdir(kaynak_images):
            shutil.copytree(kaynak_images, hedef_images, dirs_exist_ok=True)

        print(f"  - {split}: {satir_sayisi} kutu tek sınıfa çevrildi")

    data_yaml_yolu = os.path.join(cikti_klasoru, "data.yaml")
    with open(data_yaml_yolu, "w", encoding="utf-8") as f:
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{args.sinif_adi}']\n")

    print(f"\n[BAŞARILI] Tek sınıflı veri seti: {cikti_klasoru}/")
    print(f"  - Toplam kutu: {toplam_genel}")
    print(f"  - Yeni data.yaml: {data_yaml_yolu}")


if __name__ == "__main__":
    main()
