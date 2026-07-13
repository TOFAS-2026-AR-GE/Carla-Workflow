#!/usr/bin/env python3
"""BelgiumTSD annotation ve JP2 görüntülerini YOLO detection formatına çevirir.

Örnek kullanım:
    python3 belgium_to_yolo.py --root datasets/belgium \
      --out datasets/belgium/belgium_yolo
"""

import os
import cv2
import argparse


def argumanlari_oku():
    """Belgium kaynak ve YOLO çıktı klasörü argümanlarını okur."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="datasets/belgium", help="Belgium ana klasörü")
    p.add_argument("--out", default="datasets/belgium/belgium_yolo", help="Çıkış YOLO klasörü")
    return p.parse_args()


def annotation_dosyasi_bul(root, dosya_adi):
    """Olası annotation dizinlerinde istenen GT dosyasını bulur."""
    adaylar = [
        os.path.join(root, "BelgiumTSD_annotations", dosya_adi),
        os.path.join(root, "BelgiumTSD_annotations", "BelgiumTSD_annotations", dosya_adi),
    ]
    for yol in adaylar:
        if os.path.exists(yol):
            return yol
    raise FileNotFoundError(f"Annotation dosyası bulunamadı: {dosya_adi}")


def satiri_parcala(satir):
    """Noktalı virgüllü Belgium annotation satırını kutu bilgilerine ayırır."""
    # örnek:
    # 01/image.000935.jp2;1346.82;246.76;1582.12;484.41;65;2;1;1;1;935;C43;
    parcalar = [x for x in satir.strip().split(";") if x != ""]
    if len(parcalar) < 12:
        return None

    rel_img = parcalar[0]
    x1 = float(parcalar[1])
    y1 = float(parcalar[2])
    x2 = float(parcalar[3])
    y2 = float(parcalar[4])

    # 6. index = superclass
    # 1 triangles -> danger
    # 2 redcircles -> prohibitory
    # 3 bluecircles -> mandatory
    # 4 redbluecircles -> prohibitory benzeri, dahil ediyoruz
    superclass = int(parcalar[6])

    return rel_img, x1, y1, x2, y2, superclass


def yolo_kutu(x1, y1, x2, y2, w, h):
    """Köşe koordinatlarını normalize YOLO merkez/genişlik formatına çevirir."""
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)

    xc = ((xmin + xmax) / 2) / w
    yc = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return xc, yc, bw, bh


def class_map(superclass):
    """Belgium superclass değerini üç sınıflı YOLO ID'sine eşler."""
    # Çıkış sınıf sırası:
    # 0 = prohibitory
    # 1 = danger
    # 2 = mandatory
    if superclass == 2:
        return 0
    if superclass == 1:
        return 1
    if superclass == 3:
        return 2
    if superclass == 4:
        return 0
    return None


def camera_yolunu_bul(root, rel_img):
    """Annotation içindeki göreli resim yolunu gerçek kamera yoluna çevirir."""
    # rel_img ör: 01/image.000935.jp2
    cam = rel_img.split("/")[0]
    return os.path.join(root, f"camera{cam}", rel_img)


def split_donustur(root, gt_txt, split_adi, out_dir):
    """Bir annotation splitini JPG görüntü ve YOLO label dosyalarına dönüştürür."""
    with open(gt_txt, "r", encoding="utf-8") as f:
        satirlar = [s.strip() for s in f if s.strip()]

    hedef_img = os.path.join(out_dir, split_adi, "images")
    hedef_lbl = os.path.join(out_dir, split_adi, "labels")
    os.makedirs(hedef_img, exist_ok=True)
    os.makedirs(hedef_lbl, exist_ok=True)

    resimden_kutulara = {}

    for satir in satirlar:
        veri = satiri_parcala(satir)
        if veri is None:
            continue

        rel_img, x1, y1, x2, y2, superclass = veri
        cid = class_map(superclass)
        if cid is None:
            continue

        kaynak_resim = camera_yolunu_bul(root, rel_img)
        if not os.path.exists(kaynak_resim):
            continue

        if kaynak_resim not in resimden_kutulara:
            resimden_kutulara[kaynak_resim] = []

        resimden_kutulara[kaynak_resim].append((cid, x1, y1, x2, y2))

    yazilan = 0

    for kaynak_resim, kutular in resimden_kutulara.items():
        img = cv2.imread(kaynak_resim)
        if img is None:
            continue

        h, w = img.shape[:2]

        temel = (
            os.path.relpath(kaynak_resim, root)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(".jp2", "")
        )

        hedef_resim = os.path.join(hedef_img, temel + ".jpg")
        hedef_label = os.path.join(hedef_lbl, temel + ".txt")

        # jp2 -> gerçek jpg dönüştürme
        cv2.imwrite(hedef_resim, img)

        satirlar_out = []
        for cid, x1, y1, x2, y2 in kutular:
            xc, yc, bw, bh = yolo_kutu(x1, y1, x2, y2, w, h)
            satirlar_out.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(hedef_label, "w", encoding="utf-8") as f:
            f.write("\n".join(satirlar_out) + ("\n" if satirlar_out else ""))

        yazilan += 1

    return yazilan


def main():
    """Train/test annotationlarını dönüştürür ve data.yaml dosyasını yazar."""
    args = argumanlari_oku()

    train_gt = annotation_dosyasi_bul(args.root, "BTSD_training_GT.txt")
    test_gt = annotation_dosyasi_bul(args.root, "BTSD_testing_GT.txt")

    train_sayi = split_donustur(args.root, train_gt, "train", args.out)
    test_sayi = split_donustur(args.root, test_gt, "test", args.out)

    data_yaml = os.path.join(args.out, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("train: train/images\n")
        f.write("test: test/images\n")
        f.write("nc: 3\n")
        f.write("names: ['prohibitory', 'danger', 'mandatory']\n")

    print("\n[BAŞARILI] Belgium YOLO dataset oluşturuldu")
    print(f"Train images: {train_sayi}")
    print(f"Test images:  {test_sayi}")
    print(f"YAML: {data_yaml}")


if __name__ == "__main__":
    main()
