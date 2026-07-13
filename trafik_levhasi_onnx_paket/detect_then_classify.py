#!/usr/bin/env python3
"""Detector ile levhaları bulur, crop'ları classifier ile sınıflandırır.

ONNX ile örnek kullanım:
    CUDA_VISIBLE_DEVICES=-1 python3 detect_then_classify.py \
      --source datasets/gtsdb_belgium_tek_sinif_resplit/test/images \
      --detector models/detector.onnx \
      --classifier models/classifier.onnx \
      --out outputs \
      --det-conf 0.25 --cls-conf 0.50 --det-imgsz 512 --cls-imgsz 96 \
      --pad 0.25 --square --save-crops --show-det-conf
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def argumanlari_oku():
    """Komut satırı seçeneklerini tanımlar ve kullanıcı girdilerini okur."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--detector",
        default="models/detector.onnx",
        help="YOLO detection modeli (.onnx veya .pt)",
    )
    p.add_argument(
        "--classifier",
        default="models/classifier.onnx",
        help="YOLO classification modeli (.onnx veya .pt)",
    )
    p.add_argument("--source", required=True, help="Resim, klasör veya video yolu")
    p.add_argument("--out", default="outputs", help="Çıktı klasörü")
    p.add_argument("--det-imgsz", type=int, default=512, help="Detector image size")
    p.add_argument("--cls-imgsz", type=int, default=96, help="Classifier image size")
    p.add_argument("--det-conf", type=float, default=0.25, help="Detector confidence eşiği")
    p.add_argument("--det-iou", type=float, default=0.5, help="Detector NMS IoU eşiği")
    p.add_argument("--cls-conf", type=float, default=0.50, help="Bunun altındaki classifier tahminini unknown yap")
    p.add_argument("--pad", type=float, default=0.25, help="Kutunun etrafına eklenecek oran")
    p.add_argument("--square", action=argparse.BooleanOptionalAction, default=True, help="Crop'u kare yap")
    p.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True, help="Crop'ları kaydet")
    p.add_argument("--show-det-conf", action=argparse.BooleanOptionalAction, default=True, help="Detector confidence göster")
    return p.parse_args()


def kaynaklari_topla(source):
    """Kaynağı resim listesi veya tek video olarak çözümler."""
    path = Path(source)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS), "images"
    if path.suffix.lower() in IMAGE_EXTS:
        return [path], "images"
    return [path], "video"


def kirpme_koordinati(x1, y1, x2, y2, genislik, yukseklik, pad=0.25, square=False):
    """Detection kutusunu büyütür, gerekirse kare yapar ve görüntüye sınırlar."""
    bw = x2 - x1
    bh = y2 - y1

    if square:
        kenar = max(bw, bh)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        x1 = cx - kenar / 2
        x2 = cx + kenar / 2
        y1 = cy - kenar / 2
        y2 = cy + kenar / 2
        bw = bh = kenar

    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(genislik, int(round(x2)))
    y2 = min(yukseklik, int(round(y2)))
    return x1, y1, x2, y2


def dosya_adi_temizle(text):
    """Sınıf adını güvenli bir dosya adı parçasına dönüştürür."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def siniflandir(classifier, crop, imgsz, cls_conf):
    """Bir levha crop'ını sınıflandırır; düşük güveni unknown olarak döndürür."""
    sonuc = classifier.predict(crop, imgsz=imgsz, verbose=False)[0]
    probs = sonuc.probs
    cls_id = int(probs.top1)
    conf = float(probs.top1conf)
    isim = classifier.names[cls_id] if isinstance(classifier.names, dict) else classifier.names[cls_id]

    if conf < cls_conf:
        return "unknown", conf, cls_id
    return isim, conf, cls_id


def kutu_ciz(frame, box, label, det_conf, cls_conf, show_det_conf=False):
    """Detection kutusunu ve sınıflandırma metnini görüntü üzerine çizer."""
    x1, y1, x2, y2 = [int(v) for v in box]
    renk = (0, 200, 80) if label != "unknown" else (0, 180, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)

    metin = f"{label} {cls_conf:.2f}"
    if show_det_conf:
        metin += f" det:{det_conf:.2f}"

    (tw, th), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(th + 8, y1)
    cv2.rectangle(frame, (x1, y_text - th - 8), (x1 + tw + 8, y_text + 4), renk, -1)
    cv2.putText(frame, metin, (x1 + 4, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def resim_isle(detector, classifier, image_path, out_dir, args):
    """Tek görüntüde detection, crop, classification ve çıktı kaydını yapar."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ATLANDI] Resim okunamadı: {image_path}")
        return

    h, w = frame.shape[:2]

    det = detector.predict(
        frame,
        imgsz=args.det_imgsz,
        conf=args.det_conf,
        iou=args.det_iou,
        verbose=False,
    )[0]

    crop_dir = out_dir / "crops"
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    for i, box in enumerate(det.boxes):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cx1, cy1, cx2, cy2 = kirpme_koordinati(x1, y1, x2, y2, w, h, args.pad, args.square)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        label, cls_conf, _ = siniflandir(classifier, crop, args.cls_imgsz, args.cls_conf)
        det_conf = float(box.conf[0])

        kutu_ciz(frame, (x1, y1, x2, y2), label, det_conf, cls_conf, args.show_det_conf)

        if args.save_crops:
            stem = Path(image_path).stem
            crop_name = f"{stem}_{i:03d}_{dosya_adi_temizle(label)}_{cls_conf:.2f}.jpg"
            cv2.imwrite(str(crop_dir / crop_name), crop)

    out_path = out_dir / Path(image_path).name
    cv2.imwrite(str(out_path), frame)
    print(f"[OK] {image_path} -> {out_path}")


def video_isle(detector, classifier, video_path, out_dir, args):
    """Videoyu kare kare işler ve sınıflandırılmış bir video üretir."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = out_dir / f"{Path(video_path).stem}_classified.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detector.predict(
            frame,
            imgsz=args.det_imgsz,
            conf=args.det_conf,
            iou=args.det_iou,
            verbose=False,
        )[0]

        for box in det.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cx1, cy1, cx2, cy2 = kirpme_koordinati(x1, y1, x2, y2, w, h, args.pad, args.square)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            label, cls_conf, _ = siniflandir(classifier, crop, args.cls_imgsz, args.cls_conf)
            det_conf = float(box.conf[0])
            kutu_ciz(frame, (x1, y1, x2, y2), label, det_conf, cls_conf, args.show_det_conf)

        writer.write(frame)
        frame_no += 1
        if frame_no % 100 == 0:
            print(f"[VIDEO] {frame_no} frame işlendi")

    cap.release()
    writer.release()
    print(f"[OK] {video_path} -> {out_path}")


def main():
    """Modelleri yükler ve kaynak türüne uygun işleme akışını başlatır."""
    args = argumanlari_oku()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONNX dosyası yeniden adlandırılsa bile Ultralytics'in görevi yanlış
    # tahmin etmemesi için model görevlerini açıkça belirtiyoruz.
    detector = YOLO(args.detector, task="detect")
    classifier = YOLO(args.classifier, task="classify")

    kaynaklar, tip = kaynaklari_topla(args.source)
    if tip == "images":
        for image_path in kaynaklar:
            resim_isle(detector, classifier, image_path, out_dir, args)
    else:
        video_isle(detector, classifier, kaynaklar[0], out_dir, args)


if __name__ == "__main__":
    main()
