#!/usr/bin/env python3
"""YOLO detection örneklerini sınıf dağılımını gözeterek yeniden böler.

Örnek kullanım:
    python3 yeniden_split.py --data datasets/gtsdb_belgium_tek_sinif/data.yaml \
      --out datasets/gtsdb_belgium_tek_sinif_resplit \
      --val_ratio 0.10 --test_ratio 0.10 --seed 42
"""

from pathlib import Path
from collections import Counter
import argparse
import random
import shutil
import yaml

RESIM_UZANTILARI = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def argumanlari_oku():
    """Kaynak YAML, çıktı ve split oranlarını okur."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Eski data.yaml yolu")
    p.add_argument("--out", default="datasets/gtsdb_belgium_tek_sinif_resplit", help="Yeni dataset klasörü")
    p.add_argument("--val_ratio", type=float, default=0.10, help="Validation oranı")
    p.add_argument("--test_ratio", type=float, default=0.10, help="Test oranı")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def isim_listesi(data):
    """YAML içindeki dict veya liste sınıf adlarını sıralı listeye çevirir."""
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def mutlak_yol(base_dir: Path, p: str) -> Path:
    """Göreli dataset yolunu data.yaml dizinine göre mutlaklaştırır."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def label_klasoru_bul(images_dir: Path) -> Path:
    """Bir images klasörüne karşılık gelen labels klasörünü bulur."""
    if images_dir.name == "images":
        return images_dir.parent / "labels"
    return Path(str(images_dir).replace("/images", "/labels"))


def resmi_bul(images_dir: Path, stem: str):
    """Dosya kök adına uyan desteklenen uzantılı resmi bulur."""
    for ext in RESIM_UZANTILARI:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
        p2 = images_dir / f"{stem}{ext.upper()}"
        if p2.exists():
            return p2
    return None


def siniflari_oku(label_path: Path):
    """Bir YOLO label dosyasında bulunan benzersiz sınıf ID'lerini okur."""
    siniflar = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            sinif_id = int(float(satir.split()[0]))
            siniflar.add(sinif_id)
    return siniflar


def tum_ornekleri_topla(data_yaml_yolu: Path, data: dict):
    """Eski splitlerdeki benzersiz görüntü-label çiftlerini tek listede toplar."""
    base = data_yaml_yolu.parent

    splitler = [
        ("train", data.get("train")),
        ("val", data.get("val") or data.get("valid")),
        ("test", data.get("test")),
    ]

    ornekler = []
    gorulenler = set()

    for _, image_rel in splitler:
        if not image_rel:
            continue

        images_dir = mutlak_yol(base, image_rel)
        labels_dir = label_klasoru_bul(images_dir)

        if not images_dir.exists() or not labels_dir.exists():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            image_path = resmi_bul(images_dir, stem)
            if image_path is None:
                continue

            key = str(image_path.resolve())
            if key in gorulenler:
                continue
            gorulenler.add(key)

            ornekler.append({
                "image_path": image_path,
                "label_path": label_path,
                "stem": stem,
                "classes": siniflari_oku(label_path),
            })

    return ornekler


def split_sinif_sayilari(ornekler):
    """Bir örnek listesinde her sınıfın kaç görüntüde bulunduğunu sayar."""
    c = Counter()
    for ex in ornekler:
        for cid in ex["classes"]:
            c[cid] += 1
    return c


def nadirlik_puani(ornek, toplam_sinif_sayisi):
    """Nadir sınıfları içeren örneklere daha yüksek seçim puanı verir."""
    if not ornek["classes"]:
        return 0.0
    return sum(1.0 / toplam_sinif_sayisi[c] for c in ornek["classes"])


def uygun_ornek_tasi(kaynak_liste, hedef_liste, sinif_id, toplam_sinif_sayisi):
    """Belirli sınıfı taşıyan en nadirlik-değerli örneği hedef splite taşır."""
    adaylar = [ex for ex in kaynak_liste if sinif_id in ex["classes"]]
    if not adaylar:
        return False

    adaylar.sort(key=lambda ex: nadirlik_puani(ex, toplam_sinif_sayisi), reverse=True)
    secilen = adaylar[0]
    kaynak_liste.remove(secilen)
    hedef_liste.append(secilen)
    return True


def klasorleri_olustur(out_dir: Path):
    """Yeni train/val/test images ve labels klasörlerini oluşturur."""
    for split in ["train", "val", "test"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def spliti_yaz(out_dir: Path, split_adi: str, ornekler):
    """Bir splitin görüntü ve label dosyalarını benzersiz adlarla kopyalar."""
    images_out = out_dir / split_adi / "images"
    labels_out = out_dir / split_adi / "labels"

    for i, ex in enumerate(ornekler):
        yeni_stem = f"{i:05d}_{ex['stem']}"
        hedef_resim = images_out / f"{yeni_stem}{ex['image_path'].suffix.lower()}"
        hedef_label = labels_out / f"{yeni_stem}.txt"

        shutil.copy2(ex["image_path"], hedef_resim)
        shutil.copy2(ex["label_path"], hedef_label)


def data_yaml_yaz(out_dir: Path, names):
    """Yeni dataset yollarını ve sınıf adlarını içeren data.yaml yazar."""
    data = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    """Örnekleri toplar, dengeli splitlere ayırır ve yeni dataseti üretir."""
    args = argumanlari_oku()
    random.seed(args.seed)

    data_yaml_yolu = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()

    with open(data_yaml_yolu, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = isim_listesi(data)
    tum_ornekler = tum_ornekleri_topla(data_yaml_yolu, data)

    if not tum_ornekler:
        print("Hiç örnek bulunamadı.")
        return

    toplam_resim = len(tum_ornekler)
    hedef_val = round(toplam_resim * args.val_ratio)
    hedef_test = round(toplam_resim * args.test_ratio)

    toplam_sinif_sayisi = split_sinif_sayilari(tum_ornekler)

    train_liste = tum_ornekler[:]
    random.shuffle(train_liste)
    val_liste = []
    test_liste = []

    for cid, toplam in sorted(toplam_sinif_sayisi.items(), key=lambda x: x[1]):
        if toplam >= 8:
            if split_sinif_sayilari(val_liste)[cid] < 1 and len(val_liste) < hedef_val:
                uygun_ornek_tasi(train_liste, val_liste, cid, toplam_sinif_sayisi)
            if split_sinif_sayilari(test_liste)[cid] < 1 and len(test_liste) < hedef_test:
                uygun_ornek_tasi(train_liste, test_liste, cid, toplam_sinif_sayisi)

        if toplam >= 20:
            if split_sinif_sayilari(val_liste)[cid] < 2 and len(val_liste) < hedef_val:
                uygun_ornek_tasi(train_liste, val_liste, cid, toplam_sinif_sayisi)
            if split_sinif_sayilari(test_liste)[cid] < 2 and len(test_liste) < hedef_test:
                uygun_ornek_tasi(train_liste, test_liste, cid, toplam_sinif_sayisi)

    random.shuffle(train_liste)

    while len(val_liste) < hedef_val and train_liste:
        val_liste.append(train_liste.pop())

    while len(test_liste) < hedef_test and train_liste:
        test_liste.append(train_liste.pop())

    final_train = train_liste
    final_val = val_liste
    final_test = test_liste

    if out_dir.exists():
        shutil.rmtree(out_dir)

    klasorleri_olustur(out_dir)
    spliti_yaz(out_dir, "train", final_train)
    spliti_yaz(out_dir, "val", final_val)
    spliti_yaz(out_dir, "test", final_test)
    data_yaml_yaz(out_dir, names)

    print("\n[YENİ SPLIT TAMAMLANDI]")
    print(f"Çıktı klasörü: {out_dir}")
    print(f"train images: {len(final_train)}")
    print(f"val images:   {len(final_val)}")
    print(f"test images:  {len(final_test)}")


if __name__ == "__main__":
    main()
