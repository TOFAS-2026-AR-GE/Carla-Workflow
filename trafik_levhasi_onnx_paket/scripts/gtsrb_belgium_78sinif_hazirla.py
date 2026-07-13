#!/usr/bin/env python3
"""GTSRB ile seçilmiş BelgiumTSC sınıflarını 78 sınıflı dataset olarak hazırlar.

Örnek kullanım:
    python3 gtsrb_belgium_78sinif_hazirla.py --temizle

Script PPM görüntülerini PNG'ye dönüştürür, eşdeğer sınıfları birleştirir,
C43 ve yönü karışık D1b kodlarını hariç tutar, train/val splitlerini üretir.
"""

import os
import re
import json
import random
import shutil
import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


# GTSRB 43 sınıfı (id sırasıyla)
GTSRB_BASE_NAMES = [
    "speed_limit_20",                      # 0
    "speed_limit_30",                      # 1
    "speed_limit_50",                      # 2
    "speed_limit_60",                      # 3
    "speed_limit_70",                      # 4
    "speed_limit_80",                      # 5
    "end_speed_limit_80",                  # 6
    "speed_limit_100",                     # 7
    "speed_limit_120",                     # 8
    "no_passing",                          # 9
    "no_passing_trucks",                   # 10
    "priority_at_next_intersection",       # 11
    "priority_road",                       # 12
    "yield",                               # 13
    "stop",                                # 14
    "no_vehicles",                         # 15
    "no_trucks",                           # 16
    "no_entry",                            # 17
    "general_caution",                     # 18
    "dangerous_curve_left",                # 19
    "dangerous_curve_right",               # 20
    "double_curve",                        # 21
    "bumpy_road",                          # 22
    "slippery_road",                       # 23
    "road_narrows_right",                  # 24
    "road_work",                           # 25
    "traffic_signals",                     # 26
    "pedestrians",                         # 27
    "children_crossing",                   # 28
    "bicycles_crossing",                   # 29
    "ice_snow",                            # 30
    "wild_animals",                        # 31
    "end_all_speed_and_passing_limits",    # 32
    "turn_right_ahead",                    # 33
    "turn_left_ahead",                     # 34
    "ahead_only",                          # 35
    "go_straight_or_right",                # 36
    "go_straight_or_left",                 # 37
    "keep_right",                          # 38
    "keep_left",                           # 39
    "roundabout",                          # 40
    "end_no_passing",                      # 41
    "end_no_passing_trucks",               # 42
]

# GTSRB'de doğrudan karşılığı bulunmayan, özellikle ayrılan Belgium sınıfları
EXTRA_CLASS_NAMES = [
    "no_parking",              # 43  <- E1
    "no_stopping",             # 44  <- E3
    "no_left_turn",            # 45  <- C31LEFT
    "no_right_turn",           # 46  <- C31RIGHT
    "bicycle_mandatory",       # 47  <- D7
    "ped_cycle_mandatory",     # 48  <- D10
    "ped_cycle_mofa_mandatory",# 49  <- D9
    "weight_limit",            # 50  <- C21
]

# Yeni class id'leri
NO_PARKING_ID = 43
NO_STOPPING_ID = 44
NO_LEFT_TURN_ID = 45
NO_RIGHT_TURN_ID = 46
BICYCLE_MANDATORY_ID = 47
PED_CYCLE_MANDATORY_ID = 48
PED_CYCLE_MOFA_MANDATORY_ID = 49
WEIGHT_LIMIT_ID = 50


# Belgium class code -> GTSRB class id (merge edilenler)
# Bu kısım konservatif tutuldu.
BELGIUM_MERGE_TO_GTSRB = {
    # Uyarı levhaları
    "A1A": 19,  # curve left
    "A1B": 20,  # curve right
    "A1C": 21,  # double curve
    "A1D": 21,  # double curve (other variant)
    "A13": 22,  # uneven road -> closest: bumpy road
    "A14": 22,  # road bump -> closest: bumpy road
    "A15": 23,  # slippery road
    "A23": 28,  # children
    "A25": 29,  # cyclists
    "A31": 25,  # roadworks
    "A33": 26,  # traffic lights
    "A51": 18,  # general danger
    "A7C": 24,  # road narrows on the right (closest)

    # Öncelik / yasak / yön / zorunlu
    "B1": 13,   # give way
    "B5": 14,   # stop
    "B9": 12,   # priority road
    "B15A": 11, # priority at next intersection (variant)
    "B17": 11,  # priority at next intersection
    "C1": 17,   # no entry
    "C3": 15,   # no vehicles
    "C23": 16,  # no trucks
    "C35": 9,   # no overtaking
    "D1a": 35,  # ahead only
    "D3b": 36,  # go straight or right
    "D5": 40,   # roundabout
}

# Belgium class code -> yeni eklenecek özel class id
BELGIUM_NEW_CLASSES = {
    "E1": NO_PARKING_ID,
    "E3": NO_STOPPING_ID,
    "C31LEFT": NO_LEFT_TURN_ID,
    "C31RIGHT": NO_RIGHT_TURN_ID,
    "D7": BICYCLE_MANDATORY_ID,
    "D10": PED_CYCLE_MANDATORY_ID,
    "D9": PED_CYCLE_MOFA_MANDATORY_ID,
    "C21": WEIGHT_LIMIT_ID,
}

# Bu kodlar Belgium kümesinde birden fazla sayısal değeri aynı sınıfta
# topladığı için eğitime alınmaz. Örneğin C43 hem 50 hem 70 km/s içeriyor;
# hız sınırları yalnızca sayıya göre etiketlenmiş GTSRB'den öğrenilir.
BELGIUM_EXCLUDED_CODES = {"C43", "D1b"}

# Reduced Belgium kümesinde kullanılan kodların sırası. Mevcut bir GTSRB
# sınıfına veya yukarıdaki özel sınıflara eşlenmeyen her kod ayrı tutulur;
# birbirinden farklı levhalar artık tek bir "unknown" sınıfına yığılmaz.
BELGIUM_REDUCED_CODES = [
    "A13", "A14", "A15", "A1A", "A1B", "A1C", "A1D", "A23", "A25", "A29",
    "A31", "A33", "A41", "A51", "A7A", "A7B", "A7C", "B15A", "B17", "B1",
    "B19", "B5", "C1", "C11", "C21", "C23", "C27", "C29", "C3", "C31LEFT",
    "C31RIGHT", "C35", "C43", "D10", "D1a", "D1b", "D3b", "D5", "D7", "D9",
    "E1", "E3", "E5", "E7", "B21", "E9a", "E9a_miva", "E9b", "E9c", "E9d",
    "E9e", "F12a", "F12b", "F19", "F45", "F47", "F49", "F50", "F59", "F87",
    "B11", "B9",
]

# GTSRB'ye birleştirilmeyen Belgium kodlarının kullanıcıya gösterilecek
# açıklayıcı adları. Kodlar yalnızca kaynak eşlemesinde kalır.
BELGIUM_SEPARATE_NAMES = {
    "A29": "cattle_crossing",
    "A41": "guarded_railroad_crossing",
    "A7A": "road_narrows_both_sides",
    "A7B": "road_narrows_left",
    "B19": "yield_to_oncoming_traffic",
    "C11": "no_bicycles",
    "C27": "width_limit",
    "C29": "height_limit",
    "E5": "no_parking_first_half_month",
    "E7": "no_parking_second_half_month",
    "B21": "priority_over_oncoming_traffic",
    "E9a": "parking",
    "E9a_miva": "disabled_parking",
    "E9b": "car_parking",
    "E9c": "truck_parking",
    "E9d": "bus_parking",
    "E9e": "parking_on_sidewalk",
    "F12a": "residential_zone",
    "F12b": "end_residential_zone",
    "F19": "one_way_road",
    "F45": "dead_end",
    "F47": "end_pedestrian_zone",
    "F49": "pedestrian_crossing",
    "F50": "bicycle_crossing",
    "F59": "parking_direction_right",
    "F87": "speed_bump",
    "B11": "end_priority_road",
}

BELGIUM_SEPARATE_CODES = [
    code
    for code in BELGIUM_REDUCED_CODES
    if code not in BELGIUM_MERGE_TO_GTSRB
    and code not in BELGIUM_NEW_CLASSES
    and code not in BELGIUM_EXCLUDED_CODES
]
BELGIUM_SEPARATE_CLASSES = {
    code: len(GTSRB_BASE_NAMES) + len(EXTRA_CLASS_NAMES) + index
    for index, code in enumerate(BELGIUM_SEPARATE_CODES)
}
ALL_CLASS_NAMES = (
    GTSRB_BASE_NAMES
    + EXTRA_CLASS_NAMES
    + [BELGIUM_SEPARATE_NAMES[code] for code in BELGIUM_SEPARATE_CODES]
)


def argumanlari_oku():
    """Kaynak dataset yolları, split oranı ve çıktı seçeneklerini okur."""
    p = argparse.ArgumentParser()
    p.add_argument("--gtsrb-root", default="datasets/gtsrb_cls/raw", help="GTSRB root")
    p.add_argument("--belgium-root", default="datasets/belgium_tsc/raw", help="BelgiumTSC raw root")
    p.add_argument("--reduced-list", default="datasets/belgium_tsc/meta/reducedSetTS.txt", help="Belgium reducedSetTS.txt")
    p.add_argument("--out", default="datasets/gtsrb_belgium_78cls", help="Çıkış dataset klasörü")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Train içinden ayrılacak val oranı")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temizle", action="store_true")
    return p.parse_args()


def sinif_klasoru(class_id: int) -> str:
    """Sınıf ID'sinden sıralamayı koruyan okunabilir klasör adını üretir."""
    return f"{class_id:02d}_{ALL_CLASS_NAMES[class_id]}"


def reduced_codes_oku(txt_yolu):
    """Belgium reducedSetTS dosyasındaki trafik levhası kodlarını okur."""
    codes = []
    with open(txt_yolu, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("62 used traffic signs types"):
                continue
            codes.append(s)
    return codes


def sinif_klasorlerini_olustur(out_dir: Path):
    """Tüm splitlerde bütün hedef sınıf klasörlerini önceden oluşturur."""
    for split in ("train", "val", "test"):
        for cid in range(len(ALL_CLASS_NAMES)):
            (out_dir / split / sinif_klasoru(cid)).mkdir(parents=True, exist_ok=True)


def resim_mi(path: Path):
    """Dosya uzantısının desteklenen bir görüntü formatı olup olmadığını döndürür."""
    return path.suffix.lower() in IMAGE_EXTS


def resmi_kopyala(src: Path, dst: Path):
    """Ultralytics'in desteklemediği PPM dosyalarını PNG'ye dönüştürür."""
    if src.suffix.lower() == ".ppm":
        dst = dst.with_suffix(".png")
        with Image.open(src) as image:
            image.save(dst, format="PNG")
    else:
        shutil.copy2(src, dst)


def kaynak_sinif_klasorleri(root: Path, adaylar):
    """Alternatif adlar arasından var olan ilk sınıf/split klasörünü bulur."""
    for a in adaylar:
        p = root / a
        if p.exists() and p.is_dir():
            return p
    return None


def gtsrb_girdilerini_bul(root: Path):
    """GTSRB train, val ve test dizinlerini farklı ad varyasyonlarıyla bulur."""
    # Önce lower-case, sonra upper-case dene
    train_dir = kaynak_sinif_klasorleri(root, ["train", "Train"])
    val_dir   = kaynak_sinif_klasorleri(root, ["val", "valid", "Val", "Valid"])
    test_dir  = kaynak_sinif_klasorleri(root, ["test", "Test"])

    return train_dir, val_dir, test_dir


def sinif_dosyalari_topla(sinif_dir: Path):
    """Bir sınıf klasöründeki desteklenen görüntü dosyalarını sıralar."""
    return sorted(p for p in sinif_dir.iterdir() if p.is_file() and resim_mi(p))


def gtsrb_kopyala(gtsrb_root: Path, out_dir: Path, val_ratio: float, seed: int):
    """GTSRB görüntülerini hedef sınıflara kopyalar ve gerekirse val ayırır."""
    train_dir, val_dir, test_dir = gtsrb_girdilerini_bul(gtsrb_root)
    if train_dir is None:
        raise FileNotFoundError("GTSRB train klasörü bulunamadı")

    random.seed(seed)
    sayac = {"train": 0, "val": 0, "test": 0}

    # TRAIN / VAL
    for sinif_adi in sorted(d.name for d in train_dir.iterdir() if d.is_dir()):
        cid = int(sinif_adi)
        hedef_train = out_dir / "train" / sinif_klasoru(cid)
        hedef_val   = out_dir / "val" / sinif_klasoru(cid)

        dosyalar = sinif_dosyalari_topla(train_dir / sinif_adi)

        if val_dir and (val_dir / sinif_adi).exists():
            train_files = dosyalar
            val_files = sinif_dosyalari_topla(val_dir / sinif_adi)
        else:
            dosyalar = dosyalar[:]
            random.shuffle(dosyalar)
            val_adet = int(len(dosyalar) * val_ratio)
            if len(dosyalar) >= 2 and val_adet == 0:
                val_adet = 1
            val_files = dosyalar[:val_adet]
            train_files = dosyalar[val_adet:]

        for src in train_files:
            dst = hedef_train / f"gtsrb_{src.name}"
            resmi_kopyala(src, dst)
            sayac["train"] += 1

        for src in val_files:
            dst = hedef_val / f"gtsrb_{src.name}"
            resmi_kopyala(src, dst)
            sayac["val"] += 1

    # TEST
    if test_dir:
        for sinif_adi in sorted(d.name for d in test_dir.iterdir() if d.is_dir()):
            cid = int(sinif_adi)
            hedef_test = out_dir / "test" / sinif_klasoru(cid)
            for src in sinif_dosyalari_topla(test_dir / sinif_adi):
                dst = hedef_test / f"gtsrb_{src.name}"
                resmi_kopyala(src, dst)
                sayac["test"] += 1

    return sayac


def belgium_sinif_hedefi(code: str) -> int:
    """Belgium kodunu birleştirilmiş datasetteki hedef sınıf ID'sine eşler."""
    if code in BELGIUM_MERGE_TO_GTSRB:
        return BELGIUM_MERGE_TO_GTSRB[code]
    if code in BELGIUM_NEW_CLASSES:
        return BELGIUM_NEW_CLASSES[code]
    if code in BELGIUM_SEPARATE_CLASSES:
        return BELGIUM_SEPARATE_CLASSES[code]
    raise ValueError(f"Eşlemesi tanımlanmamış Belgium sınıf kodu: {code}")


def belgium_resim_hedefi(code: str, src: Path) -> int:
    """Bir Belgium görüntüsünün hedef ID'sini belirler; genişlemeye açık katmandır."""
    return belgium_sinif_hedefi(code)


def belgium_kopyala(belgium_root: Path, reduced_codes, out_dir: Path, val_ratio: float, seed: int):
    """Belgium görüntülerini eşler, böler, PNG'ye çevirir ve hedefe yazar."""
    train_dir = belgium_root / "Training"
    test_dir = belgium_root / "Testing"

    if not train_dir.exists():
        raise FileNotFoundError("Belgium Training klasörü bulunamadı")
    if not test_dir.exists():
        raise FileNotFoundError("Belgium Testing klasörü bulunamadı")

    random.seed(seed)
    sayac = {"train": 0, "val": 0, "test": 0}
    kod_raporu = {}

    # 00000..00061 -> reduced_codes sırasına göre
    for sinif_dir in sorted(d for d in train_dir.iterdir() if d.is_dir()):
        idx = int(sinif_dir.name)
        if idx >= len(reduced_codes):
            continue

        code = reduced_codes[idx]
        if code in BELGIUM_EXCLUDED_CODES:
            continue
        kod_raporu[code] = belgium_sinif_hedefi(code)

        dosyalar = sinif_dosyalari_topla(sinif_dir)
        dosyalar = dosyalar[:]

        random.shuffle(dosyalar)
        val_adet = int(len(dosyalar) * val_ratio)
        if len(dosyalar) >= 2 and val_adet == 0:
            val_adet = 1
        val_files = dosyalar[:val_adet]
        train_files = dosyalar[val_adet:]

        for src in train_files:
            hedef_cid = belgium_resim_hedefi(code, src)
            hedef_train = out_dir / "train" / sinif_klasoru(hedef_cid)
            dst = hedef_train / f"belgium_{code}_{src.name}"
            resmi_kopyala(src, dst)
            sayac["train"] += 1

        for src in val_files:
            hedef_cid = belgium_resim_hedefi(code, src)
            hedef_val = out_dir / "val" / sinif_klasoru(hedef_cid)
            dst = hedef_val / f"belgium_{code}_{src.name}"
            resmi_kopyala(src, dst)
            sayac["val"] += 1

    for sinif_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        idx = int(sinif_dir.name)
        if idx >= len(reduced_codes):
            continue

        code = reduced_codes[idx]
        if code in BELGIUM_EXCLUDED_CODES:
            continue
        for src in sinif_dosyalari_topla(sinif_dir):
            hedef_cid = belgium_resim_hedefi(code, src)
            hedef_test = out_dir / "test" / sinif_klasoru(hedef_cid)
            dst = hedef_test / f"belgium_{code}_{src.name}"
            resmi_kopyala(src, dst)
            sayac["test"] += 1

    return sayac, kod_raporu


def main():
    """Çıktıyı hazırlar, iki dataseti birleştirir ve eşleme raporlarını kaydeder."""
    args = argumanlari_oku()

    gtsrb_root = Path(args.gtsrb_root)
    belgium_root = Path(args.belgium_root)
    reduced_txt = Path(args.reduced_list)
    out_dir = Path(args.out)

    if args.temizle and out_dir.exists():
        shutil.rmtree(out_dir)

    sinif_klasorlerini_olustur(out_dir)

    reduced_codes = reduced_codes_oku(reduced_txt)
    print(f"[OKUNDU] Belgium reduced class code sayısı: {len(reduced_codes)}")

    gtsrb_sayac = gtsrb_kopyala(gtsrb_root, out_dir, args.val_ratio, args.seed)
    print(f"[GTSRB] train={gtsrb_sayac['train']} val={gtsrb_sayac['val']} test={gtsrb_sayac['test']}")

    belgium_sayac, kod_raporu = belgium_kopyala(belgium_root, reduced_codes, out_dir, args.val_ratio, args.seed)
    print(f"[BELGIUM] train={belgium_sayac['train']} val={belgium_sayac['val']} test={belgium_sayac['test']}")

    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(ALL_CLASS_NAMES)}, f, ensure_ascii=False, indent=2)

    with open(out_dir / "belgium_code_mapping.json", "w", encoding="utf-8") as f:
        json.dump(kod_raporu, f, ensure_ascii=False, indent=2)

    print(f"\n[BAŞARILI] {len(ALL_CLASS_NAMES)} sınıflı birleşik classification dataset hazırlandı")
    print(f"Çıktı klasörü: {out_dir}")
    print(f"Toplam sınıf: {len(ALL_CLASS_NAMES)}")
    print("Ek sınıflar:")
    print("  43 -> no_parking")
    print("  44 -> no_stopping")
    print("  45 -> no_left_turn")
    print("  46 -> no_right_turn")
    print("  47 -> bicycle_mandatory")
    print("  48 -> ped_cycle_mandatory")
    print("  49 -> ped_cycle_mofa_mandatory")
    print(f"Ayrı tutulan ek Belgium kodu: {len(BELGIUM_SEPARATE_CODES)}")
    print(f"Hariç tutulan Belgium kodları: {', '.join(sorted(BELGIUM_EXCLUDED_CODES))}")


if __name__ == "__main__":
    main()
