# Trafik Levhası Tespit ve Sınıflandırma Projesi

Bu proje iki aşamalı bir trafik levhası sistemi oluşturur:

1. **Detector**, görüntüdeki trafik levhalarını tek `traffic_sign` sınıfıyla bulur.
2. **Classifier**, detector tarafından kırpılan levhayı 78 sınıftan biri olarak sınıflandırır.

Temel akış:

```text
Görüntü/video
    -> YOLO detector (levha kutusu)
    -> kutuyu kırpma ve kareye tamamlama
    -> YOLO classifier (78 levha sınıfı)
    -> kutulu ve etiketli çıktı
```

## Kullanılan veri setleri

### Detection

- **GTSDB**: Alman trafik sahneleri ve levha bounding-box etiketleri.
- **BelgiumTSD**: Belçika trafik sahneleri ve bounding-box etiketleri.

İki kaynak detection aşamasında tek sınıfa indirgenmiştir:

```text
class 0 = traffic_sign
```

Detector'ın görevi levhanın türünü söylemek değil, görüntüde levha bulunan bölgeyi bulmaktır.

### Classification

- **GTSRB**: 43 sınıflı Alman trafik levhası crop veri seti.
- **BelgiumTSC**: Belçika trafik levhası crop veri seti.

GTSRB ve seçilmiş BelgiumTSC sınıfları birleştirilerek **78 sınıflı** classification veri seti oluşturulmuştur.

## Uygulanan işlemler

### 1. Belgium detection verisini YOLO formatına çevirme

`belgium_to_yolo.py`, BelgiumTSD annotation satırlarını okur, JP2 görüntülerini JPG'ye çevirir ve YOLO label dosyaları üretir.

```bash
python3 belgium_to_yolo.py \
  --root datasets/belgium \
  --out datasets/belgium/belgium_yolo
```

### 2. Detection sınıflarını tek sınıfa indirme

`tek_sinif_yap.py`, kaynak YOLO etiketlerindeki bütün sınıf ID'lerini `0` yapar.

```bash
python3 tek_sinif_yap.py \
  --girdi datasets/gtsdb/gtsdb_yolo \
  --cikti datasets/gtsdb/gtsdb_tek_sinif \
  --sinif_adi traffic_sign
```

Aynı işlem gerektiğinde Belgium YOLO verisine de uygulanır.

### 3. GTSDB ve Belgium detection verilerini birleştirme

`yolo_birlestir.py`, iki datasetin görüntü ve label dosyalarını birleştirir. Dosya çakışmasını önlemek için kaynaklara `a_` ve `b_` öneki ekler:

- `a_`: birinci kaynak (GTSDB)
- `b_`: ikinci kaynak (Belgium)

```bash
python3 yolo_birlestir.py \
  --a datasets/gtsdb/gtsdb_tek_sinif \
  --b datasets/belgium/belgium_tek_sinif \
  --out datasets/gtsdb_belgium_tek_sinif
```

### 4. Detection train/val/test splitlerini yeniden oluşturma

`yeniden_split.py`, bütün detection örneklerini toplar ve nadir sınıfları gözeterek yeniden böler.

```bash
python3 yeniden_split.py \
  --data datasets/gtsdb_belgium_tek_sinif/data.yaml \
  --out datasets/gtsdb_belgium_tek_sinif_resplit \
  --val_ratio 0.10 \
  --test_ratio 0.10 \
  --seed 42
```

Detector için kullanılan son dataset:

```text
datasets/gtsdb_belgium_tek_sinif_resplit
```

### 5. GTSRB ve Belgium classification verilerini hazırlama

`gtsrb_belgium_78sinif_hazirla.py` şu işlemleri yapar:

- GTSRB'nin 43 sınıfını okunabilir adlarla kopyalar.
- Belgium PPM görüntülerini gerçek PNG formatına dönüştürür.
- Anlamı aynı olan Belgium ve GTSRB sınıflarını birleştirir.
- GTSRB'de bulunmayan Belgium levhalarına okunabilir adlar verir.
- `C21` sınıfını genel `weight_limit` adıyla tutar.
- Farklı hız değerlerini tek sınıfta karıştıran Belgium `C43` kodunu hariç tutar.
- Hem sol hem sağ yön içeren Belgium `D1b` kodunu hariç tutar.
- `train` ve `val` klasörlerini hazırlar.

```bash
python3 gtsrb_belgium_78sinif_hazirla.py --temizle
```

Son classification dataseti:

```text
datasets/gtsrb_belgium_78cls
train: 36933 görüntü / 78 sınıf
val:    6475 görüntü / 78 sınıf
```

Sınıf adları ve Belgium eşlemeleri dataset kökündeki şu dosyalarda bulunur:

```text
class_names.json
belgium_code_mapping.json
```

## Model eğitimleri

### Detector eğitimi

Detector için YOLOv8s, `512x512` giriş boyutu ve 50 epoch kullanıldı.

```bash
python3 train_detector.py \
  --data datasets/gtsdb_belgium_tek_sinif_resplit/data.yaml \
  --model yolov8s.pt \
  --epochs 50 \
  --imgsz 512 \
  --batch 16 \
  --proje_adi gtsdb_belgium_tek_sinif_s \
  --run_adi egitim
```

Son detector ağırlığı:

```text
runs/detect/runs/detect/gtsdb_belgium_tek_sinif_s/egitim/weights/best.pt
```

### Classifier eğitimi

Classifier için YOLO11n-cls, `96x96` giriş boyutu ve 30 epoch kullanıldı.

Trafik levhalarında yatay çevirme sol ve sağ anlamını değiştirdiği için classifier eğitiminde aşağıdaki augmentation'lar kapalıdır:

```text
fliplr = 0.0
flipud = 0.0
```

Bu güvenli ayarlar `train_classifier.py` içinde sabittir.

```bash
python3 train_classifier.py \
  --data datasets/gtsrb_belgium_78cls \
  --model yolo11n-cls.pt \
  --epochs 30 \
  --imgsz 96 \
  --batch 32 \
  --run_adi gtsrb_belgium_78cls_no_flip_yolo11n
```

Son classifier ağırlığı:

```text
runs/classify/runs/classify/gtsrb_belgium_78cls_no_flip_yolo11n/weights/best.pt
```

## ONNX modelleri

Detector ONNX:

```text
runs/detect/runs/detect/gtsdb_belgium_tek_sinif_s/egitim/weights/best.onnx
Girdi: 1x3x512x512
```

Classifier ONNX:

```text
runs/classify/runs/classify/gtsrb_belgium_78cls_no_flip_yolo11n/weights/best.onnx
Girdi: 1x3x96x96
Çıktı: 1x78
```

Yeniden export etmek için:

```bash
yolo export \
  model=runs/detect/runs/detect/gtsdb_belgium_tek_sinif_s/egitim/weights/best.pt \
  format=onnx imgsz=512 batch=1 opset=17 dynamic=False

yolo export \
  model=runs/classify/runs/classify/gtsrb_belgium_78cls_no_flip_yolo11n/weights/best.pt \
  format=onnx imgsz=96 batch=1 opset=17 dynamic=False
```

## Uçtan uca test

PT veya ONNX modelleri `detect_then_classify.py` ile birlikte kullanılabilir. İki ONNX modeliyle örnek:

```bash
CUDA_VISIBLE_DEVICES=-1 python3 detect_then_classify.py \
  --source datasets/gtsdb_belgium_tek_sinif_resplit/test/images \
  --detector runs/detect/runs/detect/gtsdb_belgium_tek_sinif_s/egitim/weights/best.onnx \
  --classifier runs/classify/runs/classify/gtsrb_belgium_78cls_no_flip_yolo11n/weights/best.onnx \
  --out runs/detect_then_classify_onnx_no_flip_test \
  --det-conf 0.25 \
  --cls-conf 0.50 \
  --det-imgsz 512 \
  --cls-imgsz 96 \
  --pad 0.25 \
  --square \
  --save-crops \
  --show-det-conf
```

`cls-conf` değerinin altındaki classifier sonuçları çalışma anında `unknown` olarak gösterilir. `unknown`, eğitimde bulunan bir sınıf değildir.

## Scriptler

| Script | Görevi |
|---|---|
| `belgium_to_yolo.py` | Belgium detection annotationlarını YOLO formatına çevirir. |
| `tek_sinif_yap.py` | Detection etiketlerini tek `traffic_sign` sınıfına indirger. |
| `yolo_birlestir.py` | İki YOLO detection datasetini birleştirir. |
| `yeniden_split.py` | Detection train/val/test splitlerini yeniden oluşturur. |
| `gtsrb_belgium_78sinif_hazirla.py` | 78 sınıflı classification datasetini hazırlar. |
| `train_detector.py` | Detector modelini eğitir. |
| `train_classifier.py` | Classifier modelini yön-koruyan ayarlarla eğitir. |
| `detect_then_classify.py` | Detection ve classification modellerini art arda çalıştırır. |

Her scriptin komut satırı seçenekleri şu şekilde görülebilir:

```bash
python3 script_adi.py --help
```

## Önemli notlar

- Classifier eğitiminde `fliplr` açılmamalıdır; sol ve sağ levhaların etiketlerini bozar.
- Belgium `C43`, farklı hız değerlerini aynı kodda topladığı için kullanılmamıştır.
- Belgium `D1b`, sol ve sağ yönleri aynı kodda topladığı için kullanılmamıştır.
- Classification dataseti bağımsız, eksiksiz bir test splitine sahip değildir. Gerçek sistem testi birleşik detection test görüntülerinde uçtan uca yapılır.
- ONNX Runtime GPU paketi yoksa ONNX modelleri CPU üzerinde çalışır.
