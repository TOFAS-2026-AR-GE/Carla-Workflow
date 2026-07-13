# Trafik Levhası Model ve Geliştirme Paketi

Bu paket eğitim datasetlerini içermeden detector ve 78 sınıflı classifier
modellerini çalıştırmak veya modeller üzerinde geliştirme yapmak için gereken
model ve scriptleri içerir.

## İçerik

```text
models/detector.onnx       Trafik levhası detector modeli (512x512)
models/classifier.onnx     78 sınıflı classifier modeli (96x96)
models/detector.pt         Eğitime devam edilebilen detector PyTorch modeli
models/classifier.pt       Eğitime devam edilebilen classifier PyTorch modeli
detect_then_classify.py    İki modeli art arda çalıştıran Python scripti
class_names.json           Classifier sınıf ID/ad eşlemesi
requirements.txt           Gerekli Python paketleri
PROJE_README.md             Dataset hazırlama ve eğitim sürecinin tamamı
scripts/                    Projedeki bütün veri hazırlama/eğitim scriptleri
```

## Mevcut sanal ortamla çalıştırma

Önce paketin kök klasörüne girin:

```bash
cd trafik_levhasi_onnx_paket
```

Mevcut sanal ortamda paketler eksikse:

```bash
pip install -r requirements.txt
```

Bir görüntü veya görüntü klasörü üzerinde CPU ile çalıştırma:

```bash
CUDA_VISIBLE_DEVICES=-1 python3 detect_then_classify.py \
  --source /test/goruntuleri \
  --detector models/detector.onnx \
  --classifier models/classifier.onnx \
  --out outputs \
  --det-conf 0.25 \
  --cls-conf 0.50 \
  --det-imgsz 512 \
  --cls-imgsz 96 \
  --pad 0.25 \
  --square \
  --save-crops \
  --show-det-conf
```

Model yolları varsayılan olarak paket içindeki dosyalara ayarlı olduğu için kısa
kullanım da mümkündür:

```bash
CUDA_VISIBLE_DEVICES=-1 python3 detect_then_classify.py \
  --source /test/goruntuleri
```

Video kaynağı da `--source /video/test.mp4` şeklinde verilebilir.

## PT modellerini kullanma

ONNX yerine PT modelleriyle uçtan uca test:

```bash
python3 detect_then_classify.py \
  --source /test/goruntuleri \
  --detector models/detector.pt \
  --classifier models/classifier.pt \
  --out outputs_pt
```

`.pt` dosyaları fine-tuning, yeniden export veya model üzerinde değişiklik için
saklanmıştır. Dataset hazırlama ve eğitim komutları `PROJE_README.md` içinde,
ilgili Python araçları ise `scripts/` klasöründedir.

## Notlar

- ONNX Runtime GPU paketi yoksa modeller CPU üzerinde çalışır.
- Datasetler boyutları nedeniyle bu pakete dahil değildir.
- `--cls-conf` altındaki sınıflandırmalar çalışma anında `unknown` gösterilir.
- `unknown`, classifier'ın eğitilmiş 78 sınıfından biri değildir.
- Detector girişi `1x3x512x512`, classifier girişi `1x3x96x96` boyutundadır.
