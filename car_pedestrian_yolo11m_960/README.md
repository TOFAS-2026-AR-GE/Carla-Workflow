# YOLO11m 960 model paketi

Altı sınıf: `car`, `person`, `truck`, `bus`, `bicycle`, `motorcycle`.

En iyi checkpoint 37. epochtur: precision 0.771, recall 0.672, mAP50 0.744 ve mAP50-95 0.514.
Eğitim 47. epoch sonunda kesilmiştir; `best.pt` bundan etkilenmemiştir.

## Kurulum

```bash
cd car_pedestrian_yolo11m_960
pip install -r requirements.txt
```

PyTorch/CUDA kurulumu işletim sistemi ve GPU sürümüne göre ayrıca yapılmalıdır. Yalnız CPU ile ONNX
kullanılacaksa `onnxruntime-gpu` yerine `onnxruntime` kurulabilir.

## İçerik

- `model/yolo11m_960_best.onnx`: ONNX/CARLA kullanımı
- `model/yolo11m_960_best.pt`: eğitim, doğrulama ve yeniden export
- `scripts/detect.py`: görüntü, klasör, video, webcam veya akış üzerinde detector
- `scripts/train_yolo11m_960.py`: parametreli eğitim
- `scripts/validate_yolo11m_960.py`: sınıf bazlı doğrulama ve grafik üretimi
- `scripts/bdd100k_to_yolo.py`: BDD100K JSON etiketlerini YOLO'ya dönüştürür
- `scripts/prepare_combined_6cls.py`: BDD100K ve nuImages'i altı sınıfta birleştirir
- `graphs/` ve `samples/`: eğitim raporu ve örnek tahminler
- `results.csv`: 47 epochluk eğitim geçmişi
- `args.yaml`: Colab eğitiminde kullanılan tarihsel ayarlar; içindeki `/content` yolları çalışma yolu değildir
- `config/combined_6cls.yaml`: dataset köküne kopyalanabilecek taşınabilir YAML şablonu

## Detector

Tek görüntü:

```bash
python3 scripts/detect.py --source test.jpg --device 0
```

Video:

```bash
python3 scripts/detect.py --source video.mp4 --device 0
```

Webcam (`Q`, `Esc` veya terminalde `Ctrl+C` ile durdurulur):

```bash
python3 scripts/detect.py --source 0 --device 0 --show
```

Çıktılar varsayılan olarak `car_pedestrian_yolo11m_960/runs/detector/predict` altına kaydedilir.

## Dataset klasör yapısı

Raw datasetler pakete dahil değildir. Hazırlama öncesinde proje içinde şu yapı bulunmalıdır:

```text
car_pedestrian_yolo11m_960/datasets/
├── bdd100k/bdd100k/images/100k/{train,val}/
├── bdd100k_labels_release/bdd100k/labels/
└── nuimages/
    ├── v1.0-train/
    ├── v1.0-val/
    └── samples/
```

Önce BDD100K etiketlerini dönüştür:

```bash
python3 scripts/bdd100k_to_yolo.py
```

Ardından yaklaşık 25 bin train ve 10 bin validation görüntüsünden oluşan birleşik dataseti üret:

```bash
python3 scripts/prepare_combined_6cls.py --seed 42
```

Bu işlem `datasets/combined_6cls` klasörünü yeniden oluşturur ve içine taşınabilir `dataset.yaml`
yazar. Aynı görüntü sayısını üretmek için aynı BDD100K ve nuImages sürümleri kullanılmalıdır.

## Eğitim ve doğrulama

```bash
python3 scripts/train_yolo11m_960.py \
  --batch 48 --device 0 --imgsz 960 --epochs 50
```

6 GB VRAM'de batch düşürülmelidir. Doğrulama:

```bash
python3 scripts/validate_yolo11m_960.py \
  --batch 1 --device 0
```

Farklı konumdaki dataset için her iki komutta da `--data /yol/dataset.yaml` kullanılabilir.

## Deployment için minimum dosyalar

Sadece CARLA/ONNX kullanımı için `model/yolo11m_960_best.onnx`, `scripts/detect.py` ve
`requirements.txt` yeterlidir. Diğer dosyalar yeniden eğitim ve model raporu içindir.
