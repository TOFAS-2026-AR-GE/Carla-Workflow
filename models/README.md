# Model dosyaları

Beklenen dosyalar:

- `models/vehicle/carla_yolov8n_best.pt`
- `models/signs/detector.onnx`
- `models/signs/classifier.onnx`
- `models/signs/class_names.json`
- `models/lane/ufld_carla_best.pth` (yalnız UFLD etkinse)

Araç modeli bu repoda bulunur ve `vehicle` sınıfını içerir. Trafik levhası
algılama varsayılan olarak kapalıdır; `.env` içinde
`ENABLE_SIGN_DETECTION=true` yapıldığında iki ONNX modeli ve sınıf adları
dosyası gerekir.

UFLD şerit modeli varsayılan olarak kapalıdır. MIT lisanslı ağırlık ve kaynaklar:

- Model: <https://huggingface.co/jkdxbns/autonomous-driving-carla>
- CARLA uygulaması: <https://github.com/Jkdxbns/autonomous-driving-carla>
- Özgün UFLD: <https://github.com/cfzd/Ultra-Fast-Lane-Detection>

Modeli indirmek için:

```bash
python scripts/download_lane_model.py
```

Ardından `.env` içinde `ENABLE_LANE_DETECTION=true` yapılır. Şerit sonucu ilk
aşamada yalnız görüntüleme/doğrulama içindir; direksiyon kontrolüne bağlanmaz.
Branch içindeki `.env` bu görselleştirmeyi açık getirir. Normal kurulum betiği
modeli sabitlenmiş kaynak revizyonundan otomatik indirip SHA256 ile doğrular;
yukarıdaki komut elle indirme veya yeniden doğrulama için de kullanılabilir.

Eksik modelleri başka bir proje kopyasından almak için:

```bash
python scripts/copy_models.py /home/kullanici/Desktop/Carla-Workflow-main
```
