# Model dosyaları

Beklenen dosyalar:

- `models/vehicle/carla_yolov8n_best.pt`
- `models/signs/detector.onnx`
- `models/signs/classifier.onnx`

Araç modeli bu repoda bulunur ve `vehicle` sınıfını içerir. Trafik levhası
algılama varsayılan olarak kapalıdır; `.env` içinde
`ENABLE_SIGN_DETECTION=true` yapıldığında iki ONNX modeli de gerekir.

Eksik modelleri başka bir proje kopyasından almak için:

```bash
python scripts/copy_models.py /home/kullanici/Desktop/Carla-Workflow-main
```
