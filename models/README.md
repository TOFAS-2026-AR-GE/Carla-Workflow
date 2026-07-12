# Hazır CARLA YOLOv8n modeli

Kaynak: <https://github.com/LinkouCommander/CARLA-Object-Detection>

Model 640 piksel çözünürlükte 10 CARLA sınıfı için eğitilmiştir. Sınıf listesi `carla_yolov8n.yaml` içindedir.

Bir görüntü veya video üzerinde çalıştırma:

```powershell
python models/predict_carla.py "goruntu.png"
```

Webcam:

```powershell
python models/predict_carla.py 0 --show
```

Tahminler `models/predictions/carla_yolov8n` altında oluşur.

