# CARLA Modular Perception

Bu proje mevcut CARLA akisini daha sade ve moduler hale getirir.

## Klasorler

```text
carla_app/core/           CARLA, arac, trafik ve kontrol
carla_app/sensors/        Sensorler, senkronizasyon ve veri kaydi
carla_app/perception/     Vehicle ve trafik levhasi modelleri
carla_app/visualization/  OpenCV penceresi
scenarios/                Harita ve trafik ayarlari
models/                   Model dosyalari
scripts/                  Kurulum yardimcilari
```

## 1. Model dosyalarini kopyala

Eski proje klasorun ornegin su ise:

```text
/home/tofasailab1/Desktop/Carla-Workflow-main
```

Calistir:

```bash
python scripts/copy_models.py /home/tofasailab1/Desktop/Carla-Workflow-main
```

## 2. Paketleri kontrol et

```bash
pip install -r requirements.txt
python scripts/check_setup.py
```

## 3. CARLA'yi baslat ve projeyi calistir

CPU modu CUDA cakismasini engeller:

```bash
bash run_cpu.sh
```

Normal calistirma:

```bash
python main.py
```

OpenCV penceresinde `Q` veya `ESC` ile cikilir.

## Ayarlar

Tum temel ayarlar `.env` icindedir. Varsayilanlar:

- Town03
- 40 trafik araci
- 0.05 saniye senkron adim
- 800x600 RGB kamera
- Her 5 frame'de veri kaydi
- Her 2 frame'de perception istegi
- Vehicle ve trafik levhasi modelleri CPU'da

GPU kullanmak icin `.env` icinde cihazlari `0` yapabilirsin. Quadro T1000 ile CARLA ayni GPU'yu kullaniyorsa CPU ayari daha guvenlidir.

## Veri cikisi

Kayitlar `data/runs/run_TARIH_SAAT/` altinda olusur:

```text
rgb/       PNG kamera goruntuleri
lidar/     NPY nokta bulutlari
metadata/  Arac, GNSS, IMU ve radar JSON verileri
```
