# CARLA Perception ve Araç Kontrolü

Bu branch, CARLA içindeki ego aracı için şu canlı akışı çalıştırır:

1. Ön RGB kameradan YOLO araç tespiti ve OpenCV bbox gösterimi
2. Ön radar ile kamera bbox eşleştirmesi
3. Aynı şeritteki kararlı lead vehicle seçimi
4. Stanley şerit takibi, eğriliğe göre hız planlama, ACC/CBF ve acil fren

Varsayılan canlı kullanımda yalnızca `camera_front_wide` ve
`radar_front_long` açılır. Böylece kullanılmayan 25 sensör simülasyonu ve
gereksiz bellek tüketimi yavaşlatmaz.

## Kurulum

CARLA sunucusunu kullandığın sürümle başlat. Ardından proje ortamında:

```bash
pip install -r requirements.txt
python scripts/check_setup.py
```

`check_setup.py` satırlarının `OK` olması gerekir. Araç modeli repoda şu
konumdadır:

```text
models/vehicle/carla_yolov8n_best.pt
```

Bu modelin araç sınıfı `9: vehicle`'dır. Kod sınıf numarasını sabitlemek yerine
modelin sınıf isimlerini doğrular.

## Çalıştırma

Normal kullanım:

```bash
bash run.sh
```

CARLA ile inference aynı GPU'da sorun çıkarıyorsa CPU kullanımı:

```bash
bash run_cpu.sh
```

Doğrudan `python main.py` da kullanılabilir. OpenCV penceresinde `Q`, `ESC`
veya pencerenin X düğmesi uygulamayı güvenli biçimde kapatır.

## Bbox kontrolü

OpenCV başlığında şu bilgiler görünür:

- `Vehicles`: çizilen araç bbox sayısı
- `Lag`: gösterilen inference sonucunun simülasyona göre frame gecikmesi
- Kırmızı hata satırı: model veya device hatasının kısa açıklaması

Terminal iki saniyede bir tek `[STATUS]` satırı basar. Buradaki `bbox=0`
modelin o sonuçta araç bulmadığını, `detector_errors=vehicle` ise inference'ın
hata verdiğini gösterir. GPU kullanılamazsa araç modeli otomatik olarak CPU'ya
geçer.

## Temel `.env` ayarları

```dotenv
VEHICLE_DEVICE=auto
VEHICLE_CONFIDENCE=0.05
PERCEPTION_EVERY_N_FRAMES=1

ENABLE_SIGN_DETECTION=false
ENABLE_DATA_RECORDING=false

STATUS_PERIOD_SECONDS=2.0
MAX_RUNTIME_SECONDS=0
```

- `VEHICLE_DEVICE=auto`: CUDA varsa `0`, yoksa `cpu` seçer.
- `VEHICLE_CONFIDENCE=0.05`: repodaki özel CARLA modelinin düşük skorlu
  geçerli araç kutularını korur; yalnızca modelin araç sınıfları çizilir.
- Levha tespiti kapalıdır; açılması araç bbox sonucunu artık etkileyemez.
- Dataset kaydı kapalıyken sadece kontrol için gereken iki sensör çalışır.
- `MAX_RUNTIME_SECONDS=0` sınırsız interaktif kullanım demektir. Otomatik test
  için örneğin `30` verilebilir.

Dataset kaydı gerektiğinde `ENABLE_DATA_RECORDING=true` yap. Bu mod tam sensör
paketini açar ve kayıtları `data/runs/` altında oluşturur.

## Kontrol davranışı

- 3 metre, hareket halindeki sabit takip mesafesi değil duruş boşluğudur.
- Hareketli takip hedefi `3 m + 0.9 s × ego hızı` olarak hesaplanır.
- Uzak araç yalnızca gözlemlenir (`LEAD_FAR`); dinamik aktivasyon mesafesine
  girmeden fren komutu üretmez.
- Radar-only engel normalde iki ardışık tick doğrulanmadan ACC'ye verilmez.
  Çok yakın veya TTC'si kritik engelde güvenlik için beklenmez.
- CARLA GPU kamera callback'i birkaç tick gecikse bile gerçek kamera frame'i
  inference'a verilir; dünya frame'iyle tam eşleşmeyen görüntü artık atılmaz.
- Ham ön radar, bbox veya tracker oluşmasa bile kısa menzilde bağımsız AEB
  girdisi üretir.
- Stanley direksiyon komutu hata filtresi, hız tabanlı limit ve değişim hızı
  limitiyle yumuşatılır.

## Testler

```bash
python -m unittest discover -s tests -v
python -m compileall -q carla_app main.py scripts tests
```

Testler; direksiyon yönünü ve rate limitini, uzak aracın fren yaptırmamasını,
yakın araç takibini, acil freni, radar doğrulamasını, komşu şerit reddini,
kamera-radar füzyonunu ve levha hatasının araç bbox'ını bozmamasını kapsar.
