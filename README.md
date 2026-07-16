# CARLA Perception ve Araç Kontrolü

Bu branch, CARLA içindeki ego aracı için şu canlı akışı çalıştırır:

1. Ön RGB kameradan YOLO araç tespiti ve OpenCV bbox gösterimi
2. Ön radar ile kamera bbox eşleştirmesi
3. Aynı şeritteki kararlı lead vehicle seçimi
4. Stanley şerit takibi, eğriliğe göre hız planlama, IDM takip ve acil fren

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

- Boylamsal kontrol, literatürdeki Intelligent Driver Model (IDM) denklemini
  kullanır. Serbest yol, yaklaşma ve takip için ayrı fazlar yoktur; her tick'te
  hız, mesafe ve bağıl hızdan tek bir sürekli ivme hesaplanır.
- `2.0 m` duruş boşluğu, `1.2 s` zaman aralığı, `1.5 m/s²` rahat hızlanma ve
  `2.0 m/s²` rahat yavaşlama IDM'nin fiziksel parametreleridir.
- Araç yaklaşık 2 metre boşlukta tamamen durunca `HOLD` moduna geçer. Bu,
  eğimde kaymayı önleyen tek ayrık kontrol durumudur.
- Öndeki araç hareket ettiğinde iki ardışık tick yeterlidir. `HOLD` bırakılır
  ve gaz yine aynı IDM ivmesinden gelir; sabit süre bekleyen bir kalkış fazı
  yoktur. Durum satırındaki kısa `RESTART` etiketi yalnızca gözlem amaçlıdır.
- Düşük hızda pozitif IDM ivmesi CARLA'nın mekanik direncinde kaybolmasın diye
  küçük bir başlangıç gazı uygulanır. Böylece 3-5 metre aralığında araç
  gazsız ve frensiz biçimde sürünmez.
- İvme ve fren komutlarında ayrı jerk limitleri vardır. Gaz ve fren aynı tick'te
  hiçbir zaman birlikte verilmez.
- Kamera ve radar mesafeleri kaynak değiştirirken yakın olan radar ölçümü
  korunur; daha uzağa sıçrayan ölçümler yavaş filtrelenir.
- Radar bağıl hızında negatif değer yaklaşmayı, pozitif değer uzaklaşmayı gösterir.
- Uzak ve yaklaşmayan araç yalnızca gözlemlenir (`LEAD_FAR`). Yaklaşma hızı
  yüksekse IDM uzak mesafede de erken ve yumuşak biçimde yavaşlayabilir.
- Radar-only engel normalde iki ardışık tick doğrulanmadan ACC'ye verilmez.
  Çok yakın veya TTC'si kritik engelde güvenlik için beklenmez.
- CARLA GPU kamera callback'i birkaç tick gecikse bile gerçek kamera frame'i
  inference'a verilir; dünya frame'iyle tam eşleşmeyen görüntü artık atılmaz.
- Ham ön radar, bbox veya tracker oluşmasa bile kısa menzilde bağımsız AEB
  girdisi üretir.
- Durum satırındaki `ctrl_gap`, filtrelenip boylamsal kontrolcüde kullanılan
  gerçek takip mesafesidir; `lead` ise takip katmanının ham seçimini gösterir.
- Stanley kontrolcüsü ön aksın rota başlık ve yanal hatasını kullanır. Direksiyon
  komutu hata filtresi, hız tabanlı limit ve değişim hızı limitiyle yumuşatılır.
- Viraj hedef hızı `v = sqrt(a_y / eğrilik)` bağıntısıyla belirlenir. Rota
  hatası büyüdüğünde araç kontrollü toparlanabilmek için ayrıca yavaşlar.
- AEB normal takipten bağımsızdır. En tehlikeli kamera/radar adayının TTC'sini
  ve 2 metre boşlukta durmak için gereken yavaşlamayı izler; kritik durumda
  normal kontrolü geçersiz kılıp tam fren uygular.

Kontrol denklemlerinin temel kaynakları:

- [Intelligent Driver Model](https://mtreiber.de/publications/micro_tgf99.pdf)
- [Stanley lateral control](https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf)
- [NHTSA Automatic Emergency Braking standard](https://www.federalregister.gov/documents/2024/05/09/2024-09054/federal-motor-vehicle-safety-standards-automatic-emergency-braking-systems-for-light-vehicles)

## Testler

```bash
python -m unittest discover -s tests -v
python -m compileall -q carla_app main.py scripts tests
```

Testler; Stanley yönünü ve rate limitini, viraj hızını, gönderilen düşük hız
logunun tekrarını, gürültülü ölçümde tek seferde 2 metreye duruşu, hızlı
kalkışı, IDM takibini, acil freni, radar doğrulamasını, komşu şerit reddini,
kamera-radar füzyonunu ve levha hatasının araç bbox'ını bozmamasını kapsar.
