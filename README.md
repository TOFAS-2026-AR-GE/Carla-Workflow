# CARLA Araç Takibi ve Kontrolü

Bu branch, CARLA içindeki ego aracını şerit merkezinde sürer, önündeki aracı
kamera ve radar ile takip eder, viraja göre hızını ayarlar ve kritik çarpışma
riskinde acil fren uygular.

Kodun ana ilkesi şudur: her dosya tek bir iş yapar ve ana araç döngüsü bu
parçaları açık bir sırayla çağırır.

## Sistem akışı

Her CARLA karesinde aşağıdaki sıra izlenir:

1. `application.py` dünya karesini ilerletir ve araç durumunu okur.
2. `sensors/manager.py` ön kamera ile ön radarın en güncel verisini verir.
3. `perception/system.py` kamera görüntüsündeki araçları bulur.
4. `controller/vehicle/lead_vehicle.py` kamera ile radarı birleştirip takip
   edilecek ön aracı seçer.
5. `controller/vehicle/vehicle_controller.py` direksiyon, hedef hız, gaz ve
   fren değerlerini hesaplar.
6. `application.py` hesaplanan komutu ego aracına uygular.

## Klasör ve dosya rehberi

| Konum | Görevi |
|---|---|
| `main.py` | Uygulamayı başlatan tek giriş noktasıdır. |
| `carla_app/application.py` | Açılış, ana döngü, durum çıktısı ve kapanışı yönetir. |
| `carla_app/config.py` | `.env` ayarlarını tek yerde okur ve doğrular. |
| `carla_app/core/` | CARLA bağlantısı, ego araç, trafik, rota ve araç durumunu yönetir. |
| `carla_app/sensors/layout.py` | Sensör adlarını, konumlarını ve CARLA ayarlarını tanımlar. |
| `carla_app/sensors/factory.py` | Tanımlanan sensörleri CARLA üzerinde oluşturur. |
| `carla_app/sensors/manager.py` | Sensörleri açar, okur, kaydeder ve kapatır. |
| `carla_app/sensors/stream.py` | Canlı ön kamera ve ön radar verisini güvenli biçimde saklar. |
| `carla_app/sensors/sync.py` | Veri kaydında bütün sensörlerin aynı karesini bekler. |
| `carla_app/sensors/processors.py` | CARLA sensör verisini NumPy ve sözlük biçimine çevirir. |
| `carla_app/sensors/writer.py` | Tam sensör paketini `data/runs/` altına kaydeder. |
| `carla_app/perception/` | YOLO araç tespiti, isteğe bağlı levha tespiti ve kamera-radar eşleştirmesidir. |
| `carla_app/controller/vehicle/` | Ön araç seçimi, direksiyon, hız, gaz-fren ve acil frendir. |
| `carla_app/visualization/viewer.py` | Kamera ve araç kutularını OpenCV penceresinde gösterir. |
| `carla_app/visualization/sensor_layout.py` | Sensör yerleşimini tarayıcı verisine dönüştürür. |
| `carla_app/visualization/sensor_layout.html` | Araba şeklindeki sensör ekranıdır. |
| `scripts/` | Kurulum kontrolü, model kopyalama ve sensör ekranı komutlarıdır. |
| `scenarios/traffic.yaml` | Harita, zaman adımı ve trafik ayarlarıdır. |
| `tests/` | Kontrol, algılama, rota ve sensör davranış testleridir. |

## Kurulum

Python 3.10 ortamında:

```bash
cd ~/Desktop/Carla-Workflow-Modular
pip install -r requirements.txt
python scripts/check_setup.py
```

`check_setup.py` çıktısında gerekli paketler ve araç modeli `OK` görünmelidir.
Araç modeli şu konumdadır:

```text
models/vehicle/carla_yolov8n_best.pt
```

## Normal çalıştırma

Önce kullandığın sürüme uygun CARLA sunucusunu aç. Ardından:

```bash
cd ~/Desktop/Carla-Workflow-Modular
bash run.sh
```

CARLA ile araç modeli aynı ekran kartında sorun çıkarırsa:

```bash
bash run_cpu.sh
```

OpenCV penceresinde `Q`, `ESC` veya pencerenin kapatma düğmesi uygulamayı
güvenli biçimde sonlandırır.

## Sensör yerleşimini görme

Normal uygulama çalışırken ikinci terminalde:

```bash
cd ~/Desktop/Carla-Workflow-Modular
./run_sensor_viewer.sh
```

Tarayıcıda aracın üstten ve yandan görünümü açılır. Parlak sensörler o anda
çalışmaktadır; saydam sensörler yerleşimde vardır fakat normal kontrol modunda
açılmamıştır. Bir sensöre tıklayınca konumu, yönü, görüş açısı ve gerçek
menzili gösterilir.

Tarayıcı otomatik açılmazsa:

```bash
./run_sensor_viewer.sh --no-browser
xdg-open /tmp/carla_sensor_layout.html
```

Bu ekran ROS veya RViz kullanmaz.

## Sensör düzeni

Normal araç kontrolünde yalnızca iki sensör açılır:

| Sensör | Kullanım amacı |
|---|---|
| `camera_front_wide` | YOLO araç tespiti ve hedef kimliği |
| `radar_front_long` | Mesafe, bağıl hız, yedek takip ve acil fren |

Veri kaydı açıldığında toplam 15 gerçek CARLA sensörü kullanılır:

| Tür | Sayı |
|---|---:|
| RGB kamera | 7 |
| Otomotiv radarı | 5 |
| LiDAR | 1 |
| GNSS | 1 |
| IMU | 1 |

CARLA'da gerçek ultrasonik sensör olmadığı için radar ile ultrasonik taklidi
yapılmaz. Önceden bu amaçla kullanılan 12 kısa menzil radar bu branch'ten
tamamen kaldırılmıştır.

## Kontrol dosyaları

| Dosya | Girdi | Çıktı |
|---|---|---|
| `lead_vehicle.py` | Kamera kutuları, radar noktaları, rota | Takip edilecek ön araç ve acil fren adayı |
| `tracking.py` | Birleştirilmiş araç ölçümleri | Kareler arasında sabit araç kimliği ve yumuşatılmış hareket |
| `stanley_controller.py` | Araç konumu, yönü ve referans rota | `-1` ile `+1` arasında direksiyon |
| `speed_planner.py` | Rota eğriliği, hız ve şerit hatası | Metre/saniye cinsinden güvenli hedef hız |
| `longitudinal_controller.py` | Ego hızı, hedef hız ve ön araç | Birbirini dışlayan gaz veya fren |
| `safety_supervisor.py` | Ön araç ve ham radar tehlikesi | Acil fren gerekli mi bilgisi |
| `vehicle_controller.py` | Bütün kontrol girdileri | CARLA `VehicleControl` komutu |

### Direksiyon

Stanley kontrolcüsü aracın ön kontrol noktasını referans rotaya izdüşürür.
Rota yönü ile araç yönü arasındaki farkı ve şerit merkezine olan yanal hatayı
birleştirir. Araç şerit kenarına yaklaşırsa merkezleme etkisi artar. Direksiyon
değişimi hıza göre sınırlandırıldığı için ani sağ-sol komut üretilmez.

### Hedef hız

Hız planlayıcı yaklaşık üç saniyelik yolu inceler. Viraj eğriliğine göre rahat
yanal ivmeyle alınabilecek hızı hesaplar. Şerit hatası büyürse toparlanma hızı
seçilir. Güvenlik için yavaşlama geciktirilmez; yeniden hızlanma kademeli olur.

### Gaz, fren ve ön araç takibi

Boylamsal kontrol IDM kullanır. Temel ayarlar:

- Duruş boşluğu: `2.0 m`
- Zaman aralığı: `1.2 s`
- Rahat hızlanma: `1.5 m/s²`
- Rahat yavaşlama: `2.0 m/s²`

Araç yaklaşık iki metre boşlukta tamamen durursa `HOLD` durumuna geçer ve
eğimde kaymamak için freni tutar. Ön araç hareket ettiğinde iki yeni ölçümden
sonra yeniden kalkar. Gaz ile fren aynı çevrimde birlikte verilmez.

### Acil fren

Acil fren normal araç takibinden bağımsızdır. Takip edilen araç ile sürüş
koridorundaki en yakın ham radar tehlikesini karşılaştırır. Tek bir gürültülü
radar noktası aracı durdurmasın diye, fiziksel olarak çok yakın değilse ikinci
yeni radar karesinde de aynı tehlikenin görülmesi gerekir.

## Temel `.env` ayarları

```dotenv
HOST=127.0.0.1
PORT=2000
EGO_ROLE_NAME=ego_vehicle

VEHICLE_DEVICE=auto
VEHICLE_CONFIDENCE=0.05
PERCEPTION_EVERY_N_FRAMES=1

ENABLE_SIGN_DETECTION=false
ENABLE_DATA_RECORDING=false

STATUS_PERIOD_SECONDS=2.0
MAX_RUNTIME_SECONDS=0
MAXIMUM_SPEED_KMH=60
```

- `VEHICLE_DEVICE=auto`: CUDA varsa ekran kartını, yoksa CPU'yu seçer.
- `PERCEPTION_EVERY_N_FRAMES=1`: her kamera karesini algılamaya gönderir.
- `ENABLE_SIGN_DETECTION=false`: isteğe bağlı levha modellerini kapalı tutar.
- `ENABLE_DATA_RECORDING=false`: yalnızca kontrol için gereken iki sensörü açar.
- `MAX_RUNTIME_SECONDS=0`: kullanıcı kapatana kadar çalışır.
- `MAXIMUM_SPEED_KMH=60`: düz ve boş yoldaki üst hız hedefidir.

## Terminal durum satırı

Uygulama belirlenen aralıkta bir `[STATUS]` satırı yazar:

| Alan | Anlamı |
|---|---|
| `speed` | Ego aracının mevcut hızı |
| `target` | Viraj ve şerit durumundan sonra seçilen hedef hız |
| `mode` | `CRUISE`, `LEAD_FAR`, `FOLLOW`, `HOLD`, `RESTART` veya `EMERGENCY` |
| `steer` | Uygulanan direksiyon değeri |
| `throttle`, `brake` | Uygulanan gaz ve fren |
| `bbox` | Son algılama sonucundaki araç kutusu sayısı |
| `age` | Algılama sonucunun dünya karesine göre yaşı |
| `radar=ham/kullanılabilir` | Gelen ve zemin elemesinden geçen radar noktaları |
| `ground` | Zemin olarak elenen radar noktası sayısı |
| `cte` | Şerit merkezine olan yanal hata |
| `heading` | Araç yönü ile rota yönü arasındaki fark |
| `speed_reason` | Hedef hızın düz yol, viraj veya şerit toparlama nedeni |
| `lead` | Ön araç seçicinin ölçtüğü mesafe |
| `ctrl_gap` | Gaz-fren kontrolünde kullanılan filtrelenmiş mesafe |
| `aeb` | Acil fren nedeni ve tehlike bilgisi |

## Veri kaydı

Tam sensör paketi gerektiğinde `.env` içinde:

```dotenv
ENABLE_DATA_RECORDING=true
```

Bu mod 15 sensörü de açar ve daha fazla işlem gücü kullanır. Görüntü, LiDAR,
radar, GNSS, IMU ve kalibrasyon dosyaları `data/runs/` altında saklanır.

## Doğrulama

Kod değişikliğinden sonra iki komut birlikte çalıştırılmalıdır:

```bash
python -m compileall -q carla_app main.py scripts tests
python -m unittest discover -s tests -v
```

Testler; direksiyon yönünü, direksiyon değişim sınırını, viraj hızını, şerit
toparlamayı, ön araç takibini, iki metre duruşu, yeniden kalkışı, kamera-radar
birleşimini, komşu şerit ve zemin reddini, eski sensör karesini ve acil freni
kapsar.

Kontrol denklemlerinin temel kaynakları:

- [Intelligent Driver Model](https://mtreiber.de/publications/micro_tgf99.pdf)
- [Stanley yanal kontrol](https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf)
- [NHTSA otomatik acil fren standardı](https://www.federalregister.gov/documents/2024/05/09/2024-09054/federal-motor-vehicle-safety-standards-automatic-emergency-braking-systems-for-light-vehicles)
