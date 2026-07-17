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

## RViz sensör yerleşimi

`scripts/rviz_sensor_layout.py`, araç boyutundan hesaplanan 27 sensörün
konumunu, yönünü ve yatay görüş alanını RViz'de gösterir. Kontrol için gerçekten
çalışan sensörler dolu, yalnızca tasarımda bulunan sensörler saydam ve `[plan]`
etiketli çizilir. Bu nedenle yerleşimi görmek için
`ENABLE_DATA_RECORDING=true` yapıp bütün sensörleri çalıştırmak gerekmez.
Araç gövdesi, CARLA'dan okunan gerçek bounding-box ölçüsüyle saydam gri kutu
olarak çizilir.

Renkler:

- Yeşil: kamera
- Kırmızı: radar
- Mavi: LiDAR
- Sarı: ultrasonik
- Mor: GNSS
- Camgöbeği: IMU

Bu görünüm için CARLA ROS Bridge zorunlu değildir. CARLA sunucusunu açtıktan
sonra ilk terminalde normal uygulamayı çalıştır:

```bash
cd Carla-Workflow
bash run.sh
```

İkinci terminalde sensör marker ve TF node'unu çalıştır:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd Carla-Workflow
python scripts/rviz_sensor_layout.py
```

Üçüncü terminalde hazır RViz görünümünü aç:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd Carla-Workflow
rviz2 -d config/sensor_layout.rviz
```

Node şu TF ağacını kendi yayımlar:

```text
map -> ego_vehicle -> ego_vehicle/layout/<sensor_adi>
```

Bu nedenle sensörlerin araç üzerindeki konumu için ayrıca ROS Bridge veya
`sensor.pseudo.tf` gerekmez.

Marker konusu:

```text
/carla/ego_vehicle/sensor_layout
```

Yalnızca gerçekten spawn edilmiş sensörleri göstermek için:

```bash
python scripts/rviz_sensor_layout.py --ros-args \
  -p show_inactive:=false
```

Uzak CARLA sunucusunda host ve port parametreleri verilebilir:

```bash
python scripts/rviz_sensor_layout.py --ros-args \
  -p host:=192.168.1.50 -p port:=2000
```

Gerçek kamera görüntüsü veya radar `PointCloud2` verisini de RViz'e almak
istersen ROS Bridge'i ayrıca açabilirsin. Uygulama synchronous dünyanın tick'ini
verdiği için bridge mutlaka pasif olmalıdır:

```bash
source ~/carla-ros-bridge/install/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py \
  passive:=True synchronous_mode:=True register_all_sensors:=True
```

Bridge ile gerçek sensör verisini açarken marker node'u ve RViz'i
`--ros-args -p use_sim_time:=true` ile başlat. Ardından RViz Displays panelinde
varsayılan olarak kapalı gelen `Front Radar Points` katmanını açabilirsin.

ROS Bridge ego aracı `ego_vehicle` rolüyle bulur. Bu repo aracı varsayılan
olarak bu rolle oluşturur; gerekirse `.env` içindeki `EGO_ROLE_NAME` ile
değiştirilebilir. Resmî kaynaklar:
[CARLA ROS Bridge](https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/)
ve [RViz eklentisi](https://carla.readthedocs.io/projects/ros-bridge/en/latest/rviz_plugin/).

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
EGO_ROLE_NAME=ego_vehicle

ENABLE_SIGN_DETECTION=false
ENABLE_DATA_RECORDING=false

STATUS_PERIOD_SECONDS=2.0
MAX_RUNTIME_SECONDS=0
MAXIMUM_SPEED_KMH=60
```

- `VEHICLE_DEVICE=auto`: CUDA varsa `0`, yoksa `cpu` seçer.
- `VEHICLE_CONFIDENCE=0.05`: repodaki özel CARLA modelinin düşük skorlu
  geçerli araç kutularını korur; yalnızca modelin araç sınıfları çizilir.
- Levha tespiti kapalıdır; açılması araç bbox sonucunu artık etkileyemez.
- Dataset kaydı kapalıyken sadece kontrol için gereken iki sensör çalışır.
- `MAX_RUNTIME_SECONDS=0` sınırsız interaktif kullanım demektir. Otomatik test
  için örneğin `30` verilebilir.
- `MAXIMUM_SPEED_KMH=60`: düz ve boş yoldaki şehir içi üst hız hedefidir.
  Viraj, şerit toparlama, öndeki araç ve AEB bu değeri güvenlik için düşürebilir.

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
- Normal hedef kaynağı ön kamera ve radarın birlikte oluşturduğu
  `camera_radar_track` kaynağıdır. Kamera aracın bbox'ını ve kimliğini, radar
  ise gerçek mesafeyi ve bağıl hızı verir. `radar_direct`, kamera geçici olarak
  hedefi kaçırdığında kullanılan iki tick doğrulamalı yedek kaynaktır.
- Ön uzun menzil radarı kaput seviyesinde ve hafif yukarı bakar. Her dönüşün
  tahmini yerden yüksekliği hesaplanır; yol yüzeyine çarpan ışınlar kamera
  füzyonu, normal takip ve AEB'den önce tek noktada elenir.
- Radar bağıl hızında negatif değer yaklaşmayı, pozitif değer uzaklaşmayı gösterir.
- Uzak ve yaklaşmayan araç yalnızca gözlemlenir (`LEAD_FAR`). Yaklaşma hızı
  yüksekse IDM uzak mesafede de erken ve yumuşak biçimde yavaşlayabilir.
- Yalnız radar ile görülen engel normalde iki ardışık tick doğrulanmadan
  ACC'ye verilmez.
  Çok yakın veya TTC'si kritik engelde güvenlik için beklenmez.
- Normal takip ve AEB adayları, şerit genişliğiyle birlikte gerçek araç
  genişliğinden hesaplanan sürüş koridorundan geçer. Kaldırım, yol kenarı ve
  komşu şeritte kalan radar noktaları takip aracı olarak seçilmez.
- CARLA GPU kamera callback'i birkaç tick gecikse bile gerçek kamera frame'i
  inference'a verilir; dünya frame'iyle tam eşleşmeyen görüntü artık atılmaz.
- Ham ön radar, bbox veya tracker oluşmasa bile kısa menzilde bağımsız AEB
  girdisi üretir.
- Durum satırında `radar=ham/kullanılabilir` gösterilir. `ground=N`, o tick'te
  zemin olarak elenen radar dönüşü sayısıdır.
- Durum satırındaki `ctrl_gap`, filtrelenip boylamsal kontrolcüde kullanılan
  gerçek takip mesafesidir; `lead` ise takip katmanının ham seçimini gösterir.
- Stanley kontrolcüsü ön kontrol noktasının rota başlık ve yanal hatasını
  kullanır. Araç gövdesi şerit kenarına yaklaşınca merkezleme düzeltmesi artar;
  küçük viraj önbeslemesi dönüşe daha erken başlamaya yardım eder.
- Viraj hedef hızı `v = sqrt(a_y / eğrilik)` bağıntısıyla belirlenir. Rota
  hatası büyüdüğünde araç kontrollü toparlanabilmek için ayrıca yavaşlar. Hıza
  göre yaklaşık üç saniyelik yol taranır. Güvenlik amaçlı hız düşüşü gecikmeden
  uygulanır; yalnızca tekrar hızlanma yumuşak biçimde sınırlandırılır.
- AEB normal takipten bağımsızdır. En tehlikeli kamera/radar adayının TTC'sini
  ve 2 metre boşlukta durmak için gereken yavaşlamayı izler; kritik durumda
  normal kontrolü geçersiz kılıp tam fren uygular.

## Kodun sade kontrol akışı

1. `application.py`, kamera ve radar verisini aynı simülasyon tick'inde toplar.
2. `lead_vehicle.py`, zemini ve yan engelleri eler; kamera bbox'ı ile radar
   mesafesini birleştirerek takip edilecek aracı seçer.
3. `stanley_controller.py`, rota merkezinde kalacak direksiyonu hesaplar.
4. `speed_planner.py`, düz yol, yaklaşan viraj ve şerit hatasına göre hedef
   hızı hesaplar.
5. `longitudinal_controller.py`, IDM ile gaz veya fren değerini üretir.
6. `safety_supervisor.py`, kritik TTC durumunda diğer bütün istekleri geçersiz
   kılıp tam fren uygular.

Kontrol akışındaki yardımcı metotların adları doğrudan yaptıkları işi anlatır.
Örneğin `filter_ground_returns`, `inside_driving_corridor`,
`calculate_idm_gap` ve `limit_speed_increase` isimleri kullanılmaktadır.

Kontrol denklemlerinin temel kaynakları:

- [Intelligent Driver Model](https://mtreiber.de/publications/micro_tgf99.pdf)
- [Stanley lateral control](https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf)
- [NHTSA Automatic Emergency Braking standard](https://www.federalregister.gov/documents/2024/05/09/2024-09054/federal-motor-vehicle-safety-standards-automatic-emergency-braking-systems-for-light-vehicles)

## Testler

```bash
python -m unittest discover -s tests -v
python -m compileall -q carla_app main.py scripts tests
```

Testler; Stanley yönünü ve direksiyon değişim limitini, viraj hızını, güvenli
hız düşüşünün gecikmemesini, gönderilen düşük hız
logunun tekrarını, gürültülü ölçümde tek seferde 2 metreye duruşu, hızlı
kalkışı, IDM takibini, acil freni, radar doğrulamasını, komşu şerit reddini,
kaldırım kenarı reddini, kamera-radar füzyonunu ve levha hatasının araç
bbox'ını bozmamasını kapsar.
