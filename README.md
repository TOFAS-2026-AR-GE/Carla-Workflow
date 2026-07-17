# CARLA Araç Takibi ve Kontrolü

Bu branch, CARLA içindeki ego aracını şerit merkezinde sürer, önündeki aracı
kamera ve radar ile takip eder, viraja göre hızını ayarlar ve kritik çarpışma
riskinde acil fren uygular.

Kodun ana ilkesi şudur: her dosya tek bir iş yapar ve ana araç döngüsü bu
parçaları açık bir sırayla çağırır.

Canlı uygulama kodunda `dataclass`, `property`, `staticmethod`, `classmethod`,
`lambda` ve iç içe liste üreteçleri kullanılmaz. Sınıflar yalnızca ilişkili
durumu tutmak için, fonksiyonlar ise tek bir açık işi yapmak için kullanılır.
Tekrarlanan işlemler görünür `for` döngüleriyle yazılmıştır.

## Sistem akışı

Her CARLA karesinde aşağıdaki sıra izlenir:

1. `application.py` dünya karesini ilerletir ve araç durumunu okur.
2. `sensors/manager.py` seçilen moda göre gerekli sensörlerin en güncel
   verisini verir.
3. `perception/system.py` kontrol modunda ön kamerayı, BEV modunda yedi
   kamerayı aynı araç modeliyle işler.
4. `controller/vehicle/lead_vehicle.py` kamera ile radarı birleştirip takip
   edilecek ön aracı seçer.
5. `controller/vehicle/vehicle_controller.py` direksiyon, hedef hız, gaz ve
   fren değerlerini hesaplar.
6. `application.py` hesaplanan komutu ego aracına uygular.
7. BEV modu açıksa `bev/` sensör verilerini yalnızca görselleştirme için ego
   koordinatına taşır. Bu sonuç kontrol komutunu değiştirmez.

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
| `carla_app/bev/` | Kalibrasyonlu IPM, sensör füzyonu, takip ve occupancy grid üretir. |
| `carla_app/controller/vehicle/` | Ön araç seçimi, direksiyon, hız, gaz-fren ve acil frendir. |
| `carla_app/visualization/viewer.py` | Kamera kutularını, BEV açıksa iki panelli OpenCV penceresini gösterir. |
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

## Sensör modları

Çalışma biçimi `.env` içindeki tek bir `SENSOR_MODE` ayarıyla seçilir:

| Mod | Açılan sensör | Davranış |
|---|---:|---|
| `control` | 2 | Yalnızca `camera_front_wide` ve `radar_front_long` açılır. Mevcut kontrol akışı değişmeden çalışır. Varsayılan mod budur. |
| `bev` | 15 | Bütün sensörler açılır, yedi kamera tek YOLO modeliyle toplu işlenir ve OpenCV penceresi kamera + BEV olarak ikiye ayrılır. Diske kayıt yapılmaz. |
| `record` | 15 | Bütün sensörlerin aynı karedeki paketi beklenir ve `data/runs/` altına kaydedilir. BEV açılmaz. |

Toplam 15 gerçek CARLA sensörünün dağılımı:

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

## İsteğe bağlı BEV modülü

BEV bu sürümde projeye entegredir fakat varsayılan olarak kapalıdır. Normal
çalıştırmada `.env` içindeki değer şu şekilde kalmalıdır:

```dotenv
SENSOR_MODE=control
```

Daha sonra denemek istediğinde yalnızca şu değeri değiştirmen yeterlidir:

```dotenv
SENSOR_MODE=bev
```

BEV açıkken OpenCV penceresinin sol yarısında ön kamera ve bbox'lar, sağ
yarısında ise ego merkezli gerçek IPM kuş bakışı görünüm bulunur. Sağ panelde:

- yedi kameranın zemin görüntüleri tek kuş bakışı dokuda birleşir,
- LiDAR nesne noktaları gri, beş radarın noktaları pembe çizilir,
- dolu occupancy hücreleri kırmızı, boş hücreler yeşil gösterilir,
- yeni nesne takipleri sarı, doğrulanmış takipler yeşil kutuyla çizilir,
- referans rota turuncu gösterilir.

Koordinat sistemi metre cinsindedir: `X` aracın ilerisi, `Y` aracın sağıdır.
Kamera çözünürlüğü ve yatay FOV değerinden `K` iç kalibrasyon matrisi;
sensörün araç üzerindeki konum ve açısından `R` ile `T` dış kalibrasyon
matrisleri hesaplanır. CARLA'nın `x-ileri, y-sağ, z-yukarı` eksenleri OpenCV
kamera eksenlerine açık bir matrisle çevrilir. Zemin homografisinin tersi her
BEV hücresinin kaynak kamera pikselini bulur; bu işlem IPM'dir.

Kameraların örtüşen alanlarında görüntü merkezine, görüş açısına ve zemin
görünürlüğüne göre ağırlık verilir. Böylece aynı bölgeyi gören iki kameranın
arasında sert bir kesik yerine yumuşak geçiş oluşur. IPM düz zemin varsayar;
araç tavanı ve bina gibi dikey yüzeyler görüntüde uzayabilir. Nesne konumu bu
görsel bozulmaya bırakılmaz: bbox zemin noktası, radar ve LiDAR ölçümleri
ayrıca ortak ego koordinatında birleştirilir.

### BEV işleme sırası

| Dosya | Açık görevi |
|---|---|
| `calibration.py` | Her sensörün `K`, `R`, `T` ve zemin homografisini hesaplar. |
| `coordinate.py` | Metre-piksel dönüşümünü ve eski ölçümlerin ego hareket telafisini yapar. |
| `camera_ipm.py` | Yedi RGB kamerayı kuş bakışına çevirir ve örtüşmeleri ağırlıklı birleştirir. |
| `projector.py` | Kamera, radar, LiDAR ve rotayı ortak ego koordinatına taşır. |
| `clustering.py` | Radar ve LiDAR noktalarını komşulukla kümeler. |
| `association.py` | Aynı fiziksel nesneye ait farklı sensör ölçümlerini eşleştirir. |
| `fusion.py` | Ölçüm belirsizliğinin ters varyansıyla ortak nesne konumu üretir. |
| `tracking.py` | Nesneleri dünya koordinatında Kalman filtresiyle izler. |
| `occupancy.py` | Işınların boş ve dolu hücrelerini log-odds occupancy grid'de biriktirir. |
| `renderer.py` | IPM, occupancy, sensör noktaları, rota ve takipleri çizer. |
| `module.py` | Bu katmanları tek ve çıkarılabilir dış arayüzde toplar. |

Farklı kameralarda görülen aynı araç, sensör kimliği ve uzamsal kapı kontrolü
ile tek gruba alınır. Aynı kameradaki iki ayrı bbox birbirine yakın olsa bile
birleştirilmez. Kamera, radar ve LiDAR konumları ölçüm belirsizliğinin ters
varyansıyla ağırlıklandırılır. Kalman takibi dünya koordinatında tutulduğu için
ego araç hareket ettiğinde sabit nesneler BEV üzerinde yapay olarak kaymaz.

Occupancy grid LiDAR ve radar ışınlarının geçtiği hücreleri boş, vurduğu
yüksek noktaları dolu kabul eder. Geçmiş grid yeni ego pozuna taşınır ve zamanla
zayıflatılır. Bu yapı klasik geometrik sensör füzyonlu BEV'dir. Yol, kaldırım
ve yaya gibi her piksele sınıf veren öğrenilmiş semantic BEV değildir; bunun
için ayrıca segmentasyon modeli gerekir.

BEV yalnızca inceleme ekranıdır; araç kontrolü veya acil fren için kullanılmaz.
Kontrolcü BEV modunda da yalnızca ön geniş kamera ve ön uzun radarı kullanır.
GNSS ve IMU 15 sensörün içinde çalışır fakat ego merkezli çizimde nokta
üretmez; ileride harita konumlandırması için canlı pakette tutulur.

Sensörlerden biri birkaç kare gecikirse araç döngüsü bekletilmez. BEV her
sensörün en yeni geçerli verisini kullanır ve eski veriyi kendiliğinden atar.
IPM, füzyon, takip ve occupancy ayrı bir iş parçacığında çalışır. Kuyrukta
yalnızca en yeni iş tutulur. Bu sayede yavaş bir çevre kamerası veya ağır bir
BEV karesi direksiyon ve fren akışını durdurmaz.

Sensörlerin yatay açıları ön, sağ, arka ve sol yönlerde 10 metre test
noktalarıyla doğrulanmıştır. Çevre kameraları bu dört yönü boşluk bırakmadan
ve örtüşmeli görmektedir. Çalışan kontrol davranışını değiştirmemek için ana ön
kamera ve radar yerleşimi korunmuştur.

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

## Kullanılan algoritmalar

Aşağıdaki tablo algoritmanın adını, genel tanımını ve bu projedeki gerçek
uygulamasını birlikte gösterir.

| Algoritma | Basit tanımı | Bu projede nasıl uygulandı? |
|---|---|---|
| Kalıcı referans rota | Araç anlık olarak çizgiden ayrılsa bile takip edilen yolun değişmemesini sağlar. | `route_manager.py` aracın önünde yaklaşık 80 metrelik yol noktası tutar. Geçilen noktaları siler, rotayı ileri uzatır ve yalnızca araç 20 çevrim boyunca rotadan 8 metreden fazla uzak kalırsa rotayı yeniden kurar. |
| YOLO araç tespiti | Kamera görüntüsünde araçların bulunduğu dikdörtgen alanları bulur. | `vehicle_detector.py` ön RGB görüntüsünü YOLO modeline verir. Yalnızca araç sınıflarını kabul eder ve her sonuç için sınıf, güven ve görüntü kutusu üretir. |
| Açısal kamera-radar birleştirme | Kamera kutusunun görüş açısı ile aynı açıdaki radar noktalarını eşleştirir. | `fusion.py` kutunun sol ve sağ kenarını dereceye çevirir. Bu açı aralığındaki radar noktalarını seçer; kamera verisi birkaç kare eskiyse küçük bir açı payı ekler. |
| Uyarlamalı radar kümeleme | Birbirine yakın radar noktalarını aynı fiziksel hedef altında toplar. | `fusion.py` aynı kamera kutusundaki noktaları derinliğe göre, `lead_vehicle.py` ise önden gelen noktaları iki boyutlu uzaklığa göre kümeler. Uzak hedeflerde izin verilen küme aralığı kontrollü biçimde büyür. |
| Sabit hızlı Kalman filtresi | Gürültülü konum ölçümünü hareket tahminiyle birleştirip konum ve hız üretir. | `tracking.py` X ve Y eksenlerini ayrı ayrı izler. Her çevrim önce son hıza göre tahmin yapar, sonra yeni kamera-radar ölçümüyle tahmini düzeltir. |
| En yakın komşu eşleştirme | Yeni ölçümü kendisine en yakın mevcut araç takibiyle eşleştirir. | `tracking.py` ölçüm ile takip arasındaki bütün uygun uzaklıkları hesaplar. En yakın çiftten başlar; 5 metreden uzak çiftleri kabul etmez ve uzun süre görülmeyen takibi siler. |
| Zamansal doğrulama ve histerezis | Tek bir gürültülü ölçümle hedef değiştirmeyi önler. | `lead_vehicle.py` doğrudan radar hedefini normal takibe vermeden önce iki yeni karede görür. Benzer mesafedeki iki araç arasında geçiş yapmak için yeni aracın en az 2 metre daha avantajlı olmasını ister. |
| Stanley direksiyon kontrolü | Araç yönü ile şerit merkezine olan yanal hatayı tek direksiyon komutunda birleştirir. | `stanley_controller.py` aracın önündeki kontrol noktasını rotaya izdüşürür. Başlık hatası, yanal hata ve küçük bir viraj ileri beslemesi kullanır; direksiyon büyüklüğünü ve değişim hızını araç hızına göre sınırlar. |
| Eğrilik tabanlı hız planlama | Viraj keskinleştikçe izin verilen hızı rahat yanal ivmeye göre düşürür. | `speed_planner.py` hız yükseldikçe 35-75 metre ileriyi tarar. Yol eğriliğinden `hız = karekök(yanal ivme / eğrilik)` hesabını yapar. Şerit hatası büyürse ayrıca toparlanma hızı ister. |
| IDM araç takip kontrolü | Hedef hıza giderken ön araçla hıza bağlı güvenli zaman ve mesafe bırakır. | `longitudinal_controller.py` 2 metre duruş boşluğu, 1,2 saniye zaman aralığı, ego hızı ve yaklaşma hızından istenen ivmeyi hesaplar. Sonuç pozitifse gaz, yeterince negatifse fren üretir. |
| İvme değişim sınırı | Gaz veya fren isteğinin bir çevrimde aniden sıçramasını önler. | `longitudinal_controller.py` istenen ivme değişimini hızlanmada 3, frenlemede 6 m/s³ ile sınırlar. Gaz ve fren aynı çevrimde birlikte verilmez. |
| Üstel ölçüm yumuşatma | Yeni ölçüm ile önceki değeri belirli oranlarda birleştirir. | `lead_vehicle.py` radar mesafesi ve bağıl hızını, `longitudinal_controller.py` ise kontrolcünün kullandığı ön araç mesafesini yumuşatır. Yakınlaşan ölçüm güvenlik için uzaklaşan ölçümden daha hızlı kabul edilir. |
| TTC ve gerekli yavaşlama ile acil fren | Mesafe kapanma süresini ve çarpışmayı önlemek için gereken yavaşlamayı hesaplar. | `safety_supervisor.py` takip hedefi ile ham radar tehlikesinden daha riskli olanı seçer. Kritik olmayan tek radar noktasını yeterli saymaz; aynı tehlikeyi ikinci yeni karede de görünce tam fren uygular. |

Bu algoritmaların çağrılma sırası `vehicle_controller.py` içinde açıktır:
önce direksiyon, sonra viraj hedef hızı, ardından normal gaz-fren ve en son
bağımsız acil fren denetimi çalışır. Acil fren kararı çıkarsa normal gaz
silinir ve fren doğrudan `1.0` yapılır.

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
SENSOR_MODE=control
BEV_UPDATE_EVERY_N_FRAMES=2

STATUS_PERIOD_SECONDS=2.0
MAX_RUNTIME_SECONDS=0
MAXIMUM_SPEED_KMH=60
```

- `VEHICLE_DEVICE=auto`: CUDA varsa ekran kartını, yoksa CPU'yu seçer.
- `PERCEPTION_EVERY_N_FRAMES=1`: her kamera karesini algılamaya gönderir.
- `ENABLE_SIGN_DETECTION=false`: isteğe bağlı levha modellerini kapalı tutar.
- `SENSOR_MODE=control`: yalnızca kontrol için gereken iki sensörü açar.
- `BEV_UPDATE_EVERY_N_FRAMES=2`: BEV açıkken 20 Hz simülasyonda en fazla
  10 Hz kuş bakışı üretir; kontrol döngüsünün frekansını değiştirmez.
- `ENABLE_DATA_RECORDING=false`: eski kurulumlarla uyumluluk için tutulur.
  `true` yapılırsa `SENSOR_MODE` değerinden bağımsız olarak kayıt modu seçilir.
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
SENSOR_MODE=record
```

Bu mod 15 sensörü de açar ve daha fazla işlem gücü kullanır. Görüntü, LiDAR,
radar, GNSS, IMU ve kalibrasyon dosyaları `data/runs/` altında saklanır.
Eski `ENABLE_DATA_RECORDING=true` ayarı da aynı kayıt modunu seçmeye devam
eder.

## Doğrulama

Kod değişikliğinden sonra iki komut birlikte çalıştırılmalıdır:

```bash
python -m compileall -q carla_app main.py scripts tests
python -m unittest discover -s tests -v
```

Testler; direksiyon yönünü, direksiyon değişim sınırını, viraj hızını, şerit
toparlamayı, ön araç takibini, iki metre duruşu, yeniden kalkışı, kamera-radar
birleşimini, komşu şerit ve zemin reddini, eski sensör karesini, acil freni,
`K-R-T` kalibrasyonunu, homografi dönüşümünü, 360 derece kamera kapsamasını,
sensör tekrar silmeyi, güven ağırlıklı füzyonu, ego hareket telafisini, Kalman
takibini, occupancy grid'i, çoklu kamera akışını ve sensör modu seçimini kapsar.

Kontrol denklemlerinin temel kaynakları:

- [Intelligent Driver Model](https://mtreiber.de/publications/micro_tgf99.pdf)
- [Stanley yanal kontrol](https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf)
- [NHTSA otomatik acil fren standardı](https://www.federalregister.gov/documents/2024/05/09/2024-09054/federal-motor-vehicle-safety-standards-automatic-emergency-braking-systems-for-light-vehicles)
