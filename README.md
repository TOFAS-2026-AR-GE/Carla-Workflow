# CARLA Trafik Kuralları, Araç Takibi ve Güvenli Kontrol

Bu branch, CARLA içindeki ego aracını şerit merkezinde sürer; kamera, radar ve
LiDAR ile çevreyi izler; önündeki aracı takip eder; trafik ışığı, hız tabelası,
yaya ve viraj risklerini tek bir güvenli hedef hız kararında birleştirir.

Kodun ana ilkesi şudur: her dosya tek bir iş yapar ve ana araç döngüsü bu
parçaları açık bir sırayla çağırır.

Projeye ilk kez bakıyorsanız önce kısa [mimari haritayı](ARCHITECTURE.md) okuyun.
Bu README kurulum, parametreler ve algoritmalar için ayrıntılı başvuru belgesidir.
Kodda sınıflar ilişkili durumu, fonksiyonlar ise sınırlı ve açık işleri tutar.

## Sistem akışı

Her CARLA karesinde aşağıdaki sıra izlenir:

1. `application.py` dünya karesini ilerletir ve araç durumunu okur.
2. `sensors/manager.py` her modda 15 sensörün en güncel verisini verir.
3. `perception/system.py` YOLO nesnelerini işler; UFLD açıksa yalnız ön
   kamerada şeritleri görselleştirme/doğrulama amacıyla çıkarır.
4. `perception/road_context.py` ham kutuları standart biçime çevirir, ardışık
   karelerde izler ve kamera mesafesini LiDAR ile doğrular.
5. `controller/vehicle/lead_vehicle.py` kamera ile radarı birleştirip takip
   edilecek ön aracı seçer.
6. `controller/vehicle/behavior_planner.py` trafik kurallarını, yayayı, hız
   sınırını ve viraj hızını tek davranış kararında birleştirir.
7. `controller/vehicle/vehicle_controller.py` direksiyon, hedef hız, gaz ve
   fren değerlerini hesaplar.
8. Bağımsız güvenlik denetleyicisi kritik riski ve kontrol komutu sınırlarını
   son kez denetler; `application.py` komutu ego aracına uygular.
9. `bev/` her modda sensör verilerini ego koordinatına taşır; GNSS/IMU
   canlılığını doğrular ve ana lead kaybolduğunda yalnız taze, çok-sensörlü,
   occupancy destekli bir track ile güvenli lead recovery uygular. Mevcut ana
   lead'i hiçbir zaman silmez.
10. Navigasyon paneli CARLA'nın açık olan haritasını otomatik çizer; onaylanan
    hedefe en kısa sürüş rotasını üretir ve varışta aracı durdurur.

## Klasör ve dosya rehberi

| Konum | Görevi |
|---|---|
| `main.py` | Uygulamayı başlatan tek giriş noktasıdır. |
| `carla_app/application.py` | Açılış, ana döngü, durum çıktısı ve kapanışı yönetir. |
| `carla_app/config.py` | `.env` ayarlarını tek yerde okur ve doğrular. |
| `carla_app/core/` | CARLA bağlantısı, ego araç, trafik, rota ve araç durumunu yönetir. |
| `carla_app/navigation/` | Harita hedefi, en kısa rota, kalan mesafe ve varış durumunu yönetir. |
| `carla_app/sensors/layout.py` | Sensör adlarını, konumlarını ve CARLA ayarlarını tanımlar. |
| `carla_app/sensors/factory.py` | Tanımlanan sensörleri CARLA üzerinde oluşturur. |
| `carla_app/sensors/manager.py` | Sensörleri açar, okur, kaydeder ve kapatır. |
| `carla_app/sensors/stream.py` | Canlı ön kamera ve ön radar verisini güvenli biçimde saklar. |
| `carla_app/sensors/sync.py` | Veri kaydında bütün sensörlerin aynı karesini bekler. |
| `carla_app/sensors/processors.py` | CARLA sensör verisini NumPy ve sözlük biçimine çevirir. |
| `carla_app/sensors/writer.py` | Tam sensör paketini `data/runs/` altına kaydeder. |
| `carla_app/perception/` | YOLO nesneleri, UFLD şeritleri, zamansal takip ve kamera-LiDAR doğrulamasıdır. |
| `carla_app/bev/` | Kalibrasyonlu IPM, sensör füzyonu, takip ve occupancy grid üretir. |
| `carla_app/controller/vehicle/` | Ön araç seçimi, direksiyon, hız, gaz-fren ve acil frendir. |
| `carla_app/visualization/viewer.py` | Kamera, UFLD, BEV ve navigasyonu tek OpenCV penceresinde birleştirir. |
| `carla_app/visualization/map_renderer.py` | CARLA yol ağını, rotayı, hedefi ve canlı ego konumunu çizer. |
| `carla_app/visualization/navigation_panel.py` | Navigasyon çizim önbelleğini ve sol tık seçim/onay/iptal girişlerini yönetir. |
| `carla_app/visualization/sensor_layout.py` | Sensör yerleşimini tarayıcı verisine dönüştürür. |
| `carla_app/visualization/sensor_layout.html` | Araba şeklindeki sensör ekranıdır. |
| `scripts/` | Kurulum kontrolü, model kopyalama ve sensör ekranı komutlarıdır. |
| `scenarios/traffic.yaml` | Harita, zaman adımı ve trafik ayarlarıdır. |
| `tests/` | Kontrol, algılama, rota ve sensör davranış testleridir. |

## Kurulum

Proje Python 3.12 ve `carla` adlı Conda ortamını kullanır. İşletim sistemine
uygun betik ortam yoksa oluşturur, bağımlılıkları kurar, RTX 50 serisi için
CUDA 12.8 destekli PyTorch'u doğrular ve kurulumu denetler.
`ENABLE_LANE_DETECTION=true` ise UFLD ağırlığı da sabitlenmiş Hugging Face
revizyonundan indirilip SHA256 ile doğrulanır; doğru dosya zaten varsa yeniden
indirilmez.

Windows (RTX 5070, PowerShell):

```powershell
cd C:\Users\<kullanıcı>\Desktop\Carla-Workflow
.\run_windows.ps1 -SetupOnly
```

PowerShell betik çalıştırmayı engellerse yalnızca o çalıştırma için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -SetupOnly
```

Linux (RTX 5090):

```bash
cd ~/Desktop/Carla-Workflow
bash run_linux.sh --setup-only
```

`check_setup.py` çıktısında gerekli paketler ve araç modeli `OK` görünmelidir.
Araç modeli şu konumdadır:

```text
models/vehicle/carla_yolov8n_best.pt
```

İsteğe bağlı CARLA UFLD şerit modelini elle indirmek veya doğrulamak için:

```powershell
conda run -n carla python scripts/download_lane_model.py
```

Linux'ta aktif Conda ortamında aynı komut `python scripts/download_lane_model.py`
olarak çalıştırılır. Ağırlık yaklaşık 735 MB'dir ve Git deposuna eklenmez.

Kontrol çıktısında `OK CUDA` ve ekran kartı adı görünmelidir. CPU-only PyTorch
500–800 ms algılama gecikmesine yol açabilir. RTX 5070 üzerinde birleşik model
640 piksel sıcak inference ölçümünde tek kamerada yaklaşık 33 ms, yedi kameralı
BEV paketinde tüm algı zinciriyle yaklaşık 85 ms çalışmıştır. Bu nedenle
varsayılan `control` ve `record` modları model çıkarımını yalnız ön kamerada
yapar; yedi-kamera çıkarımı yalnız `bev` modunda açılır. Canlı hedef uçtan uca
algılama bütçesi `PERCEPTION_LATENCY_BUDGET_MS=80` değeridir.

## Normal çalıştırma

Önce kullandığın sürüme uygun CARLA sunucusunu aç. Ardından Windows'ta:

```powershell
.\run_windows.ps1
```

Linux'ta:

```bash
bash run_linux.sh
```

İlk kurulumdan sonraki açılışlarda paket kontrolünü atlamak için Windows'ta
`-SkipInstall`, Linux'ta `--skip-install` kullanılabilir. CUDA bulunamazsa betik
varsayılan olarak durur; yalnızca bilinçli tanılama için `-AllowCpu` veya
`--allow-cpu` seçeneği vardır.

Tam sensör BEV görünümü:

```powershell
.\run_windows.ps1 -SensorMode bev
```

```bash
bash run_linux.sh --bev
```

OpenCV penceresinde `Q`, `ESC` veya pencerenin kapatma düğmesi uygulamayı
güvenli biçimde sonlandırır.

## Canlı navigasyon kullanımı

Uygulama açıldığında ego araç bulunduğu yerde frenle bekler. Açık olan CARLA
haritası (`Town03`, `Town05` gibi) ayrıca seçilmez; panel o anki
`world.get_map()` verisinden otomatik üretilir.

1. Navigasyon panelindeki bir yolun yakınına farenin **sol tuşuyla** tıklayın.
2. Sarı `SECILI` işaretinin doğru sürüş şeridine oturduğunu kontrol edin.
3. `ROTAYI ONAYLA` düğmesine sol tıklayın.
4. Araç kırmızı rotayı izler; sarı ego işareti canlı konumla birlikte hareket
   eder ve kalan mesafe panelde güncellenir.
5. Araç hedefe yaklaşırken rahat fren mesafesine göre yavaşlar ve hedefte
   durur.

Onaylamadan önce `IPTAL` ile seçim kaldırılabilir. Sürüş sırasında başka bir
nokta sol tıkla seçilip onaylanırsa rota aracın güncel konumundan yeniden
hesaplanır.

Pencere boyutu ve navigasyon hızı `.env` üzerinden ayarlanabilir:

```dotenv
DASHBOARD_SIZE=640
NAVIGATION_SPEED_KMH=45
NAVIGATION_ARRIVAL_DISTANCE_M=2.5
NAVIGATION_RENDER_EVERY_N_FRAMES=4
```

UFLD modeli ön kamerada çalışmaya devam eder. Normal sürüş görünümünde şerit
pikseli/eğrisi çizilmez; gerekirse `SHOW_LANE_OVERLAY=true` ile yalnız
mühendislik incelemesi için açılır. Modelin eğitimindeki
1640×590, 150° FOV ve araç-koordinatında `(x=1.5, z=2.4)` ön kamera geometrisi
üretimde de korunur; model girişi 800×288 RGB/ImageNet'tir. 101. no-lane sınıfı
ile satır seçimi yapılır, yatay konum özgün UFLD softmax beklentisiyle çözülür.
Normalize entropi/no-lane marjı tamamen kararsız satırları elerken komşu
hücrelere yayılan geçerli tahmini korur. Dört şerit kimliğinin ham satır
ankrajları güven ağırlıklı ve aykırı-nokta dayanımlı ikinci derece `x(y)`
eğrilerine dönüştürülür. Kamera paneli en-boy oranını bozmadan letterbox ile
ölçeklenir. Varsayılan OpenCV penceresi `640x640` karedir;
`DASHBOARD_SIZE` yalnız ekran boyutunu değiştirir, UFLD eğitim kamerasının
pozunu/FOV'unu değiştirmez.
Ham model noktaları, işlenmiş eğriler veya bunlardan türetilen hiçbir değer
direksiyon, gaz ya da fren hesabına verilmez; aracın referansı navigasyon
waypoint rotası olarak kalır.

## CUDA ve canlı performans

`VEHICLE_DEVICE`, `SIGN_DEVICE` ve `LANE_DEVICE` varsayılan olarak `auto`
değerindedir. CUDA varsa üç model de ekran kartını seçer. CUDA inference için
FP16, cuDNN sabit-boyut optimizasyonu ve TF32 varsayılan olarak açıktır:

```dotenv
VEHICLE_DEVICE=auto
SIGN_DEVICE=auto
LANE_DEVICE=auto
ENABLE_FP16_INFERENCE=true
```

UFLD resize ve ImageNet normalizasyonunu CUDA üzerinde yapar; CPU'dan GPU'ya
büyük bir `float32` tensör yerine sabitlenmiş bellekteki ham kamera görüntüsü
taşınır. YOLO ve isteğe bağlı tabela modelleri FP16 çalışır. Modeller açılışta
canlı kamerayla aynı `1640x590` en-boy oranında iki kez ısıtıldığı için ilk
canlı karede yeni şekil için kernel, backend veya cuDNN benchmark hazırlama
takılması oluşmaz. Geçici CUDA bellek baskısında önce bellek önbelleği
temizlenerek CUDA aynı işlem için bir kez daha denenir; yalnız ikinci hata da
başarısızsa güvenli CPU yedeğine geçilir.

`auto` yalnız CUDA'nın varlığına değil, CARLA açıldıktan sonra gerçekten kalan
VRAM miktarına da bakar. Böylece 4 GB sınıfı ekran kartlarında CARLA aynı GPU'yu
kullanırken model yükleme işlemi uygulamayı kapatmaz. UFLD yükleme veya canlı
inference sırasında sonradan CUDA bellek hatası oluşursa model otomatik olarak
FP32 CPU moduna alınır.

`PERFORMANCE_PROFILE=auto` bilgisayarı açılışta sınıflandırır. 20 GB ve üzeri
VRAM'i boş olan güçlü kartlarda YOLO her karede 640 piksel çalışır. 4 GB
sınıfında 512 piksel ve iki karede bir inference seçilerek CARLA'ya bellek
payı bırakılır. CUDA yoksa daha hafif CPU profili kullanılır. Çalışma sırasında
inference süresi veya atılan kare oranı yükselirse algılama aralığı en fazla
üç kareye çıkar; yük düzelince üç kararlı ölçüm penceresinden sonra tekrar
hızlanır.

Elle sabit değer vermek için:

```dotenv
PERFORMANCE_PROFILE=manual
PERCEPTION_EVERY_N_FRAMES=1
MAXIMUM_PERCEPTION_PERIOD=2
VEHICLE_IMAGE_SIZE=640
CAMERA_WAIT_TIMEOUT_MS=10
```

Düşük VRAM'li bir sistemde iki modeli de baştan CPU'da açmak için:

```bash
VEHICLE_DEVICE=cpu LANE_DEVICE=cpu \
  bash run_linux.sh --skip-install
```

Komut satırından verilen cihaz seçimleri çalıştırma betiği tarafından artık
ezilmez. Büyük RTX kartlarda herhangi bir seçenek vermeden `auto` kullanılmaya
devam edilir.

Durum satırındaki performans alanları:

- `loop`: ana kare işleme süresi,
- `view`: OpenCV panel süresi,
- `infer`: algılama işçisinin model süresi,
- `q`: inference başlamadan önce kuyrukta bekleme,
- `e2e`: kuyruğa girişten model sonucunun tamamlanmasına kadar geçen süre,
- `drop`: daha yeni kare geldiği için bilinçli atılan eski inference işleri,
- `budget_over`: 50 ms simülasyon kare bütçesini aşan kare yüzdesi,
- `latency_over`: yeni algılama sonuçlarının 80 ms hedefini aşma yüzdesi.

`q` ve `drop` sürekli büyüyorsa model hızı kamera hızının gerisindedir.
Öncelikle `VEHICLE_IMAGE_SIZE` düşürülmeli veya
`PERCEPTION_EVERY_N_FRAMES=2` kullanılmalıdır. Navigasyon haritası varsayılan
olarak dört simülasyon karesinde bir, kamera paneli ise her kare çizilir.

## Sensör yerleşimini görme

Normal uygulama çalışırken ikinci bir `carla` Conda terminalinde:

```powershell
conda run --no-capture-output -n carla python scripts/sensor_layout_viewer.py
```

```bash
conda run --no-capture-output -n carla python scripts/sensor_layout_viewer.py
```

Tarayıcıda aracın üstten ve yandan görünümü açılır. Parlak sensörler o anda
çalışmaktadır; saydam sensörler yerleşimde vardır fakat normal kontrol modunda
açılmamıştır. Bir sensöre tıklayınca konumu, yönü, görüş açısı ve gerçek
menzili gösterilir.

Tarayıcı otomatik açılmazsa `--no-browser` ile HTML dosyası üretilebilir:

```bash
conda run --no-capture-output -n carla python scripts/sensor_layout_viewer.py --no-browser
```

Bu ekran ROS veya RViz kullanmaz.

## Sensör modları

Çalışma biçimi `.env` içindeki tek bir `SENSOR_MODE` ayarıyla seçilir:

| Mod | Açılan sensör | Davranış |
|---|---:|---|
| `control` | 15 | Ön-kamera YOLO+UFLD ve görselsiz BEV doğrulaması çalışır. OpenCV'de BEV paneli yoktur; varsayılan düşük-gecikme modudur. |
| `bev` | 15 | Yedi-kamera YOLO, IPM ve OpenCV BEV panelini açan mühendislik görünümüdür. |
| `record` | 15 | `control` çıkarım kapsamına ek olarak senkron sensör paketlerini `data/runs/` altına kaydeder. |

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

## Her modda BEV doğrulaması

BEV ve tam sensör canlılığı bu sürümde `control`, `bev` ve `record` modlarının
tamamında aktiftir. Normal modda IPM görüntüsü ile BEV canvas çizilmez; füzyon,
takip, occupancy, konumlandırma sağlığı ve güvenli lead recovery arka planda
görselsiz çalışır. Ön kamera sonucu da `camera_results` içine taşındığı için
kamera-radar-LiDAR çapraz doğrulaması korunur. Normal çalıştırma betikleri diske
kayıt yapmayan `control` modunu seçer:

```dotenv
SENSOR_MODE=control
```

Daha sonra betiklerde `-SensorMode bev` (Windows) veya `--bev` (Linux)
kullanılabilir. Manuel çalıştırmadaki eşdeğer değer:

```dotenv
SENSOR_MODE=bev
```

`control` ve `record` modunda OpenCV penceresinde en-boy oranı korunmuş ön
kamera ile sol alttaki küçük navigasyon katmanı bulunur. BEV inset'i
oluşturulmaz; şerit piksel/eğri kaplaması varsayılan olarak kapalıdır.
`bev` modunda kuş bakışı inset'i ve `SURUS / DEBUG` switch'i eklenir; aynı
geçiş klavyedeki `B` tuşuyla da yapılabilir.

`SURUS` modu sürücüye yönelik sade görselleştirmedir:

- koyu ve dikkat dağıtmayan yol sahnesi,
- şerit koridoru ve parlak mavi planlanan rota,
- hız, hedef hız, kontrol modu ve sensör sağlığı,
- stilize ego araç ile izlenen araçların hız ve hareket yönü,
- boş alan ve dolu occupancy bölgeleri.

`DEBUG` modu mühendislik incelemesi için ham ayrıntıları korur:

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
| `validation.py` | Güncel ve bağımsız BEV kanıtıyla kontrol algısını doğrular; tehlikeyi asla temizlemez. |
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

BEV normal kontrol algısının yerine geçmez. Kontrolcü mevcut bir lead ürettiyse
BEV bu hedefi değiştirmez veya uzaklaştıramaz. Birincil lead kısa süreliğine
kaybolursa güncel, rota koridorunda, çok sensörlü ve ham occupancy destekli
doğrulanmış BEV track'i `bev_multisensor_recovery` lead'i olarak IDM/PID
zincirine verilir. Tek modalite veya yalnız occupancy lekesi kontrol üretemez.
BEV ayrıca kontrol algısını güven dereceli ikinci görüş olarak doğrular:
`CONFIRMED`, `SUPPORTED`, `CONFLICT`, `UNKNOWN` veya `UNAVAILABLE` sonucu
üretir. Eksik/eski BEV kanıtı mevcut bir tehlikeyi hiçbir zaman reddetmez;
yalnızca bağımsız, güncel sensör kanıtı doğrulama sayılır. Kontrolcü ön geniş
kamera, ön uzun radar ve yalnızca kutuyla zaman uyumlu olduğunda tavan LiDAR
mesafesini kullanmaya devam eder.
GNSS ve IMU 15 sensörün içinde çalışır fakat ego merkezli çizimde nokta
üretmez; ileride harita konumlandırması için canlı pakette tutulur.

Bu ayrım literatür açısından önemlidir. Projedeki mevcut BEV; kalibrasyon,
IPM, geometrik sensör füzyonu ve occupancy kullanan klasik bir ortak uzamsal
temsildir. Profesyonel ekiplerin kullandığı öğrenilmiş BEV yaklaşımları ise
kamera veya farklı sensörlerden öğrenilen özellikleri ortak BEV uzayına taşıyıp
3B nesne, semantik yol ve occupancy tahminleri üretir. Yeni `SURUS` ekranı bu
veriyi profesyonel bir arayüzle sunar; mevcut geometrik BEV'yi öğrenilmiş bir
occupancy ağıymış gibi göstermez.

Sensörlerden biri birkaç kare gecikirse araç döngüsü bekletilmez. BEV her
sensörün en yeni geçerli verisini kullanır ve eski veriyi kendiliğinden atar.
Occupancy ve takip katmanı aynı sensör frame'ini ikinci kez kanıt saymaz;
decay simülasyon süresine göre uygulanır. Takip association'ı covariance
tabanlı Mahalanobis kapısı ve global atama kullanır.
Füzyon, takip ve occupancy ayrı bir iş parçacığında çalışır. `bev` modunda IPM
ve render da aynı işçiye eklenir. Kuyrukta yalnızca en yeni iş tutulur. Bu
sayede yavaş bir çevre kamerası veya ağır bir BEV karesi direksiyon ve fren
akışını durdurmaz.

Sensörlerin yatay açıları ön, sağ, arka ve sol yönlerde 10 metre test
noktalarıyla doğrulanmıştır. Çevre kameraları bu dört yönü boşluk bırakmadan
ve örtüşmeli görmektedir. Ön kamera, kullanılan CARLA UFLD checkpoint'inin
eğitim geometrisine eşitlenmiştir; radar yerleşimi korunmuştur.

## Kontrol dosyaları

| Dosya | Girdi | Çıktı |
|---|---|---|
| `lead_vehicle.py` | Kamera kutuları, radar noktaları, rota | Takip edilecek ön araç ve acil fren adayı |
| `tracking.py` | Birleştirilmiş araç ölçümleri | Kareler arasında sabit araç kimliği ve yumuşatılmış hareket |
| `pure_pursuit_controller.py` | Araç konumu, hız ve referans rota | `-1` ile `+1` arasında yumuşatılmış direksiyon |
| `pure_pursuit_mpc_controller.py` | Pure Pursuit başlangıcı, rota hatası ve araç modeli | Optimize edilmiş direksiyon; çözüm yoksa Pure Pursuit |
| `speed_planner.py` | Rota eğriliği, hız ve şerit hatası | Metre/saniye cinsinden güvenli hedef hız |
| `behavior_planner.py` | Trafik ışığı, hız tabelası, yaya, viraj ve sensör sağlığı | Davranış modu, güvenli hedef hız ve gerekirse sanal durma noktası |
| `idm_speed_planner.py` | Ego hızı, üst hız ve ön engel | PID hız referansı ve IDM ileri besleme ivmesi |
| `longitudinal_pid_controller.py` | Ego hızı, IDM referansı ve ivmesi | Birbirini dışlayan PID gaz veya fren |
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
| Pure Pursuit sıcak başlangıcı | Araçtan rota üzerinde ileride seçilen hedef noktaya geometrik olarak yönelir. | `pure_pursuit_controller.py` aracı rotaya izdüşürür ve hıza bağlı bakış noktasından ilk direksiyon komutunu bulur. Bu komut, MPC'nin geçerli direksiyon ve değişim sınırlarına uyan başlangıç dizisine dönüşürülür. |
| MPC direksiyon iyileştirmesi | Yakın gelecekteki yanal hata, yön hatası ve direksiyon hareketini birlikte küçültür. | `pure_pursuit_mpc_controller.py` doğrusal hata modelini ve SciPy SLSQP çözücüsünü kullanır. MPC zamanında geçerli bir çözüm veremezse yalnızca Pure Pursuit sonucu uygulanır; üçüncü bir yanal kontrolcü yoktur. |
| Eğrilik tabanlı hız planlama | Viraj keskinleştikçe izin verilen hızı rahat yanal ivmeye göre düşürür. | `speed_planner.py` hız yükseldikçe 35-75 metre ileriyi tarar. Yol eğriliğinden `hız = karekök(yanal ivme / eğrilik)` hesabını yapar. Şerit hatası büyürse ayrıca toparlanma hızı ister. |
| IDM hız referansı | Hız sınırı, takip mesafesi ve kapanma hızından rahat bir ivme isteği hesaplar. | `idm_speed_planner.py` 2 metre duruş boşluğu ve 1,5 saniye zaman aralığı kullanır. IDM ivmesini iki saniyelik hız referansına çevirir; kırmızı ışık ve yaya sanal duran engel gibi aynı hesaba girer. |
| PID hız kontrolü | Referans hız ile mevcut hız arasındaki hatayı aktüatör komutuna dönüştürür. | `longitudinal_pid_controller.py` IDM ivmesini ileri besleme olarak alır; P, I ve filtrelenmiş D terimleri kalan hatayı düzeltir. Takip mesafesini ikinci kez hesaplamaz ve sonuçta yalnızca gaz veya fren üretir. |
| İvme değişim sınırı | Gaz veya fren isteğinin bir çevrimde aniden sıçramasını önler. | `longitudinal_pid_controller.py` istenen ivme değişimini hızlanmada 1,4, frenlemede 3,6 m/s³ ile sınırlar. Gaz ve fren aynı çevrimde birlikte verilmez. |
| Üstel ölçüm yumuşatma | Yeni ölçüm ile önceki değeri belirli oranlarda birleştirir. | `lead_vehicle.py` radar mesafesi ve bağıl hızını, `idm_speed_planner.py` ise IDM'nin kullandığı ön araç durumunu yumuşatır. Yakınlaşan ölçüm güvenlik için uzaklaşan ölçümden daha hızlı kabul edilir. |
| TTC ve gerekli yavaşlama ile acil fren | Mesafe kapanma süresini ve çarpışmayı önlemek için gereken yavaşlamayı hesaplar. | `safety_supervisor.py` takip hedefi ile ham radar tehlikesinden daha riskli olanı seçer. Kritik olmayan tek radar noktasını yeterli saymaz; aynı tehlikeyi ikinci yeni karede de görünce tam fren uygular. |
| Basit IoU/merkez takibi | Aynı kutuyu ardışık karelerde ağır bir takip ağı olmadan eşleştirir. | `road_context.py` sınıf ailesi, IoU ve merkez yakınlığı kullanır; altı karelik kısa kaybı tolere eder ve güveni yumuşatır. |
| Trafik ışığı debounce | Tek yanlış renk veya kaçırılmış kamera karesinin sürüş kararını değiştirmesini önler. | Aynı ışığın rengi üç yeni algılama karesinde doğrulanır; aynı sonuç karesinin tekrar okunması kanıt sayılmaz. Doğrulanmış kırmızı/sarı, bir saniyeye kadar kutu kaçırmalarında korunur ve yalnızca üç yeni yeşil kanıtıyla açılır. |
| Lead trafik ışığı seçimi | Ego rotası için ilk etkili ışığı seçer. | Rota yönünün görüntüdeki hedef merkezi, yatay şerit uyumu, zaman tutarlılığı, kutu alanı ve mesafe birlikte puanlanır; yakın uyumlu ışık uzak kavşaktan önce gelir. |
| Sarı ikilem bölgesi | Konforlu duruş mümkün değilse kavşağı kararlı geçer. | Reaksiyon mesafesinden sonra gerekli yavaşlama hesaplanır; 4 m/s² altında ve kavşak dışında ise durur, aksi halde geçiş kararını ışık değişene kadar korur. |
| Kamera-LiDAR mesafe doğrulaması | Kamera kutusundaki çoklu LiDAR noktasından dayanıklı mesafe çıkarır. | Kalibrasyonla piksele taşınan en az üç noktanın yüzde 25 mesafesi alınır. Kare yaşı ikiyi aşarsa kamera tahminine dönülür; büyük çelişkide yakın değer seçilip güven düşürülür. |

Bu algoritmaların çağrılma sırası `vehicle_controller.py` içinde açıktır:
Pure Pursuit MPC'yi başlatır; hız ve davranış planları üst hızı
belirler; IDM bu sınır ve ön engelden PID referansını üretir; PID gaz
veya fren uygular. En son bağımsız acil fren denetimi çalışır. Acil fren
kararı çıkarsa normal gaz silinir ve fren doğrudan `1.0` yapılır.

### Direksiyon

Yanal zincirde Pure Pursuit her çevrim ilk uygulanabilir direksiyon dizisini
verir. MPC bu sıcak başlangıcı kullanarak rota hatasını, yön hatasını ve
direksiyon hareketlerini birlikte iyileştirir. Direksiyon açısı ve açı değişim
sınırları optimizasyonun içindedir. Düşük hız, kısa rota, zaman aşımı veya
geçersiz çözüm durumunda araç Pure Pursuit ile güvenli biçimde devam eder.

### Hedef hız

Hız planlayıcı hıza göre 45–110 metre ilerideki rotayı tarar. Tabela yoksa düz
yol hedefi 70 km/sa, geçerli tabela varsa düz yol hedefi tabeladaki değerdir.
Her rota bölümü için eğrilikten rahat viraj hızı hesaplanır. Bu hız,
1.8 m/s² konforlu yavaşlama ve sekiz metrelik giriş payıyla geriye taşınarak
aracın o anda uyması gereken hız zarfına çevrilir. Böylece araç hedef hızı
virajın başında birden kesmek yerine viraja yaklaşırken düşürür. Viraj çıkışında
hedef hız en fazla 1.2 m/s² hızlanma eğimiyle yükselir. Çok dar virajlarda
fiziksel yanal ivme sınırı için 23 km/sa altına inilebilir.

### Gaz, fren ve ön araç takibi

Boylamsal zincirde IDM istenen hareketi planlar, PID ise bunu gaz ve frenle
izler. Böylece takip mesafesi tek yerde hesaplanır; PID ayrı bir takip modeliyle
IDM'ye karşı çalışmaz. Temel ayarlar:

- Duruş boşluğu: `2.0 m`
- Zaman aralığı: `1.5 s`
- IDM rahat hızlanması: `1.5 m/s²`
- PID en yüksek hızlanması: `1.5 m/s²`
- En yüksek normal yavaşlama: `3.5 m/s²`
- PID katsayıları: `Kp=0.55`, `Ki=0.10`, `Kd=0.08`

IDM'nin ivmesi hem kısa ufuklu PID hız referansına çevrilir hem de ileri
besleme olarak PID'ye verilir. Bu, duran ön araca yaklaşırken referansın erken
sıfırlanmasını önler; durma noktasına yaklaşırken referans hız
hatası küçülse bile gerekli fren isteğinin kaybolmamasını sağlar. Araç yaklaşık
iki metre boşlukta tamamen durursa `HOLD` durumuna geçer ve eğimde kaymamak
için freni tutar. Ön araç hareket ettiğinde yeniden kalkar. Gaz ile fren aynı
çevrimde birlikte verilmez; integral doygunluğu ve türev gürültüsü sınırlıdır.

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
CAMERA_WAIT_TIMEOUT_MS=10

ENABLE_SIGN_DETECTION=false
ENABLE_LANE_DETECTION=true
SHOW_LANE_OVERLAY=false
LANE_MODEL=models/lane/ufld_carla_best.pth
LANE_DEVICE=auto
LANE_CONFIDENCE=0.15
LANE_MINIMUM_POINTS=3
ENABLE_LIDAR_FUSION=true
ENABLE_DATA_RECORDING=false
SENSOR_MODE=control
BEV_UPDATE_EVERY_N_FRAMES=2
PERCEPTION_LATENCY_BUDGET_MS=80
CAMERA_INFERENCE_BATCH_SIZE=4

STATUS_PERIOD_SECONDS=2.0
MAX_RUNTIME_SECONDS=0
MAXIMUM_SPEED_KMH=70
```

- `VEHICLE_DEVICE=auto`: CUDA varsa ekran kartını, yoksa CPU'yu seçer.
- `PERCEPTION_EVERY_N_FRAMES=1`: her kamera karesini algılamaya gönderir.
- `CAMERA_WAIT_TIMEOUT_MS=10`: kamera jitter'ının ana kontrol ve OpenCV
  döngüsünü yüzlerce milisaniye durdurmasını engeller; geciken son kare bir
  sonraki çevrimde alınır.
- `ENABLE_SIGN_DETECTION=false`: birleşik YOLO modeli araç, yaya, trafik ışığı
  ve 30/60/90 tabelalarını tek geçişte algılar. `true`, yalnız eski ayrı ONNX
  tabela dedektörü+sınıflandırıcısını ek doğrulama için ayrıca çalıştırır ve
  gecikmeyi artırır.
- `ENABLE_LANE_DETECTION=true`: ResNet-18 tabanlı klasik UFLD'yi yalnız ön
  kamerada çalıştırır. 800×288 RGB/ImageNet girdisinden dört şerit adayını,
  piksel noktalarını ve güven değerlerini üretir. Noktalar dayanıklı eğri
  uydurma ve kısa zamansal yumuşatma sonrasında görsel tanı olarak tutulur;
  Pure Pursuit/MPC rotasını değiştirmez. `SHOW_LANE_OVERLAY=false`, normal
  sürüş ekranında bu noktaları ve eğrileri çizmez.
- `ENABLE_LIDAR_FUSION=true`: zaman uyumlu LiDAR noktalarıyla kamera mesafesini doğrular.
- `SENSOR_MODE=control`: kayıt yapmadan 7 kamera, 5 radar, LiDAR, GNSS ve
  IMU'nun tamamını açar; model çıkarımını yalnız ön kamerada, BEV
  doğrulamasını görselsiz çalıştırır. Kullanılmayan altı çevre kamerası CARLA
  sensörü olarak açık kalır ancak CPU'da RGB'ye dönüştürülmez.
- `CAMERA_INFERENCE_BATCH_SIZE=4`: yalnız `PERFORMANCE_PROFILE=manual` iken
  ve `SENSOR_MODE=bev` seçiliyken geçerlidir. `control`/`record` her donanımda
  tek ön kamera çıkarımı yapar.
- `BEV_UPDATE_EVERY_N_FRAMES=2`: 20 Hz simülasyonda en fazla
  10 Hz doğrulama sahnesi üretir; `bev` modunda ayrıca kuş bakışını çizer.
- `PERCEPTION_LATENCY_BUDGET_MS=80`: yalnız yeni tamamlanan sonuçların kuyruk
  dahil uçtan uca gecikme hedefidir; aynı eski sonuç her simülasyon karesinde
  yeniden ortalamaya katılmaz.
- `ENABLE_DATA_RECORDING=false`: eski kurulumlarla uyumluluk için tutulur.
  `true` yapılırsa `SENSOR_MODE` değerinden bağımsız olarak kayıt modu seçilir.
- `MAX_RUNTIME_SECONDS=0`: kullanıcı kapatana kadar çalışır.
- `MAXIMUM_SPEED_KMH=70`: tabela görülmediğinde düz ve boş yol hedefidir.

## Terminal durum satırı

Uygulama belirlenen aralıkta bir `[STATUS]` satırı yazar:

| Alan | Anlamı |
|---|---|
| `speed` | Ego aracının mevcut hızı |
| `target` | Viraj ve şerit durumundan sonra seçilen hedef hız |
| `mode` | `CRUISE`, `LEAD_FAR`, `FOLLOW`, `HOLD`, `RESTART` veya `EMERGENCY` |
| `lateral` / `lookahead` | Aktif Pure Pursuit-MPC modu ve bakış mesafesi |
| `mpc` | MPC çözüm süresi, iterasyon ve varsa fallback nedeni |
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
| `idm_ref`, `idm_a`, `idm_gap` | IDM hız referansı, ivmesi ve dinamik takip boşluğu |
| `aeb` | Acil fren nedeni ve tehlike bilgisi |
| `light` | Seçilen lead trafik ışığının rengi ve mesafesi |
| `limit` | Zamansal olarak doğrulanmış hız sınırı |
| `pedestrian` | `MONITOR`, `SLOW`, `PREPARE_STOP` veya `EMERGENCY` yaya riski |
| `lidar_age` | Kontrolde kullanılan son LiDAR karesinin yaşı |

Yaya algısının sürüş kararına girebilmesi için model güveni en az
`0.90` olmalıdır. Daha düşük güvenli `person` sonuçları yaya yavaşlatma veya
durma kararı oluşturmaz.

## Veri kaydı

Tam sensör paketi gerektiğinde `.env` içinde:

```dotenv
SENSOR_MODE=record
```

Çalıştırma betikleriyle aynı mod Windows'ta `-SensorMode record`, Linux'ta
`--record` seçeneğiyle açılır.

Bu mod 15 sensörü de açar ve daha fazla işlem gücü kullanır. Görüntü, LiDAR,
radar, GNSS, IMU ve kalibrasyon dosyaları `data/runs/` altında saklanır.
Eski `ENABLE_DATA_RECORDING=true` ayarı da aynı kayıt modunu seçmeye devam
eder.

## Nesne modeli ve standart algılama biçimi

Birlikte gelen `carla_yolov8n_best.pt`,
[CARLA-Object-Detection](https://github.com/LinkouCommander/CARLA-Object-Detection)
veri tanımındaki on sınıfı kullanır: `bike`, `motobike`, `person`, üç trafik
ışığı rengi, 30/60/90 hız tabelaları ve `vehicle`. Kaynak veri seti 1600 adet
416x416 CARLA görüntüsüdür ve CC BY 4.0 lisanslıdır. Model ham çıktısı kontrolcüye
verilmez. `road_context.py` her kutuya sınıf, güven, merkez, genişlik, yükseklik,
kamera/LiDAR mesafesi, kare numarası, takip kimliği ve geçerlilik alanlarını
ekler.

Yerel CPU doğrulamasında 416x416 örnek görüntü, `imgsz=640` ve `conf=0.05` ile
ısınma sonrasında yaklaşık `76.8 ms` duvar süresinde işlendi (`70.2 ms` model
çıkarımı). İlk model yükleme dahil çağrı yaklaşık `3.34 s` sürdü. Bu nedenle
algılama ana kontrol döngüsünden ayrı, yalnızca en yeni kareyi tutan işçide
çalışmaya devam eder; gerçek hız donanıma göre yeniden ölçülmelidir.

Araç kutuları mevcut kamera-radar ön araç izleyicisine gider. Trafik ışığı,
hız tabelası ve yaya kutuları zamansal doğrulamadan sonra davranış planlayıcısına
gider. Kontrolcü model sınıf adlarını yorumlamaz; yalnızca davranış planının
hedef hızı ve gerekirse ürettiği sanal durma noktasını uygular.

## Trafik davranışları

- Kırmızıda reaksiyon, fren ve emniyet mesafesinden yaklaşma hızı hesaplanır;
  araç durunca sanal duran hedef sayesinde fren tutulur.
- Sarıda gerekli yavaşlama güvenli sınırdaysa durulur. Araç ikilem bölgesinde
  veya kavşak içindeyse ışık değişene kadar kararlı geçiş yapılır.
- Yeşil üç yeni karede doğrulanır; kırmızıdan sonra hedef hız en fazla
  `1.2 m/s²` artışla açılır.
- Doğrulanmış kırmızı veya sarı ışık, kamera kutuyu aralıklı kaçırsa bile bir
  saniyelik sensör zaman aşımı boyunca tutulur; eski sonuç karesinin tekrar
  okunması yeni kanıt sayılmaz.
- Hız tabelası üç yeni karede doğrulanır. Düşük limit hemen güvenli hedef olur;
  yüksek limite geçiş kademelidir.
- Yol kenarındaki yaya yalnızca izlenir. Sürüş koridorundaki yaya mesafeye göre
  yavaşlama, durmaya hazırlanma veya acil fren seviyesine çıkar.
- Ön araç ile trafik kuralının durma noktası aynı anda varsa boylamsal
  kontrolcü yakın olan hedefi izler.
- Algılama bir saniyeden uzun süre güncellenmez veya model sürekli hata verirse
  araç konforlu duruş profiliyle `SENSOR_DEGRADED` moduna geçer.

## Başlangıç güvenlik parametreleri

Tüm başlangıç değerleri `DrivingParameters` içinde tek yerde bulunur:

| Parametre | Değer |
|---|---:|
| Işık / hız tabelası doğrulama | 3 yeni algılama karesi |
| Kısa algılama kaybı toleransı | 6 kare |
| Sensör zaman aşımı | 1.0 s |
| Konforlu / en yüksek normal / acil yavaşlama | 2.0 / 4.0 / 8.0 m/s² |
| Reaksiyon süresi / duruş payı | 0.50 s / 2.0 m |
| Takip zamanı / minimum boşluk | 1.5 s / 2.0 m |
| Yaya yavaşlama / durma / acil mesafesi | 25 / 12 / 5 m |
| LiDAR azami kare farkı / asgari nokta | 2 kare / 3 nokta |
| Kamera-LiDAR çelişme eşiği | 5 m veya kamera mesafesinin %35'i |
| Yeşilde hedef hız artışı | 1.2 m/s² |

## Doğrulama

Kod değişikliğinden sonra iki komut birlikte çalıştırılmalıdır:

```bash
python -m compileall -q carla_app main.py scripts tests
python -m unittest discover -s tests -v
```

CARLA sunucusu, `ego_vehicle` rolündeki araç ve lane model dosyası hazırken
gerçek model çizgisini yalnız test sürecinde CARLA harita sınırlarıyla
karşılaştırmak için:

```bash
RUN_CARLA_LANE_GT=1 python -m unittest discover \
  -s tests -p "test_lane_ground_truth.py" -v
```

Bu canlı test, ego şeridinin CARLA waypoint merkezinden `lane_width / 2` ile
üretilen işaretli sol/sağ sınırlarını gerçek kamera kalibrasyonuyla görüntüye
projekte eder. Şerit eşleme F1'ını, ortak düşey kapsamı, ortalama ve yüzde 95
piksel hatasını denetler. Ground-truth modülleri `tests/` altında kalır ve
normal uygulama tarafından içe aktarılmaz.

Testler; direksiyon yönünü, direksiyon değişim sınırını, viraj hızını, şerit
toparlamayı, ön araç takibini, iki metre duruşu, yeniden kalkışı, kamera-radar
birleşimini, komşu şerit ve zemin reddini, eski sensör karesini, acil freni,
`K-R-T` kalibrasyonunu, homografi dönüşümünü, 360 derece kamera kapsamasını,
sensör tekrar silmeyi, güven ağırlıklı füzyonu, ego hareket telafisini, Kalman
takibini, occupancy grid'i, çoklu kamera akışını ve sensör modu seçimini kapsar.
Ek trafik testleri; 20 istenen senaryonun karar katmanlarını kırmızı/sarı/yeşil,
iki ışık, yan şerit ışığı, hız tabelaları, yaya kademeleri, algılama kaybı,
LiDAR düşmesi/çelişkisi ve model hatası örnekleriyle denetler. Canlı fizik ve
durma noktası hatası ölçümü için CARLA sunucusunun ayrıca çalıştırılması gerekir.

Kontrol denklemlerinin temel kaynakları:

- [UFLD satır-ankrajlı şerit algılama (ECCV 2020)](https://arxiv.org/abs/2004.11757)
- [TuSimple şerit doğrulama protokolü](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection)
- [CARLA harita, waypoint ve lane-marking API'si](https://carla.readthedocs.io/en/latest/core_map/)
- [Sürüş odaklı şerit algılama metrikleri](https://arxiv.org/abs/2203.16851)
- [Pure Pursuit geometrik yol takibi](https://www.ri.cmu.edu/pub_files/pub4/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
- [Intelligent Driver Model (IDM)](https://mtreiber.de/MicroApplet/IDM.html)
- [SciPy SLSQP optimizasyonu](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
- [MPC'de sıcak başlangıç](https://web.stanford.edu/~boyd/fast_mpc/)
- [BEVFusion ortak BEV sensör füzyonu](https://arxiv.org/abs/2205.13542)
- [PID kontrol](https://doi.org/10.1109/TSMC.2000.843250)
- [Otonom sürüş hareket planlama ve kontrol yöntemleri incelemesi](https://doi.org/10.1109/TIV.2016.2578706)
- [Trafik ışığı algılama ve zamansal takip incelemesi](https://doi.org/10.1109/TITS.2015.2509509)
- [NHTSA otomatik acil fren standardı](https://www.federalregister.gov/documents/2024/05/09/2024-09054/federal-motor-vehicle-safety-standards-automatic-emergency-braking-systems-for-light-vehicles)
