# Mimari harita

Bu belge projeye ilk kez bakan biri için kısa yol haritasıdır. Kurulum,
parametreler ve algoritma ayrıntıları için [README](README.md) kullanılır.

## Tek cümlede sistem

CARLA sensörleri çevreyi ölçer; algılama katmanı ölçümleri anlamlandırır;
planlayıcı güvenli hedefi seçer; kontrolcü direksiyon, gaz ve freni üretir;
bağımsız güvenlik katmanı son komutu sınırlar.

## Çalışma akışı

```text
main.py
  -> CarlaApplication
      -> core/       CARLA bağlantısı, rota haritası ve trafik üretimi
      -> sensors/    7 kamera, 5 radar, LiDAR, GNSS ve IMU verisi
      -> localization/ GNSS + IMU + isteğe bağlı teker odometrisi EKF
      -> perception/ YOLO + UFLD + zamansal yol bağlamı
      -> bev/        sürekli kuş bakışı doğrulama ve lead recovery
      -> controller/ davranış, direksiyon, hız, gaz-fren ve acil fren
      -> validation/ kontrol dışı CARLA oracle başarı ölçümü
      -> visualization/ OpenCV kamera/UFLD ve isteğe bağlı BEV görünümü
```

Her simülasyon karesinin sahibi `CarlaApplication.process_frame()` metodudur.
Bu metot sensörü okur, en güncel algılama sonucunu alır, yol bağlamını günceller,
kontrol komutunu üretir ve araca uygular.

## Eşzamanlı çalışan parçalar

| Akış | Sorumluluk | Kontrol döngüsünü bekletir mi? |
|---|---|---|
| Ana CARLA döngüsü | Araç durumu, karar, kontrol ve ekran | Ana akıştır |
| `PerceptionWorker` | Normal modda ön-kamera YOLO+UFLD; `bev` modunda yedi-kamera YOLO | Hayır; yalnız en yeni kareyi tutar |
| BEV işçisi | Her modda füzyon/takip/occupancy; `bev` modunda ayrıca IPM ve çizim | Hayır; her modda açılır |

Eski kamera işleri kuyrukta biriktirilmez. Böylece model yavaşladığında kontrol
döngüsü eski kareleri sırayla işlemeye çalışmaz.

## Kararların gerçek kaynakları

| Karar | Ana kaynak | İlgili modül |
|---|---|---|
| Araç pozu ve hızı | GNSS + IMU EKF, varsa teker/CAN odometrisi | `localization/` |
| Direksiyon | EKF pozu üzerinde CARLA harita rotası | `pure_pursuit_mpc_controller.py` |
| Viraj hızı | Rota eğriliği, şerit merkezleme hatası ve IMU kararlılığı | `speed_planner.py` |
| Ön araç takibi | Kamera + radar, covariance + Hungarian/Mahalanobis, gerektiğinde BEV recovery | `lead_vehicle.py`, `tracking.py` |
| Trafik kuralı | Zamansal doğrulanmış yol bağlamı | `road_context.py`, `behavior_planner.py` |
| Gaz ve fren | IDM hız referansı + PID | `idm_speed_planner.py`, `longitudinal_pid_controller.py` |
| Acil fren | Bağımsız yakın tehlike denetimi | `safety_supervisor.py` |
| UFLD şeritleri | Ön RGB kamera | Dayanıklı eğri + zamansal görsel filtre; yalnız `viewer.py`, kontrolcüye bağlı değildir |

Bu ayrım önemlidir: algılama sonucu doğrudan gaz, fren veya direksiyon komutu
üretmez. Önce standart yol bağlamına veya ön araç takibine dönüştürülür.

## Sensör modları

| Mod | Amaç | Çalışan omurga |
|---|---|---|
| `control` | Normal sürüş | 15 sensör, ön-kamera çıkarımı ve görselsiz BEV doğrulaması |
| `bev` | Kuş bakışı odaklı sürüş | 15 sensör, yedi-kamera çıkarımı, IPM ve BEV paneli |
| `record` | Senkron veri toplama | `control` omurgası ve ek disk yazımı |

Yedi kameranın tamamı her modda sensör/kalibrasyon kanıtı olarak canlıdır.
Ancak normal modlarda model çıkarımı yalnız birincil kamerada yapılır; donanıma
göre batch parçalama yalnız açıkça seçilen `bev` modunun yedi-kamera çıkarımını
etkiler.

UFLD ayrı bir sensör modu değildir. `ENABLE_LANE_DETECTION=true` olduğunda
yalnız birincil ön kamerada çalışır ve her modda görsel sonuç üretebilir.
CARLA waypoint tabanlı şerit ground-truth projeksiyonu yalnız `tests/` altında
bulunur; üretim algılama veya kontrol akışına yüklenmez.

OpenCV pencere ölçüsü kamera kalibrasyonundan ayrıdır. Varsayılan
`1500x600` gösterim, `1640x590` ve 150 derece UFLD eğitim kamerasını yalnız
letterbox ile küçültür; sensör pozunu ve model piksel koordinatlarını oynatmaz.

Trafik ışığında normal kontrol kaynağı kameradır. Varsayılan
`TRAFFIC_LIGHT_ORACLE_MODE=validation` ayarında CARLA'nın etkileyen ışık rengi
ve stop waypoint'i yalnız `validation/oracle.py` içinde başarı ölçümü üretir;
kontrol sözlüğüne girmez. Simülasyon güvenlik yedeği ancak açıkça
`fallback` seçildiğinde eklenir. Yeşilde duran araç ayrıca ön/arka köşe
radarlarıyla çatışma bölgesi birkaç taze kare temiz görülmeden kalkmaz.

## Nerede değişiklik yapılır?

| İstenen değişiklik | Başlangıç dosyası |
|---|---|
| Ortam değişkeni veya varsayılan değer | `carla_app/config.py` |
| Sensör konumu/FOV/çözünürlük | `carla_app/sensors/layout.py` |
| GNSS/IMU/odometri lokalizasyonu | `carla_app/localization/` |
| Oracle karşılaştırması | `carla_app/validation/oracle.py` |
| YOLO veya UFLD algılama | `carla_app/perception/` |
| Trafik ışığı, tabela veya yaya kararı | `road_context.py`, `behavior_planner.py` |
| Kavşak ve çapraz trafik gözetimi | `intersection_guard.py` |
| Ön araç mesafesi ve radar eşleştirme | `lead_vehicle.py`, `tracking.py` |
| Ölçüm yaşı/covariance kontrol zarfı | `uncertainty.py` |
| Direksiyon davranışı | `pure_pursuit_controller.py`, `pure_pursuit_mpc_controller.py` |
| Hız, gaz veya fren | `speed_planner.py`, `idm_speed_planner.py`, `longitudinal_pid_controller.py` |
| OpenCV/BEV çizimi | `carla_app/visualization/`, `carla_app/bev/renderer.py` |
| Regresyon doğrulaması | `tests/` |

## Güvenli genişletme kuralı

Yeni bir model önce bağımsız bir algılama sonucu üretmeli, hata durumunda diğer
modelleri bozmamalı ve worker içinde çalışmalıdır. Kontrole bağlanmadan önce
güven eşiği, zamansal filtreleme, sensör yaşı ve kapalı çevrim testleri ayrıca
tanımlanmalıdır.
