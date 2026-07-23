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
      -> core/       CARLA bağlantısı, ego araç, rota ve trafik
      -> sensors/    7 kamera, 5 radar, LiDAR, GNSS ve IMU verisi
      -> perception/ YOLO + UFLD + zamansal yol bağlamı
      -> bev/        sürekli kuş bakışı doğrulama ve lead recovery
      -> controller/ davranış, direksiyon, hız, gaz-fren ve acil fren
      -> visualization/ OpenCV kamera ve BEV görünümü
```

Her simülasyon karesinin sahibi `CarlaApplication.process_frame()` metodudur.
Bu metot sensörü okur, en güncel algılama sonucunu alır, yol bağlamını günceller,
kontrol komutunu üretir ve araca uygular.

## Eşzamanlı çalışan parçalar

| Akış | Sorumluluk | Kontrol döngüsünü bekletir mi? |
|---|---|---|
| Ana CARLA döngüsü | Araç durumu, karar, kontrol ve ekran | Ana akıştır |
| `PerceptionWorker` | YOLO, isteğe bağlı ONNX ve UFLD çıkarımı | Hayır; yalnız en yeni kareyi tutar |
| BEV işçisi | IPM, füzyon, takip, occupancy ve çizim | Hayır; her modda açılır |

Eski kamera işleri kuyrukta biriktirilmez. Böylece model yavaşladığında kontrol
döngüsü eski kareleri sırayla işlemeye çalışmaz.

## Kararların gerçek kaynakları

| Karar | Ana kaynak | İlgili modül |
|---|---|---|
| Direksiyon | CARLA harita rotası | `pure_pursuit_mpc_controller.py` |
| Viraj hızı | Rota eğriliği, şerit merkezleme hatası ve IMU kararlılığı | `speed_planner.py` |
| Ön araç takibi | Kamera + radar, gerektiğinde BEV recovery | `lead_vehicle.py` |
| Trafik kuralı | Zamansal doğrulanmış yol bağlamı | `road_context.py`, `behavior_planner.py` |
| Gaz ve fren | IDM hız referansı + PID | `idm_speed_planner.py`, `longitudinal_pid_controller.py` |
| Acil fren | Bağımsız yakın tehlike denetimi | `safety_supervisor.py` |
| UFLD şeritleri | Ön RGB kamera | Yalnız `viewer.py`; kontrolcüye bağlı değildir |

Bu ayrım önemlidir: algılama sonucu doğrudan gaz, fren veya direksiyon komutu
üretmez. Önce standart yol bağlamına veya ön araç takibine dönüştürülür.

## Sensör modları

| Mod | Amaç | Çalışan omurga |
|---|---|---|
| `control` | Normal sürüş | 15 sensör ve BEV doğrulaması |
| `bev` | Kuş bakışı odaklı sürüş | 15 sensör ve BEV doğrulaması |
| `record` | Senkron veri toplama | Aynı omurga ve ek disk yazımı |

Kamera çıkarımı donanıma göre parçalanır: düşük VRAM'de tek kamera/batch,
yüksek VRAM'de birden çok kamera/batch kullanılır. Bu seçim sensör kapsamını
değil yalnız çıkarımın GPU üzerindeki gruplanmasını değiştirir.

UFLD ayrı bir sensör modu değildir. `ENABLE_LANE_DETECTION=true` olduğunda
yalnız birincil ön kamerada çalışır ve her modda görsel sonuç üretebilir.

## Nerede değişiklik yapılır?

| İstenen değişiklik | Başlangıç dosyası |
|---|---|
| Ortam değişkeni veya varsayılan değer | `carla_app/config.py` |
| Sensör konumu/FOV/çözünürlük | `carla_app/sensors/layout.py` |
| YOLO veya UFLD algılama | `carla_app/perception/` |
| Trafik ışığı, tabela veya yaya kararı | `road_context.py`, `behavior_planner.py` |
| Ön araç mesafesi ve radar eşleştirme | `lead_vehicle.py` |
| Direksiyon davranışı | `pure_pursuit_controller.py`, `pure_pursuit_mpc_controller.py` |
| Hız, gaz veya fren | `speed_planner.py`, `idm_speed_planner.py`, `longitudinal_pid_controller.py` |
| OpenCV/BEV çizimi | `carla_app/visualization/`, `carla_app/bev/renderer.py` |
| Regresyon doğrulaması | `tests/` |

## Güvenli genişletme kuralı

Yeni bir model önce bağımsız bir algılama sonucu üretmeli, hata durumunda diğer
modelleri bozmamalı ve worker içinde çalışmalıdır. Kontrole bağlanmadan önce
güven eşiği, zamansal filtreleme, sensör yaşı ve kapalı çevrim testleri ayrıca
tanımlanmalıdır.
