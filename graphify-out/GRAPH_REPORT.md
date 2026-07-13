# Graph Report - Carla-Workflow  (2026-07-14)

## Corpus Check
- 30 files · ~9,998 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 284 nodes · 421 edges · 20 communities (18 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `aeef8a5f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- main.py
- gtsrb_belgium_78sinif_hazirla.py
- yeniden_split.py
- SensorSync
- DataCollector.py
- sensor_processors.py
- detect_then_classify.py
- belgium_to_yolo.py
- DataCollector
- VehicleDetector
- main
- main
- argumanlari_oku
- argumanlari_oku
- Trafik Levhası Tespit ve Sınıflandırma Projesi
- Trafik Levhası Model ve Geliştirme Paketi
- README.md
- README.md
- update_spectator_camera

## God Nodes (most connected - your core abstractions)
1. `SensorSync` - 15 edges
2. `main()` - 11 edges
3. `DataCollector` - 10 edges
4. `DatasetWriter` - 9 edges
5. `main()` - 9 edges
6. `FramePacket` - 9 edges
7. `process_sensor()` - 9 edges
8. `update_spectator_camera()` - 9 edges
9. `belgium_kopyala()` - 9 edges
10. `gtsrb_kopyala()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `DataCollector` --uses--> `SensorSync`  [INFERRED]
  DataCollector.py → processing/SensorSync.py
- `DataCollector` --uses--> `DatasetWriter`  [INFERRED]
  DataCollector.py → collection/DatasetWriter.py
- `main()` --calls--> `DataCollector`  [EXTRACTED]
  main.py → DataCollector.py
- `main()` --calls--> `update_spectator_camera()`  [EXTRACTED]
  main.py → spec_camera.py
- `main()` --calls--> `get_datas()`  [EXTRACTED]
  main.py → carla_datas.py

## Import Cycles
- None detected.

## Communities (20 total, 2 thin omitted)

### Community 0 - "main.py"
Cohesion: 0.13
Nodes (17): get_datas(), Vehicle, World, connect_carla(), clamp(), LaneFollowController, Basit şerit takip kontrolcüsü.      Görevleri:     1. İlerideki waypoint'e gö, get_vehicle_data() (+9 more)

### Community 1 - "gtsrb_belgium_78sinif_hazirla.py"
Cohesion: 0.13
Nodes (29): argumanlari_oku(), belgium_kopyala(), belgium_resim_hedefi(), belgium_sinif_hedefi(), gtsrb_girdilerini_bul(), gtsrb_kopyala(), kaynak_sinif_klasorleri(), main() (+21 more)

### Community 2 - "yeniden_split.py"
Cohesion: 0.11
Nodes (29): argumanlari_oku(), data_yaml_yaz(), isim_listesi(), klasorleri_olustur(), label_klasoru_bul(), main(), mutlak_yol(), nadirlik_puani() (+21 more)

### Community 3 - "SensorSync"
Cohesion: 0.09
Nodes (12): FramePacket, CARLA sensör verilerini frame numarasına göre eşleştirir., Sensör verilerini CARLA frame numarasına göre toplar., Bütün beklenen sensörler geldi mi?, Kullanılmayacak eski frame'leri temizler.          Bu fonksiyon çağrılırken lo, Buffer'ın bellekte sınırsız büyümesini engeller.          Bu fonksiyon çağrılı, Belirtilen frame ve öncesindeki verileri temizler., Bütün buffer'ı temizler. (+4 more)

### Community 4 - "DataCollector.py"
Cohesion: 0.16
Nodes (22): Bütün sensörleri oluşturur ve ego araca bağlar., Image, LidarMeasurement, ndarray, lidar_cloud_to_array(), Vehicle, World, CARLA lidar point cloud -> (N, 4) numpy array [x, y, z, intensity]. (+14 more)

### Community 5 - "sensor_processors.py"
Cohesion: 0.17
Nodes (16): process_frame(), process_gnss(), process_imu(), process_lidar(), process_radar(), process_rgb_camera(), process_semantic_camera(), process_sensor() (+8 more)

### Community 6 - "detect_then_classify.py"
Cohesion: 0.17
Nodes (18): argumanlari_oku(), dosya_adi_temizle(), kaynaklari_topla(), kirpme_koordinati(), kutu_ciz(), main(), Detection kutusunu ve sınıflandırma metnini görüntü üzerine çizer., Tek görüntüde detection, crop, classification ve çıktı kaydını yapar. (+10 more)

### Community 7 - "belgium_to_yolo.py"
Cohesion: 0.17
Nodes (16): annotation_dosyasi_bul(), argumanlari_oku(), camera_yolunu_bul(), class_map(), main(), Belgium kaynak ve YOLO çıktı klasörü argümanlarını okur., Train/test annotationlarını dönüştürür ve data.yaml dosyasını yazar., Olası annotation dizinlerinde istenen GT dosyasını bulur. (+8 more)

### Community 8 - "DataCollector"
Cohesion: 0.13
Nodes (10): DatasetWriter, İşlenmiş sensör verilerini diske kaydeder., Sensör verilerini düzenli klasörlere kaydeder., Tek bir sensör verisini kaydeder., Frame özetini manifest.jsonl dosyasına ekler., Bir frame'e ait bütün sensör verilerini kaydeder., DataCollector, Belirtilen CARLA frame'ine ait sensör verilerini toplar.          Args: (+2 more)

### Community 9 - "VehicleDetector"
Cohesion: 0.29
Nodes (4): VehicleDetector nesnesini oluşturur ve YOLO modelini yükler.          model_pa, Verilen RGB görüntüsünde araç, bisiklet ve motosiklet algılar.          Parame, RGB kamera görüntüsünde şu nesneleri algılar:      0 -> bike     1 -> motobik, VehicleDetector

### Community 10 - "main"
Cohesion: 0.38
Nodes (6): argumanlari_oku(), labels_donustur(), main(), Kaynak, çıktı ve yeni tek sınıf adını okur., Bir labels klasöründeki bütün sınıf ID'lerini 0 yapar., Splitleri dönüştürür, görüntüleri kopyalar ve data.yaml yazar.

### Community 11 - "main"
Cohesion: 0.38
Nodes (6): argumanlari_oku(), main(), İki kaynak dataset ve çıktı klasörü argümanlarını okur., Bir datasetin splitlerini hedefe benzersiz önekle kopyalar., Çıktıyı temizler, datasetleri birleştirir ve data.yaml oluşturur., split_kopyala()

### Community 12 - "argumanlari_oku"
Cohesion: 0.50
Nodes (4): argumanlari_oku(), main(), Classifier eğitimi için komut satırı seçeneklerini okur., Classifier modelini yön-koruyan augmentation ayarlarıyla eğitir.

### Community 13 - "argumanlari_oku"
Cohesion: 0.50
Nodes (4): argumanlari_oku(), main(), Detection eğitimi için komut satırı seçeneklerini okur., YOLO modelini oluşturur, eğitimi başlatır ve ağırlık yollarını yazdırır.

### Community 15 - "Trafik Levhası Tespit ve Sınıflandırma Projesi"
Cohesion: 0.11
Nodes (17): 1. Belgium detection verisini YOLO formatına çevirme, 2. Detection sınıflarını tek sınıfa indirme, 3. GTSDB ve Belgium detection verilerini birleştirme, 4. Detection train/val/test splitlerini yeniden oluşturma, 5. GTSRB ve Belgium classification verilerini hazırlama, Classification, Classifier eğitimi, Detection (+9 more)

### Community 16 - "Trafik Levhası Model ve Geliştirme Paketi"
Cohesion: 0.33
Nodes (5): İçerik, Mevcut sanal ortamla çalıştırma, Notlar, PT modellerini kullanma, Trafik Levhası Model ve Geliştirme Paketi

### Community 19 - "update_spectator_camera"
Cohesion: 0.31
Nodes (10): Location, Rotation, _lerp(), _lerp_angle(), _normalize_angle(), Vehicle, World, _smooth_location() (+2 more)

## Knowledge Gaps
- **20 isolated node(s):** `Basit Kurulum`, `Senaryo Nereden Değişir?`, `Hazır CARLA YOLOv8n modeli`, `Detection`, `Classification` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SensorSync` connect `SensorSync` to `DataCollector`, `DataCollector.py`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `DataCollector` connect `DataCollector` to `main.py`, `SensorSync`, `DataCollector.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `DatasetWriter` connect `DataCollector` to `DataCollector.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `SensorSync` (e.g. with `DataCollector` and `FramePacket`) actually correct?**
  _`SensorSync` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `DataCollector` (e.g. with `DatasetWriter` and `SensorSync`) actually correct?**
  _`DataCollector` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Ego araca sensörleri bağlar.      Aynı CARLA frame'ine ait sensör verilerini t`, `Bütün sensörleri oluşturur ve ego araca bağlar.`, `Belirtilen CARLA frame'ine ait sensör verilerini toplar.          Args:` to the rest of the system?**
  _112 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `main.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13105413105413105 - nodes in this community are weakly interconnected._