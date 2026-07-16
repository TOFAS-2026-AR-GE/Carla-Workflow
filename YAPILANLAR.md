# Controller Improvements

- Vehicle YOLO sınıfları model metadata'sından doğrulanıyor.
- GPU kullanılamazsa inference otomatik CPU'ya düşüyor.
- Levha modeli araç tespitinden ayrıldı; bir modelin hatası diğer sonucu silmiyor.
- OpenCV penceresinde bbox, inference gecikmesi ve detector hatası gösteriliyor.
- Pencerenin X düğmesi ana döngüyü gerçekten sonlandırıyor.
- Normal çalışmada yalnızca ön kamera ve ön radar açılıyor.
- Kamera-radar range verisinin ikinci kez extrapolate edilmesi kaldırıldı.
- Radar-only lead seçimine zamansal doğrulama, kısa kayıp toleransı ve komşu
  şerit reddi eklendi.
- Uzak lead için dinamik ACC aktivasyonu korunuyor; gereksiz yavaşlama önleniyor.
- Stanley kontrolüne hata filtresi, daha sağlam eğrilik hesabı ve hız tabanlı
  direksiyon/rate limitleri eklendi.
- Kullanılmayan eski MPC ve LaneFollow kontrolcüleri kaldırıldı.
- Tekrarlanabilir kontrol ve perception birim testleri eklendi.
