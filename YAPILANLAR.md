
## Araç kontrolcüsü
Kamera için 2D bbox bilgisi model aracılığı ile 
Radar için de mesafe hiz , aci : radar_to_list() fonksiyonu

fusion.py de bbox'in acisi hesaplandı radarın da aynı açıdakı noktaları eslendi 
en yakın noktadan mesafe + medyan hız = göreli hız çıktı
sonra yolonun gecikmeli çalışması ile radarın anlık çalışması sebebiyle aradaki frame farkını hesaplayıp ona gore extrapole ederek
zamanı hizalıyoruz

   sonuc: her tespit icin
   (range_m, bearing_deg, rel_v) -
   yani GERCEK, METRIK bir olcum

 +---------- TAKIP -----------+
        (tracking.py)
   Kalman filtresi (x,y ayri ayri):
   - eslesen olcumu track'e isle
     (gurultuyu azalt, hiz cikar)
   - eslesmeyen olcum icin yeni
     track ac
   - kayip track'leri predict()
     ile "fizige gore" ilerlet,
     cok kayipsa sil
        |
   sonuc: guvenilir, surekli,
   ID'li track'ler - her biri
   (x, y, vx, vy) biliyor ve
   "N saniye sonra nerede olur"
   diye sorulabiliyor