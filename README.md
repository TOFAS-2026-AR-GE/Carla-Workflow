
## Basit Kurulum

1. Repo'yu klonla:

```powershell
git clone https://github.com/TOFAS-2026-AR-GE/Carla-Workflow.git
cd Carla-Workflow
```

2. Gerekli paketleri kur:

```powershell
pip install -r requirements.txt
```

3. CARLA server'ı aç.

4. İstersen `.env` içinden aktif senaryoyu seç:

Trafikli dünya:

```env
SCENARIO_FILE=scenarios/traffic_scenario/configs/traffic_world.yaml
```

Trafiksiz default dünya:

```env
SCENARIO_FILE=scenarios/default_scenario/configs/default_world.yaml
```

5. Projeyi çalıştır:

```powershell
python main.py
```

## Senaryo Nereden Değişir?

Senaryo seçimi `.env` dosyasındaki `SCENARIO_FILE` satırından yapılır.

Senaryonun içindeki map ve trafik ayarları ise ilgili `.yaml` dosyasından değiştirilir:

- `scenarios/traffic_scenario/configs/traffic_world.yaml`
- `scenarios/default_scenario/configs/default_world.yaml`

