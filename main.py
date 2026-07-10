"""CARLA aracını sürer ve araç hareket ederken sensör verilerini toplar."""

import os

from dotenv import load_dotenv

from carla_datas import get_datas
from connect_carla import connect_carla
from controllers import LaneFollowController
from DataCollector import DataCollector
from spec_camera import update_spectator_camera
from vehicle import spawn_vehicle

FIXED_DELTA_SECONDS = 0.05
SAVE_EVERY_N_FRAMES = 5


def get_vehicle_data(data, control):

    return {
        "x": float(data["x"]),
        "y": float(data["y"]),
        "z": float(data["z"]),
        "yaw": float(data["yaw"]),
        "speed_mps": float(data["speed_mps"]),
        "speed_kmh": float(data["speed_kmh"]),
        "road_id": int(data["road_id"]),
        "lane_id": int(data["lane_id"]),
        "lane_width": float(data["lane_width"]),
        "is_junction": bool(data["is_junction"]),
        "control": {
            "throttle": float(control.throttle),
            "steer": float(control.steer),
            "brake": float(control.brake),
        },
    }


def main():
    load_dotenv()

    host = os.getenv(
        "HOST",
        "127.0.0.1",
    )

    port = int(
        os.getenv(
            "PORT",
            "2000",
        )
    )

    timeout = float(
        os.getenv(
            "TIMEOUT",
            "15.0",
        )
    )

    vehicle_name = os.getenv(
        "VEHICLE_NAME",
        "vehicle.tesla.model3",
    )

    client = None
    world = None
    vehicle = None
    data_collector = None
    original_world_settings = None

    try:
        client, world = connect_carla(
            host=host,
            port=port,
            timeout=timeout,
        )

        original_world_settings = world.get_settings()
        new_settings = world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = FIXED_DELTA_SECONDS

        world.apply_settings(new_settings)

        print("[OK] CARLA synchronous mode açıldı.")
        print(
            f"[INFO] Sabit simülasyon adımı: "
            f"{FIXED_DELTA_SECONDS} saniye"
        )

        # Ego aracını oluştur.
        vehicle = spawn_vehicle(
            world=world,
            vehicle_name=vehicle_name,
        )

        controller = LaneFollowController()

        data_collector = DataCollector(
            output_folder="data/runs",
            save_every_n_frames=SAVE_EVERY_N_FRAMES,
        )

        data_collector.start(
            world=world,
            vehicle=vehicle,
        )

        print("[OK] Araç ve sensör sistemi hazır.")
        print("[INFO] Çıkmak için CTRL+C tuşlarına bas.")

        while True:
            frame_id = world.tick()

            # Araç konumu, hızı ve waypoint bilgilerini al.
            data = get_datas(
                world=world,
                vehicle=vehicle,
                lookahead_distance=4.0,
            )

            # Kontrolcü gaz, fren ve direksiyon üretir.
            control = controller.run_step(data)

            # Kontrol komutunu araca uygula.
            vehicle.apply_control(control)

            # CARLA spectator kamerasını aracın arkasında tut.
            update_spectator_camera(
                world=world,
                vehicle=vehicle,
            )

            # Yalnızca kaydetmek istediğimiz temel araç bilgilerini al.
            vehicle_data = get_vehicle_data(
                data=data,
                control=control,
            )

            # Aynı frame'e ait sensör verilerini topla ve kaydet.
            data_collector.collect(
                frame_id=frame_id,
                vehicle_data=vehicle_data,
            )

    except KeyboardInterrupt:
        print("\n[INFO] Program kullanıcı tarafından durduruldu.")

    except Exception as error:
        print(f"[ERROR] Program çalışırken hata oluştu: {error}")

    finally:
        # Önce sensörleri kapat.
        if data_collector is not None:
            data_collector.stop()

        if vehicle is not None:
            try:
                vehicle.destroy()
                print("[OK] Araç silindi.")
            except Exception as error:
                print(f"[WARN] Araç silinemedi: {error}")

        if world is not None and original_world_settings is not None:
            try:
                world.apply_settings(original_world_settings)
                print("[OK] CARLA world ayarları geri yüklendi.")
            except Exception as error:
                print(
                    f"[WARN] World ayarları geri yüklenemedi: {error}"
                )


if __name__ == "__main__":
    main()