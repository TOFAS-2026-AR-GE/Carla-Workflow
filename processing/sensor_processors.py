import numpy as np


def process_frame(packet):
    """
    Aynı frame'e ait bütün sensör verilerini işler.

    packet.sensor_data örneği:

    {
        "rgb_camera": ...,
        "lidar": ...,
        "gnss": ...,
        "imu": ...,
        "radar_front": ...,
        "radar_rear": ...,
        "radar_left": ...,
        "radar_right": ...
    }
    """

    processed_data = {}

    for sensor_name, raw_data in packet.sensor_data.items():
        processed_data[sensor_name] = process_sensor(
            sensor_name,
            raw_data,
        )

    return processed_data


def process_sensor(sensor_name, raw_data):
    """
    Sensör adına bakarak uygun işleme fonksiyonunu çağırır.
    """

    if sensor_name == "rgb_camera":
        return process_rgb_camera(raw_data)

    if sensor_name == "semantic_camera":
        return process_semantic_camera(raw_data)

    if sensor_name == "lidar":
        return process_lidar(raw_data)

    # radar_front, radar_rear, radar_left ve radar_right
    # isimlerinin tamamını yakalar.
    if sensor_name.startswith("radar_"):
        return process_radar(raw_data)

    if sensor_name == "gnss":
        return process_gnss(raw_data)

    if sensor_name == "imu":
        return process_imu(raw_data)

    raise ValueError(
        f"Bilinmeyen sensör adı: {sensor_name}"
    )


def process_rgb_camera(image):
    """
    CARLA kamera görüntüsünü RGB NumPy görüntüsüne çevirir.

    Çıktı:
        shape = (height, width, 3)
    """

    raw_array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    bgra_image = raw_array.reshape(
        image.height,
        image.width,
        4,
    )

    # Alpha kanalını kaldır.
    bgr_image = bgra_image[:, :, :3]

    # BGR -> RGB
    rgb_image = bgr_image[:, :, ::-1]

    return rgb_image.copy()


def process_semantic_camera(image):
    """
    Semantic kamera görüntüsünden sınıf ID'lerini çıkarır.
    """

    raw_array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    bgra_image = raw_array.reshape(
        image.height,
        image.width,
        4,
    )

    class_ids = bgra_image[:, :, 2]

    return class_ids.copy()


def process_lidar(lidar_data):
    """
    LiDAR verisini NumPy dizisine çevirir.

    Her satır:
        [x, y, z, intensity]
    """

    points = np.frombuffer(
        lidar_data.raw_data,
        dtype=np.float32,
    )

    points = points.reshape(-1, 4)

    return points.copy()


def process_radar(radar_data):
    """
    Bir radardan gelen bütün radar noktalarını işler.

    Her radar noktası:

    {
        "distance": metre,
        "relative_velocity": metre/saniye,
        "azimuth": yatay açı,
        "altitude": dikey açı
    }
    """

    radar_points = []

    for detection in radar_data:
        radar_point = {
            "distance": float(detection.depth),
            "relative_velocity": float(detection.velocity),
            "azimuth": float(detection.azimuth),
            "altitude": float(detection.altitude),
        }

        radar_points.append(radar_point)

    return radar_points


def process_gnss(gnss_data):
    """
    GNSS verisini sözlüğe çevirir.
    """

    return {
        "lat": float(gnss_data.latitude),
        "lon": float(gnss_data.longitude),
        "alt": float(gnss_data.altitude),
    }


def process_imu(imu_data):
    """
    IMU verisini sözlüğe çevirir.
    """

    return {
        "accelerometer": {
            "x": float(imu_data.accelerometer.x),
            "y": float(imu_data.accelerometer.y),
            "z": float(imu_data.accelerometer.z),
        },
        "gyroscope": {
            "x": float(imu_data.gyroscope.x),
            "y": float(imu_data.gyroscope.y),
            "z": float(imu_data.gyroscope.z),
        },
        "compass": float(imu_data.compass),
    }