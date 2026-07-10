import numpy as np

def process_frame(packet):
    processed_data = {}
    for sensor_name, raw_data in packet.sensor_data.items():
        processed_data[sensor_name] = process_sensor(
            sensor_name,
            raw_data
        )

def process_sensor(sensor_name, data):
    if sensor_name == "gnss":
        return process_gnss(data)
    if sensor_name == "imu":
        return process_imu(data)

def process_rgb_cam(image):
    #bgr yi rgb çevirip np dizisi haline getiriyoruz
    raw_data = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )
    bgra_image = raw_data.reshape(
        image.height,
        image.width,
        4,
    )
    bgr_image = bgra_image[:, :, :3]
    rgb_image = bgr_image[:, :, ::-1]
    
    return rgb_image.copy()


def process_semantic_cam(image):
    #sınıf id alır h x w output
    raw_data = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )
    
    bgra_image = raw_data.reshape(
        image.height,
        image.width,
        4
    )
    
    labels = bgra_image[:, :, 2]
    return labels.copy()


def process_lidar(lidar_data):
    #n x 4 numpy dizisi (x,y,z, intensity)
    points = np.frombuffer(
        lidar_data.raw_data,
        dtype=np.float32,
    )

    if len(points) % 4 != 0:
        raise ValueError(
            "LiDAR verisi x, y, z, intensity formatına uygun değil."
        )
    return points.reshape(-1, 4).copy()

def process_gnss(gnss_data):
    #dict donusturur
    return {
        "lat": float(gnss_data.latitude),
        "lon": float(gnss_data.longitude),
        "alt": float(gnss_data.altitude) 
    }

def process_imu(imu_data):
    #dict donusturur
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
