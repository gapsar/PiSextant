"""Template for a new IMU driver -- copy it to imu_drivers/<your_sensor>.py.

The capture script calls open_imu(cfg) once, with the "imu" section of
configs/instrument.json as cfg, and then reads the returned object about
100 times per second. That object needs three attributes:

    acceleration   (ax, ay, az) in m/s^2, the RAW accelerometer reading
                   (no filtering, no gravity removal, no on-chip fusion):
                   at rest it is the gravity vector, |a| ~ 9.8
    gyro           (gx, gy, gz) in rad/s
    temperature    sensor temperature in degrees C (read every 5 s; return
                   None if the sensor has no thermometer)

One sample, as the capture logs it in imu.jsonl:

    {"t": 1792152031.2345, "gravity": [0.41, -9.66, 0.53], "gyro": [-0.0037, 0.0063, -0.0008]}

The axes can point in any direction: the calibration (configs/m_case.json)
finds the rotation from the IMU frame to the camera frame. But keep the
sensor's own axes -- do not swap or flip them in the driver after you have
calibrated, or the calibration becomes wrong.

Quick test (from the repo folder, in the capture venv):

    python -c "import imu_backends, json; c = json.load(open('configs/instrument.json'))['imu']; \
               imu = imu_backends.make_imu(c); print(imu.acceleration, imu.gyro, imu.temperature)"
"""


class TemplateImu:
    def __init__(self, cfg):
        # open the bus and the sensor here, e.g. with cfg["i2c_bus"], cfg["address"]
        raise NotImplementedError("write the driver for your sensor")

    @property
    def acceleration(self):
        return (0.0, 0.0, 9.81)

    @property
    def gyro(self):
        return (0.0, 0.0, 0.0)

    @property
    def temperature(self):
        return None


def open_imu(cfg):
    return TemplateImu(cfg)
