"""Deployment adapter around the unmodified YahBoom IMU library."""

from __future__ import annotations

import math
import struct
import threading
import time

from .YbImuLib.YbImuSerialLib import YbImuSerial as _VendorYbImuSerial


class YbImuSerial(_VendorYbImuSerial):
    """Snapshot gyro/quaternion separately without changing the vendor driver."""

    def __init__(self, port, debug=False):
        super().__init__(port, debug=debug)
        self._sample_lock = threading.Lock()
        self._gyro_at = self._quaternion_at = 0.0
        self._gyro = self._quaternion = None

    def _parse_data(self, ext_type, ext_data):
        # The vendor receiver invokes this only after validating the checksum.
        # A valid short/unknown frame must not kill that receiver or refresh IMU age.
        with self._sample_lock:
            try:
                super()._parse_data(ext_type, ext_data)
            except (struct.error, IndexError, ValueError):
                return
            if ext_type == self.FUNC_REPORT_IMU_RAW:
                gyro = tuple(self.get_gyroscope_data())
                if all(map(math.isfinite, gyro)):
                    self._gyro, self._gyro_at = gyro, time.monotonic()
            elif ext_type == self.FUNC_REPORT_IMU_QUAT:
                quaternion = tuple(self.get_imu_quaternion_data())
                norm = math.hypot(*quaternion)
                if all(map(math.isfinite, quaternion)) and math.isfinite(norm) and norm > 0.5:
                    self._quaternion, self._quaternion_at = quaternion, time.monotonic()

    def get_timed_sample(self):
        """Return immutable (oldest channel parse time, quaternion wxyz, gyro rad/s).

        Other packets cannot refresh these times. The publisher requires progress
        in their minimum, so a frozen channel cannot be republished indefinitely.
        Host parse time is not a wire-arrival or hardware acquisition timestamp.
        """
        with self._sample_lock:
            if self._gyro is None or self._quaternion is None:
                return None
            return min(self._gyro_at, self._quaternion_at), self._quaternion, self._gyro
