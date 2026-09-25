"""Deployment adapter around the unmodified YahBoom IMU library."""

from __future__ import annotations

import time

from .YbImuLib import YbImuSerial as _VendorYbImuSerial


class YbImuSerial(_VendorYbImuSerial):
    """Add deployment freshness tracking without modifying the vendor package."""

    def __init__(self, port, debug=False):
        super().__init__(port, debug=debug)
        self.last_valid_frame_monotonic = 0.0

    def _parse_data(self, ext_type, ext_data):
        super()._parse_data(ext_type, ext_data)
        self.last_valid_frame_monotonic = time.monotonic()
