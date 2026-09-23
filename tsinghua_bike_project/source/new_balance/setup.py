from setuptools import setup
import os
from glob import glob

package_name = "new_balance"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.py")),
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "scipy", "python-can", "pyserial"],
    entry_points={
        "console_scripts": [
            "balance_node = new_balance.main_balance:main",
            "roll_publisher = new_balance.roll_publisher:main",
            "steer_publisher = new_balance.steer_publisher:main",
            "rear_publisher = new_balance.rear_publisher:main",
            "imu_sign_check = new_balance.imu_sign_check:main",
            "imu_calibrate = new_balance.imu_zero_bias_calibration:main",
            "rc_publisher = new_balance.rc_publisher:main",
            "cascaded_balance_node = new_balance.cascaded_main_balance:main",
        ],
    },
)
