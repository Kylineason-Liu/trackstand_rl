from setuptools import setup
import os
from glob import glob

package_name = "bike_balance"

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
    install_requires=["setuptools", "numpy", "scipy"],
    entry_points={
        "console_scripts": [
            "balance_node = bike_balance.balance_node:main",
            "calibrate_imu = bike_balance.calibration_node:main",
        ],
    },
)
