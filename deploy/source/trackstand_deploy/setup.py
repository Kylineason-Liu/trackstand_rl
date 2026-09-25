import os
from glob import glob

from setuptools import find_packages, setup


package_name = "trackstand_deploy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
    ],
    install_requires=["setuptools", "numpy", "onnxruntime", "python-can", "pyserial", "smbus2"],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "imu_publisher = trackstand_deploy.imu_publisher:main",
            "motor_io = trackstand_deploy.motor_io:main",
            "rc_kickstand = trackstand_deploy.rc_kickstand:main",
            "kickstand_driver = trackstand_deploy.kickstand_driver:main",
            "policy_controller = trackstand_deploy.policy_controller:main",
        ],
    },
)
