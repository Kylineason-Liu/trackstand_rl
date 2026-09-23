from setuptools import setup
import os
from glob import glob

package_name = 'remote_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    entry_points={
        'console_scripts': [
            'sbus_reader = remote_controller.sbus_reader:main',
            'motor_controller = remote_controller.motor_controller:main',
            'servo_controller = remote_controller.servo_controller:main',
        ],
    },
)
