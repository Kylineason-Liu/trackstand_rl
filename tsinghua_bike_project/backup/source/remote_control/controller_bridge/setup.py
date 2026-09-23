from setuptools import setup
import os

package_name = 'controller_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'joy_to_cmd = controller_bridge.joy_to_cmd:main',
        ],
    },
)
