from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'yolov5'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['msg/*']
    },
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/yolov5_launch.py']),
        ('share/' + package_name, ['config/cam2image.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sithu',
    maintainer_email='sithuaung@globalwalkers.co.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_node = yolov5.yolov5_node:main'
        ],
    },
)
