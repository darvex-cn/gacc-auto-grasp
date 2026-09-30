from setuptools import find_packages, setup

package_name = 'hand_eye_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daron',
    maintainer_email='daron@todo.todo',
    description='Eye-in-hand calibration with a ChArUco target.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'calibration_node = hand_eye_calibration.calibration_node:main'
        ],
    },
)
