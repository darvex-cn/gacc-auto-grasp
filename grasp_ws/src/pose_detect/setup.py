from glob import glob

from setuptools import find_packages, setup

package_name = 'pose_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daron',
    maintainer_email='daron@todo.todo',
    description='ROS 2 GraspNet pose detection node.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pose_detect_node = pose_detect.pose_detect_node:main',
            'segmentation_test_node = pose_detect.segmentation_test_node:main',
        ],
    },
    options={
        'build_scripts': {
            'executable': "/home/daron/miniconda3/envs/graspnet/bin/python",  # 使用虚拟环境的Python
        }
    }
)
