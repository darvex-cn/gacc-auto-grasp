from setuptools import setup


setup(
    name='graspnet-adapter',
    version='0.1.0',
    description='Reusable inference API for GraspNet baseline',
    packages=['graspnet_adapter'],
    package_dir={'graspnet_adapter': '.'},
    python_requires='>=3.8',
    install_requires=['numpy'],
)
