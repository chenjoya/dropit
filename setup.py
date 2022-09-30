from setuptools import setup, find_packages

setup(
    name='dropit',
    packages=find_packages(where=("dropit")),
    version='0.1.0',
    install_requires=[
        "timm>=0.6.7",
        "torch>=1.12.1",
        "torchvision>=0.13.1"
    ]
)