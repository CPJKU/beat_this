from setuptools import setup, find_packages

setup(
    name="beat_this",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "librosa",
        "pytorch-lightning"
    ],
)