from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    long_description = f.read()
setup(
    name="beat-this",
    version="0.1",
    description="Beat This! beat tracker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Francesco Foscarin, Jan Schlüter",
    url="https://github.com/CPJKU/beat_this",
    license="MIT",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["beat_this=beat_this.cli:main"],
    },
    install_requires=[
        "numpy",
        "torch>=2",
        "torchaudio",
        "einops",
        "rotary-embedding-torch",
        "soxr",
    ],
    python_requires=">=3",
)
