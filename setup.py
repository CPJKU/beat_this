from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    long_description = f.read()
setup(
    name="beat_this",
    version="0.1",
    description="Beat This! beat tracker",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Francesco Foscarin, Jan Schlüter",
    url="https://github.com/CPJKU/beat_this",
    license="MIT",
    packages=find_packages(),
    scripts=["bin/beat_this"],
    install_requires=[
        "numpy",
        "torch>=2",
        "torchaudio",
        "einops",
        "rotary_embedding_torch",
        "soxr",
    ],
    python_requires='>=3',
)
