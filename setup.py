from setuptools import setup, find_packages

setup(
    name="pinqi",
    packages=find_packages(exclude=[]),
    version="0.0.1",
    license="MIT",
    description="PINQI: E2E QMRI",
    author="Felix Zimmermann",
    author_email="fzimmermann89@gmail.com",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "mri",
        "qmri",
    ],
    install_requires=[
        "einops>=0.6",
        "torch>=1.12,<2.1",
        "numpy>1.7",
        "h5py",
        "pytorch-lightning>=1.8",
        "scikit-image",
        "torchvision",
        "matplotlib",
        "neptune-client",
        "tqdm",
        "pytorch-msssim",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
