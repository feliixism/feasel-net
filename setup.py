from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="FeaSel-Net",
    version="0.0.1",
    author="Felix Fischer",
    author_email="felix.fischer@ito.uni-stuttgart.de",
    description="A Keras callback package for iteratively selecting the most influential input nodes during training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.tik.uni-stuttgart.de/FelixFischer/FeaSel-Net.git",
    packages=['spec_net'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['numpy',
                      'h5py',
                      'tensorflow>=2.0',
                      'sklearn',
                      'mplcursors',
                      'matplotlib>=3.5',
                      ]
)