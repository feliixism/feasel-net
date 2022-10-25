from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="FeaSel-Net",
    version="0.0.9",
    license='MIT',
    author="Felix Fischer",
    author_email="felix.fischer@ito.uni-stuttgart.de",
    description="A Keras callback package for iteratively selecting the most influential input nodes during training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.tik.uni-stuttgart.de/FelixFischer/FeaSel-Net.git",
    keywords=['feature selection', 'neural networks' 'machine learning'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
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