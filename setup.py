from setuptools import setup, find_packages

LONG_DESCR = open('readme.md').read()
LICENSE = open('license.txt', encoding='utf8').read()

setup(
    name='paprica',
    version='0.0.1',
    setup_requires='setuptools',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pyapr',
        'tqdm',
        'pandas',
        'scikit-image',
        'scikit-learn',
        'opencv-contrib-python-headless',
        'dill',
        'matplotlib',
        'napari',
        'allensdk',
        'sparse',
        'brainreg',
        'joblib',
        'pytest'
    ],
    description='APR-based image processing pipeline for microscopy data',
    long_description=LONG_DESCR,
    long_description_content_type='text/markdown',
    url='https://github.com/WyssCenter/APR-pipelines',
    author='Jules Scholler, Joel Jonsson',
    author_email='jules.scholler@wysscenter.ch, jonsson@mpi-cbg.de',
    license=LICENSE,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3'
    ],
    keywords='APR, adaptive, image, representation, processing, pipeline, registration, segmentation, stitching',
)
