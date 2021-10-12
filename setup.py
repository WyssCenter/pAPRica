from setuptools import setup, find_packages

LONG_DESCR = open('readme.md').read()
LICENSE = open('license.txt').read()

setup(
    name='pipapr',
    version='0.0.1',
    setup_requires='setuptools',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        #'PyLibAPR',
        'tqdm',
        'pandas',
        'scikit-image',
        'opencv-contrib-python-headless',
        'dill',
        'matplotlib',
        'napari',
        'allensdk',
        'sparse',
        'brainreg'
    ],
    description='APR-based image processing pipeline for brain data',
    long_description=LONG_DESCR,
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/wysscenter/apr_pipelines',
    author='Jules Scholler, Joel Jonsson',
    author_email='jules.scholler@wysscenter.ch, jonsson@mpi-cbg.de',
    license=LICENSE,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3'
    ],
    keywords='APR, adaptive, image, representation, processing, pipeline, registration, segmentation',
)
