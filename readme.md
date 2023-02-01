# pAPRica

Welcome to `pAPRica` (pipelines for Adaptive Particle Representation image compositing and analysis), a package based on the Adaptive Particle Representation (APR) to accelerate image processing and research involving imaging and microscopy.

<center>
<img src=./doc/images/pipeline_pv.png title="pipeline image" width="650"/>
</center>

`pAPRica` is built on top of:

- [LibAPR](https://github.com/AdaptiveParticles/LibAPR): the C++ backbone library
- [pyapr](https://github.com/AdaptiveParticles/pyapr/): a python wrapper for LibAPR including unique features

Briefly, `pAPRica` allows to accelerate processing of volumetric image data while lowering the hardware requirements. It
is made of several independent modules that are tailored to convert, stitch, segment, map to an atlas and visualize
data. `pAPRica` can work as a postprocessing tool and is also compatible with real time usage during acquisitions, 
enabling minimal lead time between imaging and analysis.

Tutorials and reference documentation is available at [WyssCenter.github.io/pAPRica/](https://wysscenter.github.io/pAPRica/).

## Requirements

The software should run on any operating system and python version 3.7 or higher. It is recommended that the system RAM is at least 3 times the size of a single tile, to allow voxels-to-APR conversion without decomposition of the input images.

## Be part of the community

If you have a project that you think would benefit from this software but aren't sure where to start, don't hesitate to contact us. We'd be happy to assist you.

If you encounter any problems or bugs :beetle:, please [file an issue](https://github.com/WyssCenter/pAPRica/issues).

## References:

If you use this pipeline in your research, please consider citing the following:

- [Efficient image analysis for large-scale next generation histopathology using pAPRica](https://www.biorxiv.org/content/10.1101/2023.01.27.525687v1) (bioRxiv preprint)
- [Adaptive particle representation of fluorescence microscopy images](https://www.nature.com/articles/s41467-018-07390-9) (Nature Communications)
