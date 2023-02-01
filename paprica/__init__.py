"""
Initialization module for pAPRica (Pipelines for Adaptive Particle Representation Image Compositing and Analysis).

**pAPRica** is composed of submodules that aims at tackling different tasks such as *stiching*, *segmenting* or
*viewing* your data.

More information about APR can be found in the original publication (https://www.nature.com/articles/s41467-018-07390-9).

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"

from . import loader, parser, stitcher, viewer, segmenter, atlaser, converter, runner, batcher
from .atlaser import tileAtlaser
from .converter import tileConverter
from .parser import tileParser, baseParser, autoParser
from .runner import clearscopeRunningPipeline
from .segmenter import tileSegmenter, multitileSegmenter, tileTrainer
from .stitcher import tileStitcher, channelStitcher
from .viewer import tileViewer

__all__ = ['loader', 'parser', 'stitcher', 'viewer', 'segmenter', 'atlaser', 'converter', 'runner', 'batcher']
