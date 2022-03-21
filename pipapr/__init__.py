"""
Initialization module for pipapr (PIPeline for Adaptive Particles Representation).

**pipapr** is composed of submodules that aims at tackling different tasks such as *stiching*, *segmenting* or
*viewing* your data.

More information about APR can be found in the original publication (https://www.nature.com/articles/s41467-018-07390-9).

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from . import loader, parser, stitcher, viewer, segmenter, atlaser, converter, runner

from .parser import tileParser, baseParser
from .converter import tileConverter
from .atlaser import tileAtlaser
from .segmenter import tileSegmenter, multitileSegmenter, tileTrainer
from .viewer import tileViewer
from .stitcher import tileStitcher, channelStitcher
from .runner import clearscopeRunningPipeline

# from pipapr import *
__all__ = ['loader', 'parser', 'stitcher', 'viewer', 'segmenter', 'atlaser', 'converter', 'runner']
