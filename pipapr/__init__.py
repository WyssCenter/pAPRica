"""
Initialization module for pipapr (PIPeline for Adaptive Particles Representation).

**pipapr** is composed of submodules that aims at tackling different tasks such as *stiching*, *segmenting* or
*viewing* your data.

More information about APR can be found in the original publication (https://www.nature.com/articles/s41467-018-07390-9).

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from . import loader, parser, stitcher, viewer, segmenter, atlaser, converter, runner
from .atlaser import tileAtlaser
from .converter import tileConverter
from .parser import tileParser, baseParser
from .runner import clearscopeRunningPipeline
from .segmenter import tileSegmenter, multitileSegmenter, tileTrainer
from .stitcher import tileStitcher, channelStitcher
from .viewer import tileViewer

__all__ = ['loader', 'parser', 'stitcher', 'viewer', 'segmenter', 'atlaser', 'converter', 'runner']
