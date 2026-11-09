# -*- coding: utf-8 -*-

__version__ = "2.0"

# Fix numpy/cobra compatibility issue for numpy >= 1.20
# Suppress the FutureWarning about np.object
import warnings
import sys
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    if not hasattr(np, 'object'):
        np.object = object

# Import bayflux core components
from bayflux.core import *
