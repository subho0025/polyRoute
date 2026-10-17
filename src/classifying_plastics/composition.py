"""
Composition Estimator Bridge Module.
Provides seamless import: `from composition import estimate_composition`
Delegates to `src/comp_analyzer/estimator.py`.
"""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_SRCS = [
    os.path.join(_DIR, "src"),
    os.path.join(os.path.dirname(_DIR), "src"),
    os.path.join(os.path.dirname(os.path.dirname(_DIR)), "src"),
]

for src_dir in _CANDIDATE_SRCS:
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from comp_analyzer import *
from comp_analyzer.estimator import estimate_composition
from comp_analyzer.loaders import load_confusion
