"""
The data can either be prepared to suit the needs of *classification* tasks or
these of *regression*. All other submodules are used to pre-process data
(normalization, and filering) or to evaluate previously done classifications or
regressions (metrics).
"""

from . import classification
from . import filter
from . import normalize
from . import metrics
from . import preprocess
from . import regression