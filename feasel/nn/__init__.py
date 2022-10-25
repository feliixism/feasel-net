"""
*********************************************
Non-linear transformations (:mod:`feasel.nn`)
*********************************************
"""

from . import analysis
from . import architectures
from . import tfcustom

from .architectures import Base
from .architectures import FSDNN
from .architectures import FCDNN

__all__ = ['Base', 'FSDNN', 'FCDNN']