"""
*********************************************
Linear Transformations (:mod:`feasel.linear`)
*********************************************
"""

from . import base
from . import lda
from . import pca
from . import svd

from .base import ModelContainer
from .lda import LDA
from .pca import PCA
from .svd import SVD

__all__ = ['ModelContainer', 'LDA', 'PCA', 'SVD']