"""
feasel.linear.svd
=================
"""

import numpy as np
from scipy.linalg import svd
 
class SVD:
    def __init__(self, data, thershold = None):
        self.data = data
        self._decomposed = False
    
    def __call__(self):
        if not self._decomposed:
            self.decompose()
        return self.U, self.Sigma, self.Vh
    
    def decompose(self):
        """
        Apply Singular Value Decomposition
        
        X = U * Sigma * V^H
        
        where U is a unitary matrix (left-singular matrix) 
        and Sigma a diagonal matrix with singular values in 
        descending order. V^H is the adjunct matrix of 
        another unitary matrix (right-singular matrix). 
        
        Visually explained, this is nothing else than the
        decomposition of any arbitrary dataset into a linear 
        combination of 

        Returns
        -------
        None.

        """
        self.U, self.sigma, self.Vh = svd(self.data, full_matrices = False)
        self.Sigma = np.diag(self.sigma)
        self.matrix_rank = len(np.argwhere(self.sigma > np.finfo(float).eps))
        self._decomposed = True