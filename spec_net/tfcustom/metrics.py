import numpy as np

def cross_entropy(P, Q):
    H = -np.sum(P * np.log(Q), axis = -1)
    return H

def entropy(P, Q):
    H = -np.sum(Q * np.log(Q), axis = -1)
    return H