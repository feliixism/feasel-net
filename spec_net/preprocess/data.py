import numpy as np
from scipy.signal import resample as scipy_resample

def resample(arr, num, axis = 0):
    """
    Resamples the data along specified axis.

    Parameters
    ----------
    arr : float
        Array that shall be resampled.
    num : int
        Desired number of sample points. The default is 0.2.
    axis : int, optional
        Axis along which the data is resampled. The default is 0.
    

    Returns
    -------
    resampled : float
        Resampled array of original data.

    """
    resampled = scipy_resample(arr, num, axis = axis)
    return resampled

def scatter_matrix(arr, rowvar = True):
    if not rowvar:
        arr = arr.T
    mu = np.mean(arr, axis = 1, keepdims = True)
    X = arr - mu
    S = X @ X.T
    return S

def covariance(arr, rowvar = True):
    """
    Calculates the covariance of an array.
    The covariance shows how much a variable is related to another.

    Parameters
    ----------
    arr : float
        Input array of the raw data.
    rowvar : boolean, optional
        Needs to be set to 'False' if the variables of interest are given in the rows. The default is True.

    Returns
    -------
    cov : float
        Covariance Matrix.

    """
    if not rowvar:
        n, m = arr.shape
    else:
        m, n = arr.shape
    S = scatter_matrix(arr, rowvar)
    cov = S / (n - 1)
    return cov

def correlation(arr, rowvar = True):
    """
    Calculates the Pearson correlation coefficients of the input matrix.
    Pearsons correlation only refers to linear relations between two variables.
    All correlation coefficents have values in the range between 0 and 1 where 1 denotes the highest correlation.

    arr : float
        Input array of the raw data.
    rowvar : boolean, optional
        Needs to be set to 'False' if the variables of interest are given in the rows. The default is True.

    Returns
    -------
    corr : float
        Correlation Matrix.

    """
    cov = covariance(arr, rowvar)
    D_inv = np.diag(1 / np.sqrt(np.diag(cov)))
    corr = D_inv @ cov @ D_inv
    return corr


