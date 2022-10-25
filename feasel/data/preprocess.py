"""
feasel.data.preprocess
======================

This is a preprocessing module with several functions for data reshaping and
target encodings.
"""

import numpy as np
from scipy.signal import resample as scipy_resample

def get_indices(array, type='categorical'):
  """
  Provides a list with the indices of an array that indicates where to find
  the data for each particular class or label.

  Parameters
  ----------
  array : ndarray
    The array with label entries.
  type : str
    A param

  Returns
  -------
  index_l : list
      A list with the indices.

  """
  if type == 'categorical':
    array = np.array(array, ndmin=2)

  elif type == 'one-hot':
    array = np.array(array, ndmin=2).T

  else:
    raise TypeError(f"{type} is an invalid parameter for 'type'.")

  labels, idx = np.unique(array, axis=0, return_index=True)
  labels = labels[idx.argsort()]

  index_l = []
  for i, label in enumerate(labels):
    index_l.append(np.argwhere(np.all(array == label, axis=-1)))

  return index_l

def categorical(array):
  """
  Converts an one-hot encoded array into a categorical array. If the array is
  categorical already, it will be manipulated such that the neural network can
  interpret the labels, i.e. it will rename the classes into an enumeration of
  integers starting from 0 to the number of classes n_c-1 (is obsolete when
  already happened).

  Parameters
  ----------
  array : ndarray
    The array with label entries.

  Returns
  -------
  categorical : ndarray
    An array of categorical labels.
  """
  # it will convert the label array in an interpretable label array for the
  # neural network

  # provides a list of indices
  l_indices = get_indices(array, type='categorical')

  # creates the empty categorical array:
  categorical = np.zeros(len(array))
  for i, indices in enumerate(l_indices):
    # array will be the i-th class in the specific indices
    categorical[indices] = i

  return categorical

def one_hot(array):
  """
  Converts a categorical array into an one-hot encoded array. If the array is
  one-hot-encoded already, it will be returned unmanipulated.

  Parameters
  ----------
  array : ndarray
    The array with label entries.

  Returns
  -------
  one_hot : ndarray
    An array of the one-hot encoded labels.
  """
  l_indices = get_indices(array, type='one-hot') # provides a list of indices

  # chek if one-hot already:
  if (array.ndim != 2) and (np.any(np.unique(array) != np.array([0,1]))):
    # creates the empty one-hot encoded array:
    one_hot = np.zeros([len(array), len(l_indices)])
    for i, indices in enumerate(l_indices):
      # array will be one, only if it is found in the i-th class
      one_hot[indices, i] = 1
  else:
    one_hot = array
  return one_hot

def min_max(X, axis=-1, a_max=1., a_min=0., return_scale=False):
  """
  Scales the original array such that it fits in between a maximum value a_max
  and a minimum value a_min. The old values are described by A_max and A_min.

  Parameters
  ----------
  X : ndarray
    The original data array.
  axis : int, optional
    The axis along which the scaling is undertaken. If 'None', it will scale
    the complete array. The default is -1.
  a_max : float, optional
    The new maximum value of the data after scaling. The default is 1.
  a_min : float, optional
    The new minimum value of the data after scaling. The default is 0.
  return_scale : bool
    If True, it will also return the scaling factors in a dictionary.

  Returns
  -------
  array_n : ndarray
    The min-max normalized array.

  """
  if len(X) > 1:
    A_max = np.amax(X, axis=axis, keepdims=True) # previous max entry
    A_min = np.amin(X, axis=axis, keepdims=True) # previous min entry

    scale = (a_max - a_min) / (A_max - A_min) # scaling factor
    offset = a_max - scale * A_max # offset factor

    X_min_max = scale * X + offset # new normalized array

    if return_scale:
      scale = {'scale': scale, 'offset': offset}
      return X_min_max, scale
  else:
    X_min_max = X

  return X_min_max

def standardize(X, axis = -1, return_scale=False):
  """
  Calculates the standardized score of the sample features.

    X_z = (x - x_bar) / s

  where x_bar is the arithmetic mean and s the standard deviation for each
  feature. The features are described by the given axis.

  Parameters
  ----------
  X : ndarray (float)
    The original data array.
  axis : int, optional
    Determines the axis of the features. The default is -1.
  return_scale : bool
    If True, it will also return the scaling factors in a dictionary.

  Returns
  -------
  X_z : ndarray (float)
    The standard score normalized array.

  """
  if len(X) > 1:
    x_bar = np.mean(X, axis=axis, keepdims=True) # arithmetic mean
    s = np.std(X, axis=axis, keepdims=True) # standard deviation

    X_z = (X - x_bar) / s

    if return_scale:
      scale = {'x_bar': x_bar, 's': s}
      return X_z, scale

  else:
    X_z = X

  return X_z

def mean_centered(X, axis=0, return_shift=False):
  """
  Mean-centering of the original (normalized) data.

  Parameters
  ----------
  X : ndarray
    Input array.
  axis : int, optional
    The axis along which the average is taken. The default is -1.
  return_shift : bool, optional
    Determines whether the shift (mean) is returned or not. The default is
    False.

  Returns
  -------
  X_c : ndarray
    The mean-centered array.

  """
  if len(X) > 1:
    x_bar = np.mean(X, axis=axis, keepdims=True) # arithmetic mean
    X_c = X - x_bar
    if return_shift:
      return X_c, x_bar

  else:
    X_c = X

  return X_c

def resample(arr, num, axis=0):
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

def remove_outliers(X, y=None, sigma=2., axis=0):
  """
  Removes outliers from a data array.

  Parameters
  ----------
  X : ndarray
    The data array.
  sigma : float, optional
    The deviation defining the confidence interval. The default is :math:`2`.
  axis : int, optional
    The axis along the average and standarddeviation is calculated. The default
    is :math:`0`.

  Returns
  -------
  X_r(, y_r) : ndarray
    The new array without the outliers (with label array in a tuple).

  """
  X_r = None
  if not isinstance(y, type(None)):
    classes = np.unique(y)
    y_r = None
  else:
    classes = [None]

  for c in classes:
    if c is None:
      mask = np.ones(len(X))
    else:
      mask = y == c

    X_c = X[mask]

    std = np.std(X_c, axis=axis)
    mu = np.average(X_c, axis=axis)

    maskd = np.all(-std * sigma <= X_c - mu, axis=1)
    masku = np.all(X_c - mu <= std * sigma, axis=1)

    mask_c = maskd * masku
    try:
      X_r = np.append(X_r, X_c[mask_c], axis=0)
      if not isinstance(y, type(None)):
        y_r = np.append(y_r, y[mask][mask_c], axis=0)
    except:
      X_r = X_c[mask_c]
      if not isinstance(y, type(None)):
        y_r = y[mask][mask_c]

  if not isinstance(y, type(None)):
    return X_r, y_r
  else:
    return X_r