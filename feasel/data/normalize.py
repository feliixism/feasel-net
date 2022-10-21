"""
feasel.data.normalize
=====================
"""

import numpy as np

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
      X_min_max = np.nan_to_num(X_min_max)
      return X_min_max, scale
  else:
    X_min_max = X

  X_min_max = np.nan_to_num(X_min_max)

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
    x_bar = np.nanmean(X, axis=axis, keepdims=True) # arithmetic mean
    s = np.nanstd(X, axis=axis, keepdims=True) # standard deviation

    X_z = (X - x_bar) / s
    X_z = np.nan_to_num(X_z)

    if return_scale:
      scale = {'x_bar': x_bar, 's': s}
      return X_z, scale

  else:
    X_z = np.nan_to_num(X)

  return X_z