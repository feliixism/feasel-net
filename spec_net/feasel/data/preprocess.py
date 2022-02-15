"""
This is a preprocessing module with several functions for data reshaping and
target encodings.
"""

import numpy as np

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
  if (array.ndim != 2) and (np.unique(array) != [0,1]):
    # creates the empty one-hot encoded array:
    one_hot = np.zeros([len(array), len(l_indices)])
    for i, indices in enumerate(l_indices):
      # array will be one, only if it is found in the i-th class
      one_hot[indices, i] = 1
  else:
    one_hot = array
  return one_hot

def shape(X, y):
  """
  Automatically reshapes the input data to match the input layer
  dimensionality.

  Parameters
  ----------
  X : ndarray
    Input data.
  y : ndarray
    Target array.

  Returns
  -------
  None.

  """
  SHAPE = {2: _ann,
           3: _1d_conv,
           4: _2d_conv,
           5: _3d_conv}
  dims = X.ndim
  SHAPE[dims](X, y)


def _ann(X, y):
  """
  Automatically reshapes the input data to match Dense type layers.

  Parameters
  ----------
  X : ndarray
    Input data.
  y : ndarray
    Target array.

  Returns
  -------
  None.

  """
  if len(X.shape) != 2:
    old_shape = X.shape
    X = X.reshape([len(y), int(len(X.flatten()) / len(y))])
    print(f"Shape {old_shape} is converted to shape {X.shape}.")

  else:
    print(f"Shape {X.shape} is already correct.")

  return X

def _1d_conv(X, y, features=1):
  """
  Automatically reshapes the input data to match Dense type layers.

  Parameters
  ----------
  X : ndarray
    Input data.
  y : ndarray
    Target array.
  features : int
    Number of features in the third dimension (e.g. color channels). The
    default is '1'.

  Returns
  -------
  None.

  """
  if len(X.shape) != 3:
    old_shape = X.shape
    X = X.reshape([len(y), int(len(X.flatten()) / (len(y) * features)), features])
    print(f"Shape {old_shape} is converted to shape {X.shape}.")

  else:
    print(f"Shape {X.shape} is already correct.")

  return X

def _2d_conv(X, y, features=None):
  """
  Not implemented yet.

  Returns
  -------
  None.

  """
  return

def _3d_conv(X, y, features=None):
  """
  Not implemented yet.

  Returns
  -------
  None.

  """
  return

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
