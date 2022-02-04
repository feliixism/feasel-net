import numpy as np

def get_indices(array):
    """
    Provides a list with the indices of an array that indicates where to find
    the data for each particular class or label.

    Parameters
    ----------
    array : ndarray
        The array with label entries.

    Returns
    -------
    l_indices : list
        A list with the indices.

    """
    labels, idx = np.unique(array, axis=0, return_index=True)
    labels = labels[idx.argsort()]

    if len(array.shape) == 1:
      array = np.array(array, ndmin=2).T

    l_indices = []
    for i, label in enumerate(labels):
        l_indices.append(np.argwhere(np.all(array == label, axis=-1)))

    return l_indices

def categorical(array):
    """
    Converts an one-hot encoded array into a categorical array. If the array is
    categorical already, it will be manipulated such that the neural network
    can interpret the labels, i.e. it will rename the classes into an
    enumeration of integers starting from 0 to the number of classes n_c - 1
    (is obsolete when already happened).

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
    l_indices = get_indices(array) # provides a list of indices

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
    l_indices = get_indices(array) # provides a list of indices

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
  SHAPE = {2: _ann,
           3: _1d_conv,
           4: _2d_conv,
           5: _3d_conv}
  dims = X.ndim
  SHAPE[dims](X, y)


def _ann(X, y):
    if len(x.shape) != 2:
        old_shape = x.shape
        x = x.reshape([len(y), int(len(x.flatten()) / len(y))])
        print(f"Shape {old_shape} is converted to shape {x.shape}.")
    else:
        print(f"Shape {x.shape} is already correct.")
    return x

def _1d_conv(X, y, features = 1):
  if len(x.shape) != 3:
    old_shape = x.shape
    x = x.reshape([len(y), int(len(x.flatten()) / (len(y) * features)), features])
    print(f"Shape {old_shape} is converted to shape {x.shape}.")
  else:
    print(f"Shape {x.shape} is already correct.")
  return x

def _2d_conv(X, y, features):
  return

def _3d_conv(X, y):
  return

def min_max(X, axis=-1, a_max=1., a_min=0.):
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

  Returns
  -------
  array_n : ndarray
    The min-max normalized array.

  """
  A_max = np.amax(X, axis=axis, keepdims=True) # previous max entry
  A_min = np.amin(X, axis=axis, keepdims=True) # previous min entry

  scale = (a_max - a_min) / (A_max - A_min) # scaling factor
  offset = a_max - scale * A_max # offset factor

  X_min_max = scale * X + offset # new normalized array

  return X_min_max

def standardize(array, axis = -1):
  """
  Calculates the standardized score of the sample features.

    z = (x - x_bar) / s

  where x_bar is the arithmetic mean and s the standard deviation for each
  feature. The features are described by the given axis.

  Parameters
  ----------
  arr : ndarray (float)
    The original data array.
  axis : int, optional
    Determines the axis of the features. The default is -1.

  Returns
  -------
  array_n : ndarray (float)
    The standard score normalized array.

  """
  x_bar = np.mean(array, axis=axis, keepdims=True) # arithmetic mean
  s = np.std(array, axis=axis, keepdims=True) # standard deviation

  array_n = (array - x_bar) / s

  return array_n
