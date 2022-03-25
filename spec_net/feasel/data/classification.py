import numpy as np
from sklearn.model_selection import train_test_split

from ..parameters import DataParams
from . import preprocess as pre

class DataContainer:
  def __init__(self, X, y=None, features=None, **kwargs):
    """
    This is a data object for the classification with neural networks based on
    the tensorflow and keras frameworks. Within this class all necessary
    pre-processing and data-handling tasks can be undertaken.

    Parameters
    ----------
    X : ndarray
      Input array for the training of neural networks.
    y : ndarray, optional
      Target array for the loss estimation if provided. The default is None.
    features : ndarray, optional
      Defines the feature's names and must have the same size as the feature
      shape. If None, the features are iterated from 0 to the number of
      features-1. The default is None.
    **kwargs : dict
      All keywords that are allowed in the DataParams class. Those are:

        sample_axis : int, optional
          Defines the sample axis. If None, the algorithm tries to find the
          sample axis on its own. The default is None.
        input_layer : str, optional
          Defines the shape of the input layer and thus the shape of the input
          data. The possible types are given by the official Keras layer names.
          The default is None.
        normalization : str, optional
          Defines the normalization type. Possible arguments are 'standardize'
          and 'min_max'. The default is None.

    Raises
    ------
    ValueError
      Number of samples given in 'sample_axis' does not match with samples in
      given in 'y'. Please change your sample axis.

    Returns
    -------
    Data container object.

    """
    self.X = X

    if y is not None:
      self.y = np.array(y, ndmin=1)
      self.classes = np.unique(y)
      self.n_classes = len(self.classes)

    else:
      self.y = y
      self.classes = None
      self.n_classes = None

    self.features = features
    self.feature_shape = None
    self.n_features = None
    self.n_samples = None

    # class variables for the pre-processing of the data:
    # values will be stored in them during training
    self._label_encoding_map = None
    self._feature_scale = None

    #params container
    self.params = DataParams()

    for key in kwargs:
      if key in self.params.__dict__.keys():
        self.params.update(key)(kwargs[key])
      else:
        raise KeyError(f"'{key}' is not a valid key for the generic neural "
                       "network useage.")

    self._params = {"data": self.params}

    self.X_train, self.y_train = self.prepare(self.X, self.y)
    self.X_test, self.y_test = None, None

  def __repr__(self):
      return (f"Data(Size of Dataset: {self.X.shape}, "
              f"Number of Samples: {self.n_samples}, "
              f"Number of Classes: {self.n_classes})")

  def get_params(self, type = None):
      if type is None:
          return self._params
      else:
          try:
              return self._params[f"{type}"]
          except:
              raise NameError(f"'{type}' is an invalid argument for 'type'.")

  # DATA PREPARATION:
  def prepare(self, X, y=None):
    """
    This method is applied at the beginning of the training process and
    prepares the input data and labels. It ensures that all the parameters are
    adjusted.

    Parameters
    ----------
    X : ndarray
      The input array.
    y : ndarray, optional
      The corresponding label array. The default is None.

    Raises
    ------
    ValueError
      DESCRIPTION.

    Returns
    -------
    X : ndarray
      The prepared data (normalized and shape converted).
    y : ndarray
      The prepared label array.

    """

    self.feature_shape = self.assort(X, y, self.params.sample_axis)
    self.n_features = np.prod(np.array(self.feature_shape))
    shape = (self.n_samples, ) + self.feature_shape

    # reshapes the input array to match the target data in terms of samples
    # axis and features
    X = X.reshape(shape)

    # conversion to shape of input layer
    if self.params.input_layer:
      X = self.convert_to_input_shape(X, self.params.input_layer)

    # normalization:
    X = self.normalize(X, self.params.normalization)

    return X, y

  def assort(self, X, y=None, sample_axis=None):
    """
    Assorts the input array to the target array and compares both arrays to
    find a matching sample axis. If there is no target given, the function will
    use the 'sample_axis' parameter.

    Parameters
    ----------
    X : ndarray
      The input array.
    y : ndarray, optional
      The target array. The default is None.
    sample_axis : int, optional
      Dimension for the sample axis. The default is None.

    Raises
    ------
    ValueError
      If sample axes do not match.

    Returns
    -------
    feature_shape : tuple
      The shape of the features of the input array.

    """

    X = np.array(X, ndmin=2)

    if y is not None:
      y = np.array(y, ndmin=1)
      # take number of samples and the feature shape from the sample axis:
      if self.params.sample_axis:
        self.n_samples = X.shape[sample_axis]
        feature_shape = np.take(X, 0, axis=sample_axis).shape

      # take number of samples and the feature shape from the first axis:
      else:
        self.n_samples = X.shape[0]
        feature_shape = X.shape[1:]

      # check whether sample axis is chosen right by comparing number of
      # samples in y and in given axis

      if self.n_samples != len(y):
        raise ValueError("Number of samples given in 'sample_axis' "
                         f"({self.n_samples}) does not match with samples in "
                         f"given in 'y' ({len(y)}). "
                         "Please change your sample axis.")

    # compare number of samples in target with the X array and take the axis
    # that matches the number of samples:
    else:

      # finding out sample axis if not specified and stores axis in params:
      self.params.sample_axis = self.get_sample_axis(X, sample_axis)
      self.n_samples = X.shape[self.params.sample_axis]

      #get feature shape:
      feature_shape = np.take(X, 0, axis = self.params.sample_axis).shape

    return feature_shape


  def get_sample_axis(self, X, sample_axis=None):
    """
    Automatically looks for the sample axis if no argument is given in
    'sample_axis'.

    Parameters
    ----------
    X : ndarray
      The input array.
    sample_axis : int, optional
      The dimension of the sample axis. The default is None.

    Returns
    -------
    sample_axis : int
      The spotted sample axis.

    """
    if not sample_axis:
      axis = np.argwhere(np.array(X.shape) == self.n_samples)
      try:
        sample_axis = int(axis)
      except:
        sample_axis = 0
    else:
      sample_axis = self.params.sample_axis

    return sample_axis

  # NORMALIZATION
  # In this version of the FeaSel Algorithm, we implemented two different
  # options for the noramlization (min-max scaling and standardization).
  def normalize(self, X, method='min-max'):
    """
    Normalizes the input array X.

    Parameters
    ----------
    X : ndarray
      Input array.
    method : str, optional
      Normalization type. The default is 'min-max'.

    Raises
    ------
    NameError
      If wrong normalization type is specified.

    Returns
    -------
    X_n : ndarray
      Normalized output array.

    """
    if self._feature_scale:
      X_n = self._scale(X)

    else:
      if method:
        try:
          #normalize and transform shape to correct input layer shape
          NORM = {'standardize': self._standardize,
                  'min_max': self._min_max}
          X_n = NORM[method](X)

        except:
          raise NameError(f"'{method}' is not implemented as normalization "
                          "technique. Try 'standardize' or 'min-max'.")
      else:
        X_n = X

    return X_n

  def _scale(self, X):
    """
    Scales the input values X in case that the feature scaling parameters are
    provided.

    Parameters
    ----------
    X : ndarray
      Input array.

    Returns
    -------
    X_n : ndarray
      Normalized array.

    Raises
    ------
    NotImplementedError
      If feature scaling parameters are not provided.

    """
    scale = self._feature_scale

    try:
      # standardization:
      if 'x_bar' in scale.keys():
        X_n = (X - scale['x_bar']) / scale['s']

      # min-max normalization:
      else:
        X_n = X * scale['scale'] + scale['offset']

    except:
      raise NotImplementedError('There are no feature scales to scale the '
                                'input data X.')

    return X_n

  def _standardize(self, X):
    X, self._feature_scale = pre.standardize(X, axis=0, return_scale=True)
    return X

  def _min_max(self, X, a_max = 1., a_min = 0.):
    X, self._feature_scale = pre.min_max(X, axis=0,
                                         a_max=a_max, a_min=a_min,
                                         return_scale=True)
    return X

  # DIMENSIONALITY CHANGES
  # The following methods change the dimensionality of the input arrays if they
  # are wrongly shaped. This secures a training in each instance.
  def convert_to_input_shape(self, X, input_layer):
    """
    Automatically converts the given dataset to match the requirements of the
    input layers from Keras (e.g. Dense or Convolutional Layers).

    If the dimensionality of the original data is too low, it will be expanded
    at the previous last dimension. On the other hand, if it is too high, the
    dimensionality will be reduced by multiplying the last two dimensions in
    order to get only one remaining dimension with all features encoded in this
    last dimension. The process iterates as long as the dimensionalities do not
    match.

    Parameters
    ----------
    X : ndarray
      Input array.
    input_layer : str
      Defines the input layer type. The layer types correspond to the
      expressions used by Keras.

    Raises
    ------
    NameError
      The input layer is either not a possible keras input layer or not
      implemented yet. Please try another layer.

    Returns
    -------
    None.

    """
    layer_dims = {"Dense": 2,
                  "Conv1D": 3,
                  "Conv2D": 4,
                  "Conv3D": 5}
    try:
      input_layer_dim = layer_dims[f"{input_layer}"]

      while True:
        X_dim = X.ndim

        # expansion of dimensionality by adding another dimension after the
        # previous last dimension
        if input_layer_dim > X_dim:
          X = np.expand_dims(X, axis = -1)

        # reduction of dimensionality by multiplying the last two
        # dimensionalities as the new reduced dimension
        elif input_layer_dim < X_dim:
          shape = ((self.n_samples, ) + X.shape[1:-2]
                   + (X.shape[-2] * X.shape[-1], ))
          X = X.reshape(shape)

        else:
          break

    except:
      raise NameError(f"'{input_layer}' is either not a possible keras input "
                      "layer or not implemented yet. Please try one of the "
                      "following layers: "
                      f"{layer_dims.keys()}.")

    return X

  def categorical_labels(self, y):
    """
    Converts an one-hot encoded array into a categorical array. If the array is
    categorical already, it will be manipulated such that the neural network
    can interpret the labels, i.e. it will rename the classes into an
    enumeration of integers starting from 0 to the number of classes n_c - 1
    (is obsolete when already happened).

    Parameters
    ----------
    y : ndarray
      The array with label entries.

    Returns
    -------
    categorical : ndarray
      An array of categorical labels.

    """
    y = pre.categorical(y)
    return y

  def one_hot_labels(self, y):
    """
    Converts a categorical array into an one-hot encoded array. If the array is
    one-hot-encoded already, it will be returned unmanipulated.

    Parameters
    ----------
    y : ndarray
      The array with label entries.

    Returns
    -------
    one_hot : ndarray
      An array of the one-hot encoded labels.
    """
    if y is not None:
      y = pre.one_hot(y)
    return y

  def train_test_split(self, X, y, test_split = 0.25, **kwargs):
    """
    The sklearn function 'train_test_split' is applied on the input data (X,y).
    Given a 'test_split' argument, the function will provide a matching ratio
    of training and test data (X_train, y_train) and (X_test, y_test).

    Parameters
    ----------
    X : ndarray
      Input array.
    y : ndarray
      Corresponding label array.
    test_split : float, optional
      The ratio of test to training data. The default is 0.25.
    **kwargs : dict
      All key-words that are accepted by the original function.

    Returns
    -------
    split : tuple
      A tuple consisting of training and test data (X_train, X_test, y_train,
      y_test).

    """
    split = train_test_split(X, y, test_size=test_split, **kwargs)

    return split