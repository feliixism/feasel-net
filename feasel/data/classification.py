"""
feasel.data.classification
==========================
"""

import numpy as np
from sklearn.model_selection import train_test_split

from ..parameter import data
from . import preprocess as prep

class Linear:
  """
  This is a data object for the regression or classification with principal
  component analysis (PCA) or Linear Discriminant Analysis (LDA) respectively.
  Within this class all necessary pre-processing and data-handling tasks can be
  executed.

  Parameters
  ----------
  X : ndarray [int or float]
    Input array for the fitting of the analyses.
  y : ndarray [int, float or str], optional
      Target array for the loss estimation if provided. This data is only
      necessary when dealing with LDAs, but can also be used for PCAs (e.g.
      for plot functionalities). The default is None.
  features : ndarray [int, float or str], optional
    Defines the feature's names. It must have the same size as the feature
    shape. If None, the features are iterated from :math:`0` to the number of
    features :math:`n_f-1`. The default is `None`.
  **kwargs : dict
    All keywords that are allowed in the DataParams class.

    These are:
      sample_axis : int, optional
        Defines the sample axis. If None, the algorithm tries to find the
        sample axis on its own. The default is None.
      normalization : str, optional
        Defines the normalization type. Possible arguments are 'standardize'
        and 'min_max'. The default is None.

  Raises
  ------
  ValueError
    If invalid parameter key.

  """
  def __init__(self, X, y=None, features=None, **kwargs):
    self.X = X

    if y is not None:
      self.y = np.array(y, ndmin=1)
      self.classes = np.unique(y)
      self.n_classes = len(self.classes)
    else:
      self.y = None
      self.classes = None
      self.n_classes = None

    self.features = features
    self.feature_shape = None
    self.n_features = None
    self.n_samples = None

    # class variables for the pre-processing of the data:
    # values will be stored in them during training
    self._feature_scale = None
    self.mu = None


    self.class_props = {'n': [],
                        'mu': [],
                        'sigma': [],
                        'prior': [],
                        'scatter': [],
                        'covariance': []}

    self.evals = None
    self.evecs = None
    self.loadings = None
    self.scores = None
    self.score_variance = None
    self.oev = None
    self.cev = None
    self.contribution = None
    self.importance = None

    self.params = data.DataParamsLinear(**kwargs)

    for key in kwargs:
      if key in self.params.__dict__.keys():
        self.params.update(key)(kwargs[key])
      else:
        raise KeyError(f"'{key}' is not a valid key for the generic neural "
                       "network useage.")

    self.X_train, self.y_train = self.prepare(self.X, self.y)

    if self.params.test_split:
      split = self.train_test_split(self.X_train, self.y_train,
                                    test_split=self.params.test_split,
                                    random_state=42)
      self.X_train, self.X_test, self.y_train, self.y_test = split

    else:
      self.X_test, self.y_test = None, None

  def __repr__(self):
      return (f"Data(Size of Dataset: {self.X.shape}, "
              f"Number of Samples: {self.n_samples}, "
              f"Number of Classes: {self.n_classes})")

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

    X = self.assort(X, y, self.params.sample_axis)

    if isinstance(self.features, type(None)):
      self.features = np.arange(self.n_features).reshape(self.feature_shape)

    self.sample_axis = 0 # resetted after reshape

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
    X : ndarray
      The re-ordered array.

    """

    X = np.array(X, ndmin=2)
    dims = X.shape

    sample_axis = self._get_sample_axis(X, y, sample_axis)

    if sample_axis:
      # set n_samples and feature_shape:
      self.n_samples = dims[sample_axis]
      self.feature_shape = np.take(X, 0, axis=sample_axis).shape

      X = np.moveaxis(X, sample_axis, 0) # re-order dims

    # take number of samples and the feature shape from the first axis:
    else:
      self.n_samples = dims[0]
      self.feature_shape = dims[1:]

    #check whether sample axis is chosen right by comparing number of
    #samples in y and in given axis
    if self.n_samples != len(y):
      raise ValueError("Number of samples given in 'sample_axis' "
                       f"({self.n_samples}) does not match with samples in "
                       f"given in 'y' ({len(y)}). "
                       "Please change your sample axis.")

    # sets number of features
    self.n_features = np.prod(np.array(self.feature_shape))

    return X

  def _get_sample_axis(self, X, y, sample_axis):
    """
    Compares input data and targets to match the sample axis. If a sample axis
    is specified, this will be skipped and the specified axis is returned.

    Parameters
    ----------
    X : ndarray
      Input array.
    y : ndarray
      Target array.
    sample_axis : int
      The sample dimension.

    Returns
    -------
    sample_axis : int
      The actual sample dimension.

    """
    if sample_axis:
      sample_axis = sample_axis

    else:
      # comparison with target values:
      if y is not None:
        sample_axis = int(np.argwhere(np.array(X.shape) == len(y)))

      else:
        sample_axis = sample_axis

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
      try:
        #normalize and transform shape to correct input layer shape
        if not isinstance(method, type(None)):
          NORM = {'standardize': self._standardize,
                  'min_max': self._min_max}
          X_n = NORM[method](X)

        else:
          X_n = X

      except:
        raise NameError(f"'{method}' is not implemented as normalization "
                        "technique. Try 'standardize' or 'min-max'.")

    return X_n

  def _scale(self, X):
    """
    Scales the input values X in case that feature scaling parameters are
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
      if 'x_bar' in scale.keys():
        X_n = (X - scale['x_bar']) / scale['s'] # standardization
      else:
        X_n = X * scale['scale'] + scale['offset'] # min-max normalization
    except:
      raise NotImplementedError('There are no feature scales to scale the '
                                'input data X.')
    return X_n

  def _standardize(self, X):
    X, self._feature_scale = prep.standardize(X, axis=0, return_scale=True)
    return X

  def _min_max(self, X, a_max = 1., a_min = 0.):
    X, self._feature_scale = prep.min_max(X, axis=0,
                                          a_max=a_max, a_min=a_min,
                                          return_scale=True)
    return X

  def mean_centered(self, X):
    if isinstance(self.mu, type(None)):
      X_c, self.mu = prep.mean_centered(X, return_shift=True)
    else:
      X_c = prep.mean_centered(X, return_shift=False)
    return X_c

  def scatter_matrix(self, X):
    X_c = self.mean_centered(X)
    S = X_c.T @ X_c
    return S

  def covariance_matrix(self, X):
    """
    Calculates the covariance of an array.
    The covariance shows how much a variable is related to another.

    Parameters
    ----------
    X : float
      Input array of the raw data.

    Returns
    -------
    Sigma : float
      Covariance Matrix.

    """
    n, m = X.shape
    S = self.scatter_matrix(X)
    Sigma = S / (n - 1)
    return Sigma

  def correlation_matrix(self, X):
    """
    Calculates the Pearson correlation coefficients of the input matrix.
    Pearsons correlation only refers to linear relations between two variables.

    All correlation coefficents have values in the range between 0 and 1, where
    1 denotes the highest correlation.

    X : float
      Input array of the raw data.

    Returns
    -------
    R : float
      Correlation Matrix.

    """
    Sigma = self.covariance_matrix(X)
    D_inv = np.diag(1 / np.sqrt(np.diag(Sigma)))
    R = D_inv @ Sigma @ D_inv
    return R

  def synchronous(self, X):
    syn = self.covariance(X)
    return syn

  def asynchronous(self, X):
    n, m = X.shape
    X_c, self.mu = prep.mean_centered(X, return_shift=True)

    def hilbert_noda(X):
        N = np.empty([m, n])
        for j in range(m):
            for k in range(n):
                if j == k:
                    N[j,k] = 0
                else:
                    N[j,k] = 1 / (np.pi * (k-j))
        return N

    N = hilbert_noda(X_c)

    S = X_c @ (X_c * N).T
    asyn = S / (n - 1)
    return asyn

  # TARGET ENCODING:
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
    y = prep.categorical(y)
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
      y = prep.one_hot(y)
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

class NN:
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

  Attributes
  ----------
  n_classes : int
    The number of classes.

  Raises
  ------
  ValueError
    Number of samples given in 'sample_axis' does not match with samples in
    given in 'y'. Please change your sample axis.

  """
  def __init__(self, X, y=None, features=None, **kwargs):

    self.X = X

    if y is not None:
      self.y = np.array(y, ndmin=1)
      self.classes = np.unique(y)
      self.n_classes = len(self.classes)

    else:
      self.y = y
      self.classes = None
      self.n_classes = None

    try:
      len(features)
      self.features = features
    except:
      self.features = np.arange(X.shape[1])

    self.feature_shape = None
    self.n_features = None
    self.n_samples = None

    # class variables for the pre-processing of the data:
    # values will be stored in them during training
    self._label_encoding_map = None
    self._feature_scale = None

    #params container
    self.params = data.DataParamsNN()

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
    y = prep.categorical(y)
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
      y = prep.one_hot(y)
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