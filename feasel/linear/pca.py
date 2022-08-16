import numpy as np
from scipy.linalg import eigh

from .base import ModelContainer
from .svd import SVD
from ..data import regression
from ..parameters import ParamsLinear
from ..plots import LinearTransformationVisualizer as Visualizer
from ..utils.time import get_timestamp

class PCA(ModelContainer):
  def __init__(self, X, y=None,
               features=None,
               name=None,
               **kwargs):
    """
    A class environment for analysing X using the theory of principal
    component analysis. This is an unsupervised technique for dimensionality
    reduction.

    Parameters
    ----------
    X : float
      2D-data array.
    y : float, optional
      1D-class array. The default is 'None'.
    features : float
      1D-Array with features. If 'None', the features are incremented from 0 to
      N-1. The default is 'None'.
    name : str, optional
      The model name. The default is 'None'.

    Returns
    -------
    None.

    """
    self.X = X
    self.y = y
    self.features = features

    self.set_name(name)
    self.timestamp = get_timestamp()

    self._get_params(**kwargs)

    #data container
    self._data = regression.Linear(X, y, features,
                                   sample_axis=self.params.data.sample_axis,
                                   normalization=self.params.data.normalization)

    # embed plot class
    self._plot = Visualizer(self)

    self.p = self.data.n_features
    self.q = self.params.build.n_components

  def __str__(self):
    return 'ModelContainer for PCAs'

  def __repr__(self):
    return (f"PCA(n_features: {self.data.n_features}, "
            f"n_samples: {self.data.n_samples}, "
            "n_components: {self.params.build.n_components}, "
            "normalization: {self.params.data.normalization})")

  @property
  def data(self):
    return self._data

  @property
  def plot(self):
    return self._plot

  def set_name(self, name):
    """
    Sets the name of the container.

    Parameters
    ----------
    name : str
      Name of the container. If None, it will use the class' name. The default
      is None.

    Returns
    -------
    None.

    """
    if not name:
      self.name = str(self)
    else:
      self.name = name

  def _get_params(self, **kwargs):
    """
    Automatically checks the kwargs for valid parameters for each parameter
    object and updates them.

    Parameters
    ----------
    **kwargs : kwargs
      The parameter keywords.

    Raises
    ------
    KeyError
      If kwarg cannot be assigned to any parameter object.

    Returns
    -------
    None.

    """
    # parameter container with train, build and data parameters
    self.params = ParamsLinear()

    for key in kwargs:
      containers = {'build': self.params.build,
                    'data': self.params.data}

      # updates all values that are summarized in an extra container:
      if key in containers:
        for sub_key in kwargs[key]:
          if sub_key in containers[key].__dict__.keys():
            containers[key].update(sub_key)(kwargs[key][sub_key])

      # updates keys if they are not defined in an extra container:
      elif key in self.params.build.__dict__.keys():
        self.params.build.update(key)(kwargs[key])

      elif key in self.params.data.__dict__.keys():
        self.params.data.update(key)(kwargs[key])

      elif key in self.params.train.__dict__.keys():
        self.params.train.update(key)(kwargs[key])

      else:
        raise KeyError(f"'{key}' is not a valid key for the generic "
                       "linear transformation useage.")

  def get_dependencies(self, X, type='covariance'):
    """
    Calculation of either covariance or correlation matrix. This is a measure
    of the linear dependencies between the features.

    Returns
    -------
    Sigma or R : ndarray
      The covariance or correlation matrix of the input dataset.

    """
    TYPE = {'covariance': self.data.covariance_matrix,
            'correlation': self.data.correlation_matrix}
    try:
      return TYPE[type](X)

    except:
      raise NameError(f'{type} is an invalid parameter for type.')

  # internal getters
  def get_eigs(self):
    """
    Solves the eigenvalue problem of the quadratic covariance or
    correlation matrix respectively. The order of evecs and evals is sorted
    by the magnitude of the evals from highest to lowest.

    Returns
    -------
    None.

    """
    if self.data.evals is not None:
      return

    D = self.get_dependencies(self.data.X_train, self.params.build.method)

    if self.params.build.solver == 'eigs':
      evals, evecs = eigh(D, lower=False)

      # ensure positive values (zero can also be expressed by negative values
      # with very low exponents):
      evals = np.abs(np.flip(evals))
      evecs = np.flip(evecs, axis=1)

    elif self.params.build.solver == 'svd':
      svd = SVD(D)
      U, Sigma, Vh = svd()
      evals, evecs = np.diagonal(Sigma), Vh.T

    self.data.score_variance = evals**2
    self.data.evals, self.data.evecs = np.array(evals, ndmin=2), evecs
    self.get_overall_explained_variance()

  def get_important_features(self, n_var, d_min = 0):
      """
      A method to find the most relevant features in the Xset. This is
      done by sorting the loading values along each principal component,
      which is nothing else than the correlation of the original variable
      and the principal component.

      Parameters
      ----------
      n_var : int
        Number of most important features per component.
      d_min : int, optional
        Minimum difference between indices of important features. The default
        is 0.

      Returns
      -------
      dict : dict
        Dictionary of variable(s), loading (correlation) value(s) and index
        (indices) per component.

      """
      most_important_features = {}

      for component in range(self.n_components):
        sort_mask = np.argsort(np.abs(self.loadings.T)[:, component])
        i_sorted = np.flip(sort_mask)
        x_sorted = np.flip(self.features[sort_mask])
        a_sorted = np.flip(self.data.loadings[component][sort_mask])

        #generating dictionary with information about variable, correlation and index
        count = 0
        x, a, i = [], [], []

        for pos in range(len(i_sorted)):
          if count == 0:
              x.append(x_sorted[pos])
              a.append(a_sorted[pos])
              i.append(i_sorted[pos])
              count += 1

          elif np.sum((np.abs(np.array(x) - x_sorted[pos])) > d_min) == count:
              x.append(x_sorted[pos])
              a.append(a_sorted[pos])
              i.append(i_sorted[pos])
              count += 1

          if count >= n_var:
              most_important_features[f"PC{component+1}"] = {"feature": x,
                                                             "a": a,
                                                             "i": i}
              break

      return most_important_features

  def get_contribution(self):
    if not isinstance(self.data.contribution, np.ndarray):
      if not isinstance(self.data.loadings, np.ndarray):
        self.get_loadings()

      self.data.contribution = np.empty(self.data.loadings.shape)

      for i in range(len(self.data.loadings)):
        self.data.contribution[i] = (np.abs(self.data.loadings[i])
                                     / np.sum(np.abs(self.data.loadings[i])))