import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from .svd import SVD
from .data import Regression
from .parameters import Params
from .plots import LinearTransformationVisualizer as Visualizer
from ..utils.time import get_timestamp

class ModelContainer:
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
    self._data = Regression(X, y, features,
                            sample_axis=self.params.data.sample_axis,
                            normalization=self.params.data.normalization)

    # embed plot class
    self._plot = Visualizer(self)

  def __str__(self):
    return 'ModelContainer for Linear Trasformations'

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
    self.params = Params()

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
    if self.data.evals:
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
      evals, evecs = np.diagonal(Sigma)**2 / (self.data.n_samples-1), Vh

    self.data.score_variance = evals**2
    self.data.evals, self.data.evecs = np.array(evals, ndmin=2), evecs
    self.get_overall_explained_variance()

  def get_overall_explained_variance(self):
    """
    Calculates the overall explained variance of each principal component.

    Returns
    -------
    None.

    """
    var = self.data.evals
    oev = var / np.sum(var) * 100
    cev = np.zeros(self.data.evals.shape)

    for i in range(len(self.data.evals[0])):
      if i > 0:
        cev[0, i] = (oev[0, i] + cev[0, i - 1])

      else:
        cev[0, i] = oev[0, i]

    self.data.oev = oev
    self.data.cev = cev

  def get_loadings(self):
    """
    Calculates the so-called loading-factors of each principal component.
    The loadings can be interpreted as weights for the orthogonal linear
    transformation of the original features into the new latent features.

    Returns
    -------
    None.

    """
    if not isinstance(self.data.evals, np.ndarray):
      self.get_eigs()

    evals, evecs = self.data.evals, self.data.evecs

    # select q most important components:
    p = self.data.n_features
    I = np.identity(p)

    self.data.loadings = I * np.sqrt(evals) @ evecs.T

  def get_scores(self, X_test=None):
    """
    Calculates the values of the original features projected into the new
    latent variable space described by the corresponding principal
    components.

    Returns
    -------
    None.

    """
    if not isinstance(self.data.loadings, np.ndarray):
      self.get_loadings()

    if not X_test:
      X_test = self.data.X_train
      scores = (self.data.loadings @ X_test.T).T
      self.data.scores = scores

    else:
      scale = self.data._feature_scale

      # standardization:
      if 'x_bar' in scale.keys():
        X_test = (X_test - scale['x_bar']) / scale['s']

      # min-max normalization:
      elif 'offset' in scale.keys():
        X_test = scale['scale'] * X_test + scale['offset']

      scores = (self.data.loadings @ X_test.T).T

    return scores

    # for class_ in self.classes[0]:
    #   mask = np.where(self.y == class_)[0]
    #   self.log[f"{class_}"]["scores"] = self.scores[mask]
    #   self.log[f"{class_}"]["scores_var"] = np.var(self.log[f"{class_}"]["scores"], axis = 0)
    #   self.log[f"{class_}"]["scores_mean"] = np.mean(self.log[f"{class_}"]["scores"], axis = 0)

    # self.var_scores = np.var(self.scores, axis = 0, ddof = 1)
    # self._scores = True

  def decode(self):
    if not self.data.scores:
      self.get_scores()

    X_decoded = (self.scores @ self.evecs[:self.n_components]).T
    return X_decoded

  def plot_reduction(self):
    if self._reduced == False:
        self.get_reduced_X()

    n_pixels = self.n_samples * self.n_features
    reduction = np.round((1 - (self.scores.size + self.evecs[:self.n_components].size) / n_pixels) * 100, 2)
    mse = np.sum((self.reduced_X - self.preprocessed_X)**2) / n_pixels
    psnr = -np.log10(mse / (np.amax(self.preprocessed_X) + np.abs(np.amin(self.preprocessed_X)))**2)
    fig, axs = plt.subplots(2,1)
    axs[0].imshow(self.X.T)
    axs[0].title.set_text("Original")
    axs[1].imshow(self.reduced_X.T)
    axs[1].title.set_text(f"Reduction: {reduction} \%; PSNR: {psnr:0.2f} dB")

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