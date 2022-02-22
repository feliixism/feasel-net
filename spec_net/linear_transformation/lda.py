import numpy as np
from scipy.linalg import eigh

from .base import ModelContainer
from .pca import PCA
from .svd import SVD
from .data import Classification
from .parameters import Params
from .plots import LinearTransformationVisualizer as Visualizer
from ..utils.time import get_timestamp

class LDA(ModelContainer):
  def __init__(self, X, y,
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
    y : float
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
    self._data = Classification(X, y, features,
                                sample_axis=self.params.data.sample_axis,
                                normalization=self.params.data.normalization)

    # embed plot class
    self._plot = Visualizer(self)

    self.p = self.data.n_features
    self.q = self.params.build.n_components

  def __str__(self):
    return 'ModelContainer for LDAs'

  def __repr__(self):
    return (f"LDA(n_features: {self.data.n_features}, "
            f"n_samples: {self.data.n_samples}, "
            "n_components: {self.params.build.n_components}, "
            "normalization: {self.params.data.normalization})")

  @property
  def data(self):
    return self._data

  @property
  def plot(self):
    return self._plot

  def get_dependencies(self, X, type='scatter'):
    """
    Calculation of either covariance or correlation matrix. This is a measure
    of the linear dependencies between the features.

    Returns
    -------
    Sigma or R : ndarray
      The covariance or correlation matrix of the input dataset.

    """
    TYPE = {'scatter': self.data.scatter_matrix }
    try:
      self.data.mean_centered(self.data.X_train) # sets 'data.mu'
      self.data.scatter_w = np.zeros([self.data.n_features, self.data.n_features])
      self.data.scatter_b = np.zeros([self.data.n_features, self.data.n_features])
      for c in self.data.classes: # handbook of statistics (Cheriet 2013)
        idx = np.argwhere(self.y == c).flatten()
        data = self.data.X_train[idx]
        # add data to each class:
        self.data.class_props['n'].append(len(data))
        self.data.class_props['prior'].append(data.shape[0]
                                              / self.data.n_samples)
        self.data.class_props['mu'].append(np.mean(data,
                                                   axis=0,
                                                   keepdims=True))
        self.data.class_props['sigma'].append(np.std(data,
                                                     axis=0,
                                                     ddof=1,
                                                     keepdims=True))
        self.data.class_props['scatter'].append(TYPE[type](data))
        S = self.data.class_props['scatter'][-1]
        n = self.data.class_props['n'][-1]
        self.data.class_props['covariance'].append(S / (n - 1))
        self.data.scatter_w += S
        d_mu = self.data.class_props['mu'][-1] - self.data.mu
        self.data.scatter_b += n * d_mu.T @ d_mu
      self.data.common_covariance = (self.data.scatter_b
                                     / (self.data.n_samples
                                        - self.data.n_classes))

      sw_sb = np.linalg.inv(self.data.scatter_w) @ self.data.scatter_b

      return sw_sb

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
    if isinstance(self.data.evals, np.ndarray):
      return

    D = self.get_dependencies(self.data.X_train, 'scatter')

    if self.params.build.solver == 'eigs':
      evals, evecs = np.linalg.eig(D) # complex eigenvalues

      # this ensures the evals to be real valued:
      evals = np.abs(evals)
      idx = np.flip(np.argsort(evals))

      evals = evals[idx]
      evecs = evecs[:, idx].real

    elif self.params.build.solver == 'svd':
      svd = SVD(D)
      U, Sigma, Vh = svd()
      evals, evecs = np.diagonal(Sigma), Vh.T

    self.data.score_variance = evals**2
    self.data.evals, self.data.evecs = np.array(evals, ndmin=2), evecs
    self.get_overall_explained_variance()

  def get_pca(self, n_components=None):
    """
    Gets a PCA for the pre-processing of the LDA's data. This is necessary, if
    the number of features is bigger than the number of samples.

    Parameters
    ----------
    n_components : int, optional
      The number of components for the PCA. The default is None.

    Returns
    -------
    None.

    """
    if n_components is None:
        n_components = int(self.n_samples * 0.1) # 10 % of sample data
    self.pca = PCA(self.X, self.y,
                   n_components=self.params.build.n_components,
                   normalization=self.params.build.normalization,
                   solver=self.params.build.solver)
    self.pca.get_eigs()

  def decision_rule(self, scores, components=None, transformed=False):

    if not isinstance(components, type(None)):
      loadings = self.data.loadings[components]
      scores = scores[:, components]

    else:
      loadings = self.data.loadings[:self.params.build.n_components]
      scores = scores[:, :self.params.build.n_components]

    common_covariance = loadings @ self.data.common_covariance @ loadings.T
    common_covariance_ = np.linalg.inv(common_covariance)

    # actual decision rule
    delta = np.zeros([len(scores), self.data.n_classes])

    for i, c in enumerate(self.data.classes):
      mu = loadings @ self.data.class_props["mu"][i].T
      delta[:, i] = (np.log10(self.data.class_props["prior"][i])
                     - 1/2 * mu.T @ common_covariance_ @ mu
                     + scores @ common_covariance_ @ mu).flatten()

    return delta

  def predict(self, X_test, components=None):
    scores = self.get_scores(X_test)

    delta = self.decision_rule(scores, components, transformed=True)

    idx = np.argmax(delta, axis=1)
    y_pred = self.data.classes[idx]

    return y_pred

