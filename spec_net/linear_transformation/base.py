# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.linalg import eigh
# from spec_net.preprocess.data import covariance, correlation

# from .data import Classification
# from .parameters import Params
# from .plots import LinearTransformationVisualizer as Visualizer

# class ModelContainer:
#   def __init__(self, X, y=None,
#                features=None,
#                name=None,
#                solver="eigs",
#                **kwargs):
#     """
#     A class environment for analysing X using the theory of principal
#     component analysis. This is an unsupervised technique for dimensionality
#     reduction.

#     Parameters
#     ----------
#     X : float
#       2D-data array.
#     y : float
#       1D-class array. The default is 'None'.
#     n_components : int, optional
#       Sets the number of principal components. The default is 2.
#     features : float
#       1D-Array with features. If 'None', the features are incremented from 0 to
#       N-1. The default is 'None'.
#     rowvar : boolean, optional
#       Indicates the axis of the features. If features are described by the
#       columns then set to 'False'. The default is True.
#     normalize : boolean, optional
#       Optional standardization of the X along sample axis if set to 'True'.
#       The default is False.

#     Returns
#     -------
#     None.

#     """
#     self.X = X
#     self.y = y

#     if features is not None:
#       self.features = np.array(features)
#     else:
#       self.features = features

#     self.get_params(**kwargs)

#     self.set_name(name)

#     #data container
#     self._data = Classification(X, y, features,
#                                 sample_axis=self.params.data.sample_axis,
#                                 normalization=self.params.data.normalization)

#     # if self._rowvar == False:
#     #   self.X = self.X.T

#     # if y is not None:
#     #   classes = np.unique(y)
#     #   self.n_classes = len(classes)
#     #   self.classes = np.array([classes, np.arange(self.n_classes)])
#     #   self.log = self.get_log()

#     # #number features
#     # self.n_features, self.n_samples = self.X.shape
#     # self.n_components = n_components

#     # if features is None:
#     #   idx = np.arange(self.n_features)
#     #   self.features = {"idx": idx, "name": idx}

#     # else:
#     #   idx = np.arange(self.n_features)
#     #   features_label = np.array([features]).squeeze()
#     #   self.features = {"idx": idx, "name": features_label}

#     # #solver information
#     # self.solver = solver

#     # #boolean features
#     # self._normalized = normalize
#     # self._eigs, self._loadings, self._scores = False, False, False
#     # self._preprocessed, self._reduced = False, False
#     # self._contribution = False

#     # embed plot class
#     self._plot = Visualizer(self)


#   def __str__(self):
#     return 'ModelContainer for Linear Transformations'

#   def __repr__(self):
#     return ("ModelContainer generic Linear Transformation object"
#             f"(n_features: {self.n_features}, "
#             f"n_samples: {self.n_samples}, "
#             f"n_components: {self.n_components}, "
#             f"normalized: {self._normalized})")

#   @property
#   def data(self):
#     return self._data

#   @property
#   def plot(self):
#     return self._plot

#   def set_name(self, name):
#     """
#     Sets the name of the container.

#     Parameters
#     ----------
#     name : str
#       Name of the container. If None, it will use the class' name. The default
#       is None.

#     Returns
#     -------
#     None.

#     """
#     if not name:
#       self.name = str(self)
#     else:
#       self.name = name

#   def _get_params(self, **kwargs):
#     """
#     Automatically checks the kwargs for valid parameters for each parameter
#     object and updates them.

#     Parameters
#     ----------
#     **kwargs : kwargs
#       The parameter keywords.

#     Raises
#     ------
#     KeyError
#       If kwarg cannot be assigned to any parameter object.

#     Returns
#     -------
#     None.

#     """
#     # parameter container with train, build and data parameters
#     self.params = Params()

#     for key in kwargs:
#       containers = {'build': self.params.build,
#                     'data': self.params.data,
#                     'train': self.params.train}

#       # updates all values that are summarized in an extra container:
#       if key in containers:
#         for sub_key in kwargs[key]:
#           if sub_key in containers[key].__dict__.keys():
#             containers[key].update(sub_key)(kwargs[key][sub_key])

#       # updates keys if they are not defined in an extra container:
#       elif key in self.params.build.__dict__.keys():
#         self.params.build.update(key)(kwargs[key])

#       elif key in self.params.data.__dict__.keys():
#         self.params.data.update(key)(kwargs[key])

#       elif key in self.params.train.__dict__.keys():
#         self.params.train.update(key)(kwargs[key])

#       else:
#         raise KeyError(f"'{key}' is not a valid key for the generic "
#                        "neural network useage.")

#   def get_log(self):
#     dict={}
#     for class_ in self.classes[0]:
#       data = self.X[ : , np.argwhere(self.y == class_)].squeeze()
#       dict[f"{class_}"] = {"data": data}
#     return dict

#   def preprocess(self):
#     """
#     Mean-centering of the X. If 'normalized' is 'True', the X is
#     also normalized by dividing it by the standard deviation of each
#     variable.

#     Returns
#     -------
#     None.

#     """
#     self.mu = np.mean(self.X, axis = 1, keepdims = True)
#     self.sigma = np.std(self.X, axis = 1, ddof = 1, keepdims = True)

#     if self._normalized == True:
#         sigma = self.sigma

#     else:
#         sigma = np.ones(self.mu.shape)

#     self.preprocessed_X = (self.X - self.mu) / sigma
#     self._preprocessed = True

#     #log information
#     for class_ in self.classes[0]:
#       data = self.preprocessed_X[ : , np.argwhere(self.y == class_)[:, 0]]
#       self.log[f"{class_}"]["N"] = data.shape[-1]
#       self.log[f"{class_}"]["data"] = data
#       self.log[f"{class_}"]["mu"] = np.mean(data, axis = 1, keepdims = True)
#       self.log[f"{class_}"]["sigma"] = np.std(data, axis = 1, ddof = 1, keepdims = True)

#   # internal getters
#   def get_eigs(self):
#     """
#     Solves the eigenvalue problem of the quadratic covariance or
#     correlation matrix respectively. The order of evecs and evals is sorted
#     by the magnitude of the evals from highest to lowest.

#     Returns
#     -------
#     None.

#     """
#     if not self._preprocessed:
#         self.preprocess()

#     if self.solver == "eigs":
#       if self._normalized:
#         evals, evecs = eigh(correlation(self.preprocessed_X), lower = False)

#       else:
#         evals, evecs = eigh(covariance(self.preprocessed_X), lower = False)

#       evals = np.flip(evals)
#       evecs = np.flip(evecs.T, 0)

#     elif self.solver == "svd": # Flipped signs every second row in evecs. Why?
#       svd = SVD(self.preprocessed_X.T)
#       U, Sigma, Vh = svd()
#       evals, evecs = np.diagonal(Sigma)**2 / (self.n_samples - 1), Vh

#     self.var_scores = evals**2
#     self.evals, self.evecs = np.array(evals, ndmin=2), evecs
#     self.get_overall_explained_variance()
#     self._eigs = True

#   def get_overall_explained_variance(self):
#     """
#     Calculates the overall explained variance of each principal component.

#     Returns
#     -------
#     None.

#     """
#     oev = self.evals / np.sum(self.evals) * 100
#     cev = np.zeros(self.evals.shape)
#     for i in range(len(self.evals[0])):
#       if i > 0:
#         cev[0, i] = (oev[0, i] + cev[0, i - 1])
#       else:
#         cev[0, i] = oev[0, i]
#     self.oev = oev
#     self.cev = cev

#   def get_loadings(self):
#     """
#     Calculates the so-called loading-factors of each principal component.
#     The loadings can be interpreted as weights for the orthogonal linear
#     transformation of the original features into the new latent features.

#     Returns
#     -------
#     None.

#     """
#     if not self._eigs:
#       self.get_eigs()
#     self.loadings = (np.identity(len(self.evals[0]))*np.sqrt(self.evals[0])
#                      @ self.evecs)
#     self._loadings = True

#   def get_scores(self):
#     """
#     Calculates the values of the original features projected into the new
#     latent variable space described by the corresponding principal
#     components.

#     Returns
#     -------
#     None.

#     """
#     if not self._loadings:
#       self.get_loadings()
#     self.scores = (self.loadings @ self.preprocessed_X).T

#     for class_ in self.classes[0]:
#       mask = np.where(self.y == class_)[0]
#       self.log[f"{class_}"]["scores"] = self.scores[mask]
#       self.log[f"{class_}"]["scores_var"] = np.var(self.log[f"{class_}"]["scores"], axis = 0)
#       self.log[f"{class_}"]["scores_mean"] = np.mean(self.log[f"{class_}"]["scores"], axis = 0)
#     self.var_scores = np.var(self.scores, axis = 0, ddof = 1)
#     self._scores = True

#   def get_reduced_X(self):
#     if self._scores == False:
#       self.get_scores()

#     self.reduced_X = (self.scores @ self.evecs[:self.n_components]).T
#     self._reduced = True

#   def plot_reduction(self):
#     if self._reduced == False:
#         self.get_reduced_X()

#     n_pixels = self.n_samples * self.n_features
#     reduction = np.round((1 - (self.scores.size + self.evecs[:self.n_components].size) / n_pixels) * 100, 2)
#     mse = np.sum((self.reduced_X - self.preprocessed_X)**2) / n_pixels
#     psnr = -np.log10(mse / (np.amax(self.preprocessed_X) + np.abs(np.amin(self.preprocessed_X)))**2)
#     fig, axs = plt.subplots(2,1)
#     axs[0].imshow(self.X.T)
#     axs[0].title.set_text("Original")
#     axs[1].imshow(self.reduced_X.T)
#     axs[1].title.set_text(f"Reduction: {reduction} \%; PSNR: {psnr:0.2f} dB")

#   def get_important_features(self, n_var, d_min = 0):
#       """
#       A method to find the most relevant features in the Xset. This is
#       done by sorting the loading values along each principal component,
#       which is nothing else than the correlation of the original variable
#       and the principal component.

#       Parameters
#       ----------
#       n_var : int
#         Number of most important features per component.
#       d_min : int, optional
#         Minimum difference between indices of important features. The default is 0.

#       Returns
#       -------
#       dict : dict
#         Dictionary of variable(s), loading (correlation) value(s) and index (indices) per component.

#       """
#       most_important_features = {}

#       for component in range(self.n_components):

#         i_sorted = np.flip(np.argsort(np.abs(self.loadings.T)[:, component]))
#         x_sorted = np.flip(self.features["name"][np.argsort(np.abs(self.loadings.T)[:, component])])
#         a_sorted = np.flip(self.loadings[component][np.argsort(np.abs(self.loadings.T)[:, component])])

#         #generating dictionary with information about variable, correlation and index
#         count = 0
#         x, a, i = [], [], []

#         for pos in range(len(i_sorted)):
#           if count == 0:
#               x.append(x_sorted[pos])
#               a.append(a_sorted[pos])
#               i.append(i_sorted[pos])
#               count += 1

#           elif np.sum((np.abs(np.array(x) - x_sorted[pos])) > d_min) == count:
#               x.append(x_sorted[pos])
#               a.append(a_sorted[pos])
#               i.append(i_sorted[pos])
#               count += 1

#           if count >= n_var:
#               most_important_features[f"PC{component + 1}"] = {"var": x, "a": a, "i": i}
#               break

#       return most_important_features

#   def get_contribution(self):
#     if not self._contribution:
#       if not self._loadings:
#         self.get_loadings()
#       self.contribution = np.empty(self.loadings.shape)

#       for i in range(len(self.loadings)):
#         self.contribution[i] = (np.abs(self.loadings[i])
#                                 / np.sum(np.abs(self.loadings[i])))
#       self._contibution = True