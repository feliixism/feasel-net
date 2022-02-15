# import numpy as np
# from scipy.linalg import eigh
# from spec_net.preprocess.data import covariance, correlation, scatter_matrix
# from .pca import PCA
# from .svd import SVD
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from .plots.utils._visualization import contour_gauss

# class LDA(PCA):
#     def __init__(self, X, y, n_components = 2, features = None, rowvar = True, standardize = False, solver = "eigs", two_staged = False):
#         self._two_staged = two_staged
#         self._estimated = False
#         self.classes = np.unique(y)
#         self.n_classes = len(self.classes)
#         super().__init__(X, y, n_components, features, rowvar, standardize, solver)
#         self.log = self.get_log()

#     def __repr__(self):
#         return f"LDA(n_features: {self.n_features}, n_samples: {self.n_samples}, n_classes: {self.n_classes}, n_components: {self.n_components}, standardized: {self._standardized})"

#     def get_log(self):
#         dict={}
#         for class_ in self.classes[0]:
#             data = self.X[ : , np.argwhere(self.y == class_)].squeeze()
#             dict[f"{class_}"] = {"data": data, "a-priori": len(data[0]) / len(self.X[0])}
#         return dict

#     # estimating gaussian parameters:
#     def preprocess(self):
#         """
#         Mean-centering of the data. If 'standardized' is 'True', the data is
#         also standardized by dividing it by the standard deviation of each
#         variable.

#         Returns
#         -------
#         None.

#         """
#         if self._two_staged:
#             self.get_pca(10)
#             self.X = self.pca.evecs @ self.X

#         self.mu = np.mean(self.X, axis = 1, keepdims = True)
#         self.sigma = np.std(self.X, axis = 1, ddof = 1, keepdims = True)

#         if self._standardized == True:
#             sigma = self.sigma

#         else:
#             sigma = np.ones(self.mu.shape)

#         self.preprocessed_X = (self.X - self.mu) / sigma
#         self.covariance = covariance(self.preprocessed_X)

#         # update of sigma and mu
#         self.prep_mu = np.mean(self.preprocessed_X, axis = 1, keepdims = True)
#         self.prep_sigma = np.std(self.X, axis = 1, ddof = 1, keepdims = True)

#         self._preprocessed = True

#     def estimate(self): # handbook of statistics (Cheriet 2013)
#         # scatter within class, scatter between classes:
#         if not self._preprocessed:
#             self.preprocess()

#         self.scatter_within = np.zeros([len(self.X), len(self.X)])
#         self.scatter_between = np.zeros([len(self.X), len(self.X)])

#         for class_ in self.classes[0]:
#             data = self.preprocessed_X[ : , np.argwhere(self.y == class_)[:, 0]]
#             self.log[f"{class_}"]["N"] = data.shape[-1]
#             self.log[f"{class_}"]["prior"] = data.shape[-1] / self.X.shape[-1]
#             self.log[f"{class_}"]["data"] = data
#             self.log[f"{class_}"]["mu"] = np.mean(data, axis = 1, keepdims = True)
#             self.log[f"{class_}"]["sigma"] = np.std(data, axis = 1, ddof = 1, keepdims = True)
#             self.log[f"{class_}"]["scatter"] = scatter_matrix(data)
#             self.log[f"{class_}"]["covariance"] = self.log[f"{class_}"]["scatter"] / (self.log[f"{class_}"]["N"] - 1)
#             self.scatter_within += self.log[f"{class_}"]["scatter"]
#             self.scatter_between += self.log[f"{class_}"]["N"] * (self.log[f"{class_}"]["mu"] - self.prep_mu) @ (self.log[f"{class_}"]["mu"] - self.prep_mu).T
#         self.common_covariance = self.scatter_within / (self.n_samples - self.n_classes)
#         self._estimated = True

#     def get_pca(self, n_components = None):
#         if n_components is None:
#             n_components = int(self.n_samples * 0.1)
#         self.pca = PCA(self.X, self.y, n_components = n_components, standardize = self._standardized, solver = self.solver)
#         self.pca.get_eigs()

#     # internal getters
#     def get_eigs(self):
#         """
#         Solves the eigenvalue problem of the quadratic covariance or
#         correlation matrix respectively. The order of evecs and evals is sorted
#         by the magnitude of the evals from highest to lowest.

#         Returns
#         -------
#         None.

#         """
#         if not self._estimated:
#             self.estimate()

#         sw_sb = np.linalg.inv(self.scatter_within) @ self.scatter_between

#         if self.solver == "eigs":
#             evals, evecs = np.linalg.eig(sw_sb)
#             evals = np.abs(evals)
#             idx = np.flip(np.argsort(evals))
#             evals, evecs = evals[(idx,)], evecs[:, (idx,)].squeeze().real.T

#         elif self.solver == "svd": # Flipped signs every second row in evecs. Why?
#             self.svd = SVD(sw_sb)
#             U, Sigma, Vh = self.svd()
#             evals, evecs = np.square(np.diagonal(Sigma)) / (self.n_samples - 1), Vh

#         self.var_scores = evals**2
#         self.evals, self.evecs = np.array(evals, ndmin=2), evecs
#         self.get_overall_explained_variance()
#         self._eigs = True

#     def get_scores(self):
#         """
#         Calculates the values of the original features projected into the new
#         latent variable space described by the corresponding principal
#         components.

#         Returns
#         -------
#         None.

#         """
#         if not self._loadings:
#             self.get_loadings()
#         self.scores = (self.loadings @ self.preprocessed_X).T
#         for class_ in self.classes[0]:
#             mask = np.where(self.y == class_)[0]
#             self.log[f"{class_}"]["scores"] = self.scores[mask]
#             self.log[f"{class_}"]["scores_var"] = np.var(self.log[f"{class_}"]["scores"], axis = 0)
#             self.log[f"{class_}"]["scores_mean"] = np.mean(self.log[f"{class_}"]["scores"], axis = 0)
#         self.var_scores = np.var(self.scores, axis = 0, ddof = 1)
#         self._scores = True

#     def predict(self, x):
#         if not self._eigs:
#             self.get_eigs()

#         x = np.array(x, ndmin = 2)

#         if not self._rowvar:
#             x = x.T

#         x = (x - self.mu) / self.sigma #preprocessing step is applied to test data

#         if self._two_staged:
#             x = x @ self.pca.loadings

#         delta = self.decision_rule(x)

#         print("Results of Delta-Function:")
#         for i, d in enumerate(delta):
#             print(f"\t{self.classes[i]}: ", d)
#         print("\tMost likely to be: ", self.classes[np.argmax(delta)])

#         return self.classes[np.argmax(delta)]

#     def decision_rule(self, x, components = None, transformed = False):

#         if components is not None:
#             loadings = self.loadings[components]
#         else:
#             loadings = self.loadings

#         if not transformed:
#             # transformation into LDA space
#             x = loadings @ x

#         try:
#             common_covariance = loadings @ self.common_covariance @ loadings.T
#             common_covariance_ = np.linalg.inv(common_covariance)

#             # actual decision rule
#             delta = np.zeros([self.n_classes])
#             for i, class_ in enumerate(self.classes[0]):
#                 mu = (loadings @ self.log[f"{class_}"]["mu"]).T
#                 delta[i] = np.log10(self.log[f"{class_}"]["prior"]) - 1/2 * mu @ common_covariance_ @ mu.T + x.T @ common_covariance_ @ mu.T

#             return delta
#         except:
#             return np.zeros([self.n_classes])