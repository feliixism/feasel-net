import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mplcursors
from scipy.linalg import eigh
from spec_net.preprocess.data import covariance, correlation
from spec_net.plot.pca import LinearTransformation
# from spec_net.plot.visualization import format_label_string_with_exponent
from .svd import SVD

class PCA:
    def __init__(self, X, y = None, n_components = 2, features = None, rowvar = True, standardize = False, solver = "eigs"):
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
        n_components : int, optional
            Sets the number of principal components. The default is 2.
        features : float
            1D-Array with features. If 'None', the features are incremented 
            from 0 to N-1. The default is 'None'.
        rowvar : boolean, optional
            Indicates the axis of the features. If features are described by 
            the columns then set to 'False'. 
            The default is True.
        standardize : boolean, optional
            Optional standardization of the X along sample axis if set to 
            'True'. The default is False.

        Returns
        -------
        PCA class object.

        """
        self.X = X
        self.y = y
        
        self._rowvar = rowvar
        if self._rowvar == False:
            self.X = self.X.T
        
        if y is not None:
            classes = np.unique(y)
            self.n_classes = len(classes)
            self.classes = np.array([classes, np.arange(self.n_classes)])
            self.log = self.get_log()      
        
        #number features
        self.n_features, self.n_samples = self.X.shape
        self.n_components = n_components
        
        if features is None:
            idx = np.arange(self.n_features)
            self.features = {"idx": idx, "name": idx}
        else:
            idx = np.arange(self.n_features)
            features_label = np.array([features]).squeeze()
            features_label = np.char.replace(features_label, "_", "\_")
            self.features = {"idx": idx, "name": features_label}
            
        #solver information
        self.solver = solver
        
        #boolean features
        self._standardized = standardize
        self._eigs, self._loadings, self._scores, self._preprocessed, self._reduced = False, False, False, False, False
        
        #embed plot class
        self.plot = LinearTransformation(self)
        
    def __repr__(self):
        return f"PCA(n_features: {self.n_features}, n_samples: {self.n_samples}, n_components: {self.n_components}, standardized: {self._standardized})"
    
    def get_log(self):
        dict={}
        for class_ in self.classes[0]:
            data = self.X[ : , np.argwhere(self.y == class_)].squeeze()
            dict[f"{class_}"] = {"data": data}
        return dict
    
    def preprocess(self):
        """
        Mean-centering of the X. If 'standardized' is 'True', the X is 
        also standardized by dividing it by the standard deviation of each 
        variable.
        
        Returns
        -------
        None.

        """
        self.mu = np.mean(self.X, axis = 1, keepdims = True)
        self.sigma = np.std(self.X, axis = 1, ddof = 1, keepdims = True)
        if self._standardized == True:
            sigma = self.sigma  
        else:
            sigma = np.ones(self.mu.shape)
        
        self.preprocessed_X = (self.X - self.mu) / sigma
        self._preprocessed = True
        
        #log information
        for class_ in self.classes[0]:
            data = self.preprocessed_X[ : , np.argwhere(self.y == class_)[:, 0]]
            self.log[f"{class_}"]["N"] = data.shape[-1]
            self.log[f"{class_}"]["data"] = data
            self.log[f"{class_}"]["mu"] = np.mean(data, axis = 1, keepdims = True)
            self.log[f"{class_}"]["sigma"] = np.std(data, axis = 1, ddof = 1, keepdims = True)
    
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
        if not self._preprocessed:
            self.preprocess()
        
        if self.solver == "eigs":
            if self._standardized:
                evals, evecs = eigh(correlation(self.preprocessed_X), lower = False)
            else:
                evals, evecs = eigh(covariance(self.preprocessed_X), lower = False)
            evals = np.flip(evals)
            evecs = np.flip(evecs.T, 0)
        
        elif self.solver == "svd": # Flipped signs every second row in evecs. Why?
            svd = SVD(self.preprocessed_X.T)
            U, Sigma, Vh = svd()
            evals, evecs = np.diagonal(Sigma)**2 / (self.n_samples - 1), Vh
        
        self.var_scores = evals**2
        self.evals, self.evecs = evals, evecs
        self.get_overall_explained_variance()
        self.evals, self.evecs = evals, evecs
        self._eigs = True
        
    def get_overall_explained_variance(self):
        """
        Calculates the overall explained variance of each principal component.

        Returns
        -------
        None.

        """
        self.overall_explained_variance = self.evals / np.sum(self.evals) * 100
        self.cummulative_explained_variance = np.zeros(self.evals.shape)
        for i in range(len(self.evals)):
            if i > 0:
                self.cummulative_explained_variance[i] = self.overall_explained_variance[i] + self.cummulative_explained_variance[i - 1]
            else:
                self.cummulative_explained_variance[i] = self.overall_explained_variance[i]
    
    def get_loadings(self):
        """
        Calculates the so-called loading-factors of each principal component.
        The loadings can be interpreted as weights for the orthogonal linear
        transformation of the original features into the new latent features.

        Returns
        -------
        None.

        """
        if not self._eigs:
            self.get_eigs()
        self.loadings = self.evecs[ : self.n_components] * np.sqrt(np.expand_dims(self.evals[ : self.n_components], 0).T)
        self._loadings = True
        
    def get_scores(self):
        """
        Calculates the values of the original features projected into the new
        latent variable space described by the corresponding principal 
        components.

        Returns
        -------
        None.

        """
        if not self._loadings:
            self.get_loadings()
        self.scores = (self.loadings @ self.preprocessed_X).T
        for class_ in self.classes[0]:
            mask = np.where(self.y == class_)[0]
            self.log[f"{class_}"]["scores"] = self.scores[mask]
            self.log[f"{class_}"]["scores_var"] = np.var(self.log[f"{class_}"]["scores"], axis = 0)
            self.log[f"{class_}"]["scores_mean"] = np.mean(self.log[f"{class_}"]["scores"], axis = 0)
        self.var_scores = np.var(self.scores, axis = 0, ddof = 1)
        self._scores = True
        
    def get_reduced_X(self):
        if self._scores == False:
            self.get_scores()
        self.reduced_X = (self.scores @ self.evecs[:self.n_components]).T
        self._reduced = True
    
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
            Minimum difference between indices of important features. The default is 0.

        Returns
        -------
        dict : dict
            Dictionary of variable(s), loading (correlation) value(s) and index (indices) per component.

        """
        most_important_features = {}
        for component in range(self.n_components):
            i_sorted = np.flip(np.argsort(np.abs(self.loadings.T)[:, component]))
            x_sorted = np.flip(self.features["name"][np.argsort(np.abs(self.loadings.T)[:, component])])
            a_sorted = np.flip(self.loadings[component][np.argsort(np.abs(self.loadings.T)[:, component])])
            
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
                    most_important_features[f"PC{component + 1}"] = {"var": x, "a": a, "i": i}
                    break
        return most_important_features
    
    # def get_cos2(self, direction = "variable"):
    #     self.get_loadings()
    #     cos2 = np.empty(self.loadings.shape)
    #     if direction == "variable":
    #         for i in range(self.n_components):
    #             cos2[i] = self.loadings[i]**2 / np.sum(self.loadings[i]**2) * 100
    #     elif direction == "component":
    #         for i in range(self.n_features):
    #             cos2[:, i] = self.loadings[:, i]**2 / np.sum(self.loadings[:, i]**2) * 100
    #     else:
    #         raise ValueError("Please specify a valid direction. Valid input is 'component' or 'variable'.")
    #     self.cos2 = cos2
        