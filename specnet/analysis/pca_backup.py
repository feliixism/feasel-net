# pca backup
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.linalg import eigh
import pandas as pd
from spec_net.preprocess.data import covariance, correlation, resample

class PCA:
    def __init__(self, data, n_components = 2, rowvar = True, lower_boundary = None, upper_boundary = None, standardize_inputs = False):
        """
        A class environment for analysing data using the theory of principle component analysis.

        Parameters
        ----------
        data : float
            Data array.
        n_components : int, optional
            Sets the number of principle components. The default is 2.
        rowvar : boolean, optional
            Indicates the axis of the variables. If variables are described by the columns then set to 'False'. 
            The default is True.
        lower_boundary : float, optional
            Sets the lower boundary value to specified value. The default is None.
        upper_boundary : float, optional
            Sets the upper boundary value to specified value. The default is None.
        standardize : boolean, optional
            Optional standardization of the data along sample axis if set to 'True'. The default is False.

        Returns
        -------
        None.

        """
        if isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data
        self.rowvar = rowvar
        if self.rowvar == False:
            self.data = self.data.T
        self.original_data = self.data
        if (lower_boundary and upper_boundary) is None:
            self.x = np.arange(len(self.data))
        elif upper_boundary is None and lower_boundary is not None:
            self.x = np.arange(len(self.data)) + lower_boundary
        elif (upper_boundary and lower_boundary) is not None:
            self.x = np.linspace(lower_boundary, upper_boundary, len(self.data))
        self.n_variables, self.n_samples = self.data.shape
        self.n_components = n_components
        self._eigs = False
        self._preprocessed = False
        self._standardized = standardize_inputs
        
    def preprocess(self):
        m, n = self.data.shape
        mu = np.empty([m])
        sigma = np.empty([m])
        for i in range(m):
            mu[i] = np.mean(self.data[i])
            # ist das wirklich ne vernünftige Idee?!!?!
            if self._standardized == True:
                sigma[i] = np.sqrt(1/(len(self.data[i]) - 1) * np.sum(self.data[i]**2))
            else:
                sigma[i] = 1
            self.data[i] = (self.data[i] - mu[i]) / sigma[i]
        self.preprocessed = True
        return self.data
    
    # internal getters
    def get_eigs(self):
        self.cov = covariance(self.data)
        self.corr = correlation(self.data)
        self.evals, self.evecs = eigh(self.cov)
        self.evals = np.flip(self.evals)
        self.get_overall_explained_variance()
        self.evecs = np.flip(self.evecs.T, 0)
        self._eigs = True
        
    def get_overall_explained_variance(self):
        self.overall_explained_variance = self.evals / np.sum(self.evals)
    
    def get_loadings(self):
        ' variance for each pc: sqrt eval'
        if self._eigs == False:
            self.get_eigs()
        self.loadings = np.empty([self.n_components, self.n_variables])
        for i in range(self.n_components):
            self.loadings[i] = self.evecs[i] / np.sqrt(self.evals[i])
    
    def get_scores(self):
        # predicted values found by projecting original points into the hyperplane defined by the principal components
        # linear combination of initial values
        self.get_loadings()
        self.scores = np.empty([self.n_samples, self.n_components])
        for i in range(self.n_samples):
            for j in range(self.n_components):
                score = np.sum(self.loadings[j] * self.data[:, i])
                self.scores[i, j] = score
    
    def get_corr_input_pc(self):
        # safe ein fehler drin
        self.interpret = np.zeros([self.n_variables, self.n_components])
        for variable in range(self.n_variables):
            sigma_x = np.sqrt(1/(len(self.original_data[variable]) - 1) * np.sum(self.original_data[variable]**2))
            for component in range(self.n_components):
                sigma_y = np.sqrt(self.evals[component])
                for sample in range(self.n_samples):
                    self.interpret[variable, component] += 1 / (self.n_samples - 1) * (self.original_data[variable, sample]) * (self.loadings[component, variable]) / (sigma_x * sigma_y)
        return self.interpret
    
    # visualization    
    def scree_plot(self):
        """
        A scree plot shows the individual weights of each Principle Component (PC).
        All PCs sum up to 1. The scree plot shows only as much PCs as initiallized in the class.

        Returns
        -------
        None.

        """
        if self._eigs == False:
            self.get_eigs()
        list_PCs = []
        plt.figure("Scree Plot")
        for i in range(self.n_components):
            list_PCs.append(f"PC{i + 1}")    
        oev = self.overall_explained_variance[0 : self.n_components] * 100
        plt.bar(list_PCs, oev)
        plt.ylabel("Contribution [$\%$]")
        plt.grid(True)
        for i in range(len(list_PCs)):
            plt.text(list_PCs[i], oev[i] + 0.15, f"{np.round(oev[i], 2)}", horizontalalignment = "center")
    
    def loadings_lineplot(self, cummulativ = False):
        if self._eigs == False:
            self.get_eigs()
        self.get_loadings()
        fig = plt.figure("Loadings Lineplot")
        plt.clf()
        ax = fig.add_subplot(111)
        for i in range(self.n_components):
            ax.plot(self.x, self.loadings[i], label = f"PC{i + 1}, {np.round(self.evals[i] / np.sum(self.evals) * 100, 1)}$\%$")
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("Wavenumber [cm$^{-1}$]")
        ax.set_xlim(self.x[0], self.x[-1])
        plt.show()
    
    def score_lineplot(self):
        if self._eigs == False:
            self.get_eigs()
        self.get_scores()
        fig = plt.figure("Scores Lineplot")
        ax = fig.add_subplot(111)
        for i in range(self.n_components):
            ax.plot(self.scores[ : , i], label = f"PC{i + 1}, {np.round(self.evals[i] / np.sum(self.evals) * 100, 1)}$\%$")
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("Sample")
    
    def loadings_plot(self, labels = True, num = None, points_of_interest = None, weighted = False):
        offset = 0.01
        if self._eigs == False:
            self.get_eigs()
        self.get_loadings()
        if num is not None:
            x = resample(self.x, num)
            loadings = resample(self.loadings, num, axis = 1)
        else:
            x = self.x
            loadings = self.loadings
        if weighted == True:
            loadings[0] = loadings * self.evals[0] / np.sum(self.evals[0:2])
            loadings[1] = loadings * self.evals[1] / np.sum(self.evals[0:2])
        fig = plt.figure("Loadings")
        ax = fig.add_subplot(111)
        ax.scatter(loadings[0], loadings[1])
        if points_of_interest is not None:
            if type(points_of_interest) != list:
                points_of_interest = [points_of_interest]
            for point in points_of_interest:
                idx = np.abs(x - point).argmin()
                ax.scatter(loadings[0, idx], loadings[1, idx], color = "orange")
        labels = np.round(x, 1)
        mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))
        ax.plot([0, 0], [np.amin(loadings[1]) - offset, np.amax(loadings[1]) + offset], color = "k")
        ax.plot([np.amin(loadings[0]) - offset, np.amax(loadings[0]) + offset], [0, 0], color = "k")
        ax.set_ylabel("PC2")
        ax.set_xlabel("PC1")
        ax.set_ylim(np.amin(loadings[1]) - offset, np.amax(loadings[1]) + offset)
        ax.set_xlim(np.amin(loadings[0]) - offset, np.amax(loadings[0]) + offset)
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.show()
    
    def score_plot(self, label):
        self.get_scores()
        offset = np.amax(np.abs(self.scores)) * 0.1
        labels = np.unique(label)
        fig = plt.figure("Scores")
        plt.clf()
        ax = fig.add_subplot(111)
        for i in range(len(labels)):
            mask = np.where(label == labels[i])[0]
            ax.scatter(self.scores[mask, 0], self.scores[mask, 1], label = f"{labels[i]}")
        ax.axhline(0, color = "k")
        ax.axvline(0, color = "k")
        ax.set_ylabel("PC2")
        ax.set_xlabel("PC1")
        ax.set_ylim(-np.amax(np.abs(self.scores)) - offset, np.amax(np.abs(self.scores)) + offset)
        ax.set_xlim(-np.amax(np.abs(self.scores)) - offset, np.amax(np.abs(self.scores)) + offset)
        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal")
        fig.show()
    
    def biplot(self, label, points_of_interest = None):
        self.get_scores()
        offset = np.amax(np.abs(self.scores)) * 0.1
        labels = np.unique(label)
        fig = plt.figure("Biplot")
        plt.clf()
        ax1 = fig.add_subplot(111)
        for i in range(len(labels)):
            mask = np.where(label == labels[i])[0]
            ax1.scatter(self.scores[mask, 0], self.scores[mask, 1], label = f"{labels[i]}")
        ax1.axhline(0, color = "k")
        ax1.axvline(0, color = "k")
        if points_of_interest is not None:
            scale = np.amax([np.abs(np.amin(self.scores)), np.amax(self.scores)]) / np.amax([np.abs(np.amin(self.loadings)), np.amax(self.loadings)])
            def scaling_down(x, scale = scale):
                return x * scale
            def scaling_up(x, scale = scale):
                return x / scale
            ax2 = ax1.secondary_xaxis("top", functions=(scaling_up, scaling_down))
            ax3 = ax1.secondary_yaxis("right", functions=(scaling_up, scaling_down))
            
            if type(points_of_interest) != list:
                points_of_interest = [points_of_interest]
            for point in points_of_interest:
                idx = np.abs(self.x - point).argmin()
                ax1.annotate(f"{np.round(self.x[idx], 1)}", xy = (0, 0), xytext = (self.loadings[0, idx] * scale, self.loadings[1, idx] * scale), va = "center", ha = "center", color = "k", arrowprops=dict(arrowstyle="<|-", shrinkA=0, shrinkB=0))
            ax2.set_label("PC1")
        ax1.set_xlim(-np.amax(np.abs(self.scores)) - offset, np.amax(np.abs(self.scores)) + offset)
        ax1.set_ylim(-np.amax(np.abs(self.scores)) - offset, np.amax(np.abs(self.scores)) + offset)
        ax1.set_ylabel("PC2")
        ax1.set_xlabel("PC1")
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect("equal")
        fig.show()
        
        