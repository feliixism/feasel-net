import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
from spec_net.plot.utils._utils import click_connect
from spec_net.preprocess.data import covariance
from spec_net.plot.utils._visualization import confidence
from spec_net.plot.utils import _default

class LinearTransformation:
    def __init__(self, analysis):
        """
        The base plot class for all implemented linear transformation 
        algorithms (e.g. PCA and LDA).

        Parameters
        ----------
        analysis : class
            The chosen linear transformation class. Accepts 'PCA' and 'LDA'.

        Raises
        ------
        TypeError
            Error, if instantiated analysis class is not of kind 'PCA' or 
            'LDA'.

        Returns
        -------
        None.

        """
        self.settings = _default.AdvancedPlot()
        self.settings.set_default()
        self.c_cmap = self.settings.get_cmap()
        self.im_cmap = self.settings.im_cmap
        self.analysis = analysis
        self.set_index()
        
        if "PCA" in repr(analysis):
            self.component_type = "PC"
        elif "LDA" in repr(analysis):
            self.component_type = "LD"
        else:
            raise TypeError("Plot function is not supported for other"
                            "analysis objects than 'PCA' or 'LDA'.")
        
        idx, components = [], []
        
        for i in range(self.analysis.n_components):
            idx.append(i)
            components.append(f"{self.component_type}$_{int(i + 1)}$")
        
        self.components = {'idx': idx, 'name': components}
        self._labels = False
        
    def set_index(self, idx=None):
        """
        Sets the indices for the components that are being plotted.

        Parameters
        ----------
        idx : list or int
            The index or indices of the plotted component. The default is None.

        Returns
        -------
        None.

        """
        if idx is None:
            idx = np.arange(self.analysis.n_components)
        
        else:
            idx = np.array(idx) - 1
        
        self.idx = np.array(idx)
        
    def get_labels(self):
        if not self._labels:
            labels = {'idx': [], 'name': []}
            for i in range(self.analysis.n_components):
                labels['idx'].append(i)
                labels['name'].append(f"{self.component_type}$_{int(i+1)}$ "
                                      f"({np.round(self.analysis.oev[0,i],2)}$"
                                      "\,\%$)")
            labels["idx"] = np.array(labels["idx"])
            labels["name"] = np.array(labels["name"])
            self.labels = labels
            self._labels = True
    
    def scree(self, type = "bar", show_average = False):
        """
        The scree plot shows the individually contributed variance explained 
        for each Principal Component (PC) or Linear Discriminant (LD) in 
        sorted order.
        All n contributions sum up to 1, whereas the scree plot only shows the 
        as variances for as much PCs or LDs as originally initialized in the 
        class.
        
        Parameters
        ----------
        type : str, optional
            Determines the visualization type for the scree plot. Valid
            options are: 'bar', 'line' or 'both'. The default is 'bar'.
        show_average : boolean, optional
            Displays average of all n contributions. The default is 'False'.
    
        Returns
        -------
        None.
    
        """
        
        if not self.analysis._eigs:
            self.analysis.get_eigs()
        self.get_labels()
        
        fig = plt.figure("Scree Plot")
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_ylabel("Variance Explained [$\%$]")
        ax.grid(True)
        
        oev = self.analysis.oev[0, 0 : self.analysis.n_components]
        cev = self.analysis.cev[0, 0 : self.analysis.n_components]
        
        #ticklabels
        ticklabels = []
        for i in range(len(self.components["name"])):
            ticklabels.append(self.labels["name"][i])
        
        if type == "bar":
            ax.bar(ticklabels, oev)
        
        elif type == "cummulative":
            ax.plot(cev, color = "k", marker = ".")
            ax.set_xticks(self.components["idx"])
            ax.set_xticklabels(ticklabels)
        
        elif type == "both":
            ax.bar(ticklabels, oev)
            ax.plot(cev, color = "k", marker = ".")
        
        if show_average:
            avg_var = np.sum(self.analysis.oev[0] / 
                             len(self.analysis.oev[0]))
            ax.axhline(y = avg_var, color = "k", ls = "-.", 
                       label = "Average Variance", zorder = 0)
            
            if type == "bar" or type == "both":
                ax.bar(ticklabels, 
                       oev * np.less(oev, np.ones(oev.shape) * avg_var))
            
            ax.legend()
                
        ax.set_ylim(0, oev[0] * 1.1)
        if type == "cummulative" or type == "both":
            ax.set_ylim(0, cev[-1] * 1.1)
        fig.tight_layout()
        fig.show()
        
    def loadings_line(self, x_label = "Features", idx=None):
        """
        The loadings line plot shows the correlation of the original features 
        with the newly generated components (LDs or PCs). The importance of
        of each feature for the analysis can be extracted out of the line plot.

        Returns
        -------
        None.

        """
        
        if not self.analysis._loadings:
            self.analysis.get_loadings()
        self.get_labels()
        
        fig = plt.figure("Loadings Lineplot")
        fig.clf()
        ax = fig.add_subplot(111)
        
        if idx:
            self.set_index(idx)
        
        for i in self.idx:
            ax.plot(self.analysis.features["idx"], 
                    self.analysis.loadings[i], 
                    label = f"{self.labels['name'][i]}")
            
        ax.legend(loc = "upper right")
        ax.grid(True)
        ax.axhline(0, color = "k", zorder = 0)
        ax.set_ylabel("Loadings [-]")
        ax.set_xlabel(f"{x_label}")
        ax.set_xlim(self.analysis.features["idx"][0], 
                    self.analysis.features["idx"][-1])
        ax.set_xticks(self.analysis.features["idx"])
        ax.set_xticklabels([f"{i}" for i in self.analysis.features["name"]], 
                           rotation = 45, ha = "right")
        fig.tight_layout()
        fig.show()
    
    def contribution_bar(self, 
                         component = 1, 
                         show_average = False, 
                         x_label = "Features"):
        """
        The contribution plot is a visualization tool, that sorts the features
        according to their loading values for the given component and plots
        the features in descending order.

        Parameters
        ----------
        component : int, optional
            Defines the displayed component. The default is 1.

        Returns
        -------
        None.

        """
        fig = plt.figure("Contribution Bar Plot")
        fig.clf()
        ax = fig.add_subplot(111)
        
        component = component - 1
        
        if not self.analysis._contribution:
            self.analysis.get_contribution()
        self.get_labels()
        
        height = self.analysis.contribution[component] * 100
        idx = np.flip(np.argsort(height))
        features = self.analysis.features["name"][idx]
        ax.bar(features, height = height)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation = 45, ha = "right")
        if show_average:
            ax.axhline(np.mean(height), color = "k", ls = "-.", label='average',
                       zorder = 0)
            ax.legend(loc = "upper right")
        ax.set_ylabel("Conribution to " 
                      f"{self.labels['name'][component]}")
        ax.set_xlabel(f"{x_label}")
        ax.set_ylim(0, height[0] * 1.1)
        ax.grid(True)
        fig.tight_layout()
        fig.show()
        
    def contribution_bars(self, 
                          components = [1, 2], 
                          show_average = False, 
                          x_label = "Features"):
        """
        The contribution plot is a visualization tool, that sorts the features
        according to their loading values for the given component and plots
        the features in descending order.

        Parameters
        ----------
        component : int, optional
            Defines the displayed component. The default is 1.

        Returns
        -------
        None.

        """
        fig = plt.figure("Contribution Bars Plot")
        fig.clf()
        ax = fig.add_subplot(111)
        
        components = np.array(components) - 1
        width = 0.8 / len(components)
        
        if not self.analysis._contribution:
            self.analysis.get_contribution()
        self.get_labels()
        
        heights = self.analysis.contribution[components]
        
        heights = heights / np.sum(heights) * 100  
        height = np.sum(heights, axis=0)
        
        idx = np.flip(np.argsort(np.abs(height)))
        height = height[idx] / np.sum(height) * 100
        features = self.analysis.features["name"][idx]
        x = np.arange(len(self.analysis.loadings[0]))
        ax.bar(x, height=np.abs(height))
        for i, h in enumerate(heights):
            ax.bar(x-0.2 + i * width, 
                   height=heights[i, idx], 
                   label=self.labels["name"][components[i]], width=width)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation = 45, ha = "right")
        if show_average:
            ax.axhline(np.mean(height), color = "k", ls = "-.", zorder = 0)
            ax.legend(loc = "upper right")
        ax.set_ylabel("Conribution to Components [$\%$]")
        ax.set_xlabel(f"{x_label}")
        ax.set_ylim(0, height[0] * 1.1)
        ax.grid(True)
        fig.tight_layout()
        fig.show()
    
    def plot_features_of_interest(self, FOI, x, y):
        if FOI is None:
            return
        if isinstance(FOI, list or np.ndarray):
            FOI = [FOI]
        for P in FOI:
            idx = np.argwhere(self.analysis.features["name"] == P)
            plt.scatter(self.analysis.loadings[x, idx], 
                        self.analysis.loadings[y, idx])
        
    def contribution_circle(self, x = 1, y = 2, FOI = None):
        """
        Similarily to the loadings lineplot, the contribution circle is a 
        visualization tool to show how much each variable contributes to the 
        corresponding component (PC or LD).
        It is only suitable for standardized datasets, since it uses an unit
        circle for comparison. For unstandardized data use '.loadings_plot()'.

        Parameters
        ----------
        x : int, optional
            Defines the displayed component in x direction. The default is 1.
        y : int, optional
            Defines the displayed component in y direction. The default is 2.
        FOI : list, optional
            Mask that can highlight some relevant FOInts. The default is 
            'None'.

        Returns
        -------
        None.

        """
        fig = plt.figure("Contribution Circle")
        ax = fig.add_subplot(111)
        
        if not self.analysis._loadings:
            self.analysis.get_loadings()
        self.get_labels()
        
        x, y = x - 1, y - 1
        
        ax.scatter(self.analysis.loadings[x], 
                   self.analysis.loadings[y])
        self.plot_features_of_interest(FOI = FOI, 
                                     x = x, 
                                     y = y)
        click_connect(ax, self.analysis.features["name"])
        ax.axhline(0, color = "k", zorder = 0)
        ax.axvline(0, color = "k", zorder = 0)
        circle = plt.Circle((0, 0), 1, color = "k", ls = "-.", fill = False)
        ax.add_patch(circle)
        ax.set_ylabel(self.labels["name"][y])
        ax.set_xlabel(self.labels["name"][x])
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(-1.1, 1.1)
        ax.set_aspect("equal")
        # ax.ticklabel_format(style="scientific", scilimits = (0,0))
        ax.grid(True)
        fig.tight_layout()
        fig.show()
        
    def contribution_heatmap(self, idx=None, show_sum=False, weighted=True):
        """
        The contribution heatmap shows all the loading values of the p 
        components (LD or PC) and visualizes their contribution with circles 
        of different sizes. The bigger the circle, the more does this feature 
        contribute to the corresponding component.

        Returns
        -------
        None.

        """
        fig = plt.figure("Contribution Heatmap")
        fig.clf()
        ax = fig.add_subplot(111)
        
        if not self.analysis._contribution:
            self.analysis.get_contribution()
        self.get_labels()

        #mapping circles onto heatmap
        if not idx:
            n = self.analysis.n_components
            idx = np.arange(n)
        else:
            if isinstance(idx, int):
                idx = [idx]
            idx = np.array(idx) - 1
            n = len(idx)
        contribution = self.analysis.contribution[idx]
        
        if show_sum:
            weights = np.ones([n, 1]) / n
            if weighted:
                evals = self.analysis.evals[:, idx]
                weights = (evals/np.sum(evals)).T
            con_sum = np.sum(contribution*weights, axis=0, keepdims=True)
            n += 1 # for additional axis
            contribution = np.concatenate([contribution, con_sum], 
                                          axis = 0)
        con_max = np.amax(contribution)
        con_min = np.amin(contribution)
        
        for i in range(self.analysis.n_features):
            for j in range(n):
                con = contribution[j, i]
                circle = plt.Circle((j, i), 
                                    radius=con/(2*con_max),
                                    color=self.im_cmap(con/con_max))
                ax.add_patch(circle)
        
        #setup plot axes
        ax.set_xticks(np.arange(n))
        xticks = list(np.array(self.components["name"])[idx])
        if show_sum:
            if weighted:
                xticks.append('$\Sigma_w$')
            else:
                xticks.append('$\Sigma$')
        ax.set_xticklabels(xticks)
        ax.set_yticks(self.analysis.features["idx"])
        ax.set_yticklabels(self.analysis.features["name"])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, len(self.analysis.features["idx"]) - 0.5)
        ax.set_aspect("equal")
        ax.grid(False)
        
        #rastering
        for i in range(len(self.analysis.features["idx"])):
            ax.axhline(i + 0.5, color="k", lw=0.5)
        for i in range(n):
            ax.axvline(i + 0.5, color="k", lw=0.5)
        
        #colormap and legend
        ticks = np.round(np.linspace(0, con_max*100, 5), 1)
        norm = mpl.colors.Normalize(vmin=0, vmax=ticks[-1])
        sm = plt.cm.ScalarMappable(cmap = self.im_cmap, norm = norm)
        sm.set_array([])
        ticks = np.round(np.linspace(0, con_max*100, 5), 1)
        cbar = ax.figure.colorbar(sm, ticks=ticks)
        cbar.ax.set_ylabel("Contribution [$\%$]")
        fig.tight_layout()
        fig.show()
    
    def loadings(self, x = 1, y = 2, FOI = None):
        """
        Similarily to the loadings lineplot, the loadings plot is a 
        visualization tool to show how much each variable contributes to the 
        corresponding component (PC or LD).

        Parameters
        ----------
        x : int, optional
            Defines the displayed component in x direction. The default is 1.
        y : int, optional
            Defines the displayed component in y direction. The default is 2.
        FOI : list, optional
            Mask that can highlight some relevant points. The default is 
            'None'.

        Returns
        -------
        None.

        """
        fig = plt.figure("Loadings Plot")
        fig.clf()
        ax = fig.add_subplot(111)

        if not self.analysis._loadings:
            self.analysis.get_loadings()
        self.get_labels()
        
        x, y = x - 1, y - 1
        
        ax.scatter(self.analysis.loadings[x], 
                   self.analysis.loadings[y])
        self.plot_features_of_interest(FOI = FOI, 
                                        x = x, 
                                        y = y)
        click_connect(ax, self.analysis.features["name"])
        ax.axhline(0, color = "k", zorder = 0)
        ax.axvline(0, color = "k", zorder = 0)
        
        ax.set_ylabel(self.labels["name"][y])
        ax.set_xlabel(self.labels["name"][x])
        
        #plot scale
        max_value = np.amax(np.abs(self.analysis.loadings[[x, y]]))
        offset = max_value * 0.1
        ax.set_ylim(-max_value - offset, max_value + offset)
        ax.set_xlim(-max_value - offset, max_value + offset)
        ax.set_aspect("equal")
        
        ax.grid(True)
        fig.tight_layout()
        fig.show()
        
        return ax 
    
    def scores(self, x = 1, y = 2, projection = False, 
               decision_boundaries = False, confidence_interval = 3):
        """
        The scores plot shows the samples in the new latent feature space 
        defined by the given components.
        The scores plot is the probably most important plot in the analyses
        based on linear transformations, since it directly provides information 
        about the ability to cluster the samples.

        Parameters
        ----------
        x : int, optional
            Defines the displayed component in x direction. The default is 1.
        y : int, optional
            Defines the displayed component in y direction. The default is 2.

        Returns
        -------
        None.

        """
        fig = plt.figure("Scores Plot")#, figsize = (8,8))
        fig.clf()
        if projection:
            spec = gridspec.GridSpec(2, 2, wspace = 0.3, hspace = 0.3, 
                                     width_ratios = (1,5), 
                                     height_ratios = (5,1))
            ax = fig.add_subplot(spec[0,1])
            ax.set_aspect(1)
            ax1 = fig.add_subplot(spec[0,0], sharey = ax)
            ax1.axis("off")
            ax2 = fig.add_subplot(spec[1,1], sharex = ax)
            ax2.axis("off")
        else:
            ax = fig.add_subplot(111)
        
        if not self.analysis._scores:
            self.analysis.get_scores()
        self.get_labels()
        
        x, y = x - 1, y - 1
        
        #plot scale
        max_value = np.amax(np.abs(self.analysis.scores[:, [x, y]]))
        ax.set_ylim(-max_value * 1.1, max_value * 1.1)
        ax.set_xlim(-max_value * 1.1, max_value * 1.1)
        
        if self.component_type == "PC":
            if projection:
                self.scores_pca(ax, x, y, 
                                projection = projection, 
                                projection_axes = [ax1, ax2],
                                confidence_interval = confidence_interval)
            else:
                self.scores_pca(ax, x, y,
                                confidence_interval = confidence_interval)
        
        elif self.component_type == "LD":
            if projection:
                self.scores_lda(ax, x, y,
                                projection = projection, 
                                projection_axes = [ax1, ax2], 
                                decision_boundaries = decision_boundaries,
                                confidence_interval = confidence_interval)
            else:
                self.scores_lda(ax, x, y,
                                projection = projection, 
                                decision_boundaries = decision_boundaries,
                                confidence_interval = confidence_interval)
            
        ax.axhline(0, color = "k", zorder = 0)
        ax.axvline(0, color = "k", zorder = 0)
        ax.set_ylabel(self.labels["name"][y])
        ax.set_xlabel(self.labels["name"][x])
        
        ax.grid(True)
        fig.tight_layout()
        fig.show()
    
    def scores_pca(self, ax, x = 1, y = 2, 
                   projection = False, projection_axes = [None, None], 
                   confidence_interval = 3):
        """
        The scores plot shows the samples in the new latent feature space 
        defined by the given components.
        The scores plot is the probably most important plot in the analyses
        based on linear transformations, since it directly provides information 
        about the ability to cluster the samples.

        Parameters
        ----------
        ax : axis (matplotlib)
            Defines the current axis of the inserted plot.
        x : int, optional
            Defines the displayed component in x direction. The default is 1.
        y : int, optional
            Defines the displayed component in y direction. The default is 2.

        Returns
        -------
        None.

        """
        limit = np.amax(np.abs(self.analysis.scores[:, [x, y]]))
        if self.analysis.y is not None:
            for i in range(self.analysis.n_classes):
                scores = self.analysis.log[f"{self.analysis.classes[0][i]}"]["scores"].T[[x, y]]
                cov = covariance(scores)
                mu = (self.analysis.loadings[[x, y]] @ self.analysis.log[f"{self.analysis.classes[0][i]}"]["mu"]).T.squeeze()
                
                if confidence_interval:
                    confidence(ax, cov, mu, s = confidence_interval)
                
                ax.scatter(scores[0], scores[1], alpha = 0.7, edgecolors = "k", 
                           label = f"{self.analysis.classes[0][i]}")
                ax.scatter(mu[0], mu[1], marker = "X", edgecolors = "k")
                #projection of scatter plot onto x- and y-axis 
                if projection:
                    x = np.linspace(- limit, limit, 100)
                    projection_0 = 1 / np.sqrt(2 * np.pi * cov[0, 0]) * np.exp(-(x - mu[0])**2 / (2 * cov[0, 0]))
                    projection_1 = 1 / np.sqrt(2 * np.pi * cov[1, 1]) * np.exp(-(x - mu[1])**2 / (2 * cov[1, 1]))
                    projection_axes[1].plot(x, -projection_0)
                    projection_axes[0].plot(-projection_1, x)
            ax.legend(loc = "upper right")
        else:
            ax.scatter(self.analysis.scores[:, x], 
                       self.analysis.scores[:, y], 
                       alpha = 0.7, edgecolors = "k")        
    
    def scores_lda(self, ax, x = 1, y = 2, 
                   projection = False, 
                   projection_axes = [None, None], 
                   decision_boundaries = False, 
                   confidence_interval = 3):
        """
        The scores plot shows the samples in the new latent feature space 
        defined by the given components.
        The scores plot is the probably most important plot in the analyses
        based on linear transformations, since it directly provides information 
        about the ability to cluster the samples.

        Parameters
        ----------
        ax : axis (matplotlib)
            Defines the current axis of the inserted plot.
        x : int, optional
            Defines the displayed component in x direction. The default is 1.
        y : int, optional
            Defines the displayed component in y direction. The default is 2.

        Returns
        -------
        None.

        """
        limit = np.amax(np.abs(self.analysis.scores[:, [x, y]]))
        for i in range(self.analysis.n_classes):
            scores = self.analysis.log[f"{self.analysis.classes[0][i]}"]["scores"].T[[x, y]]
            cov = covariance(scores)
            mu = (self.analysis.loadings[[x, y]] @ self.analysis.log[f"{self.analysis.classes[0][i]}"]["mu"]).T.squeeze()
            
            if confidence_interval:
                confidence(ax, cov, mu, s = confidence_interval)
                        
            ax.scatter(scores[0], scores[1], alpha = 0.7, edgecolors = "k", 
                       label = f"{self.analysis.classes[0][i]}")
            ax.scatter(mu[0], mu[1], marker = "X", edgecolors = "k")
            #projection of scatter plot onto x- and y-axis 
            if projection:
                X = np.linspace(- limit, limit, 100)
                projection_0 = 1 / np.sqrt(2 * np.pi * cov[0, 0]) * np.exp(-(X - mu[0])**2 / (2 * cov[0, 0]))
                projection_1 = 1 / np.sqrt(2 * np.pi * cov[1, 1]) * np.exp(-(X - mu[1])**2 / (2 * cov[1, 1]))
                projection_axes[1].plot(X, -projection_0)
                projection_axes[0].plot(-projection_1, X)
        ax.legend(loc = "upper right")
        
        # get decision boundary and confidence levels
        if decision_boundaries:
            sampling = 100
            x_decision = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], sampling)
            y_decision = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], sampling)
            
            delta = np.zeros([sampling, sampling])
            conf = np.zeros([sampling, sampling])
            for i, X in enumerate(x_decision):
                for j, Y in enumerate(y_decision):
                    d = self.analysis.decision_rule(np.array([[X, Y]]).T, 
                                                    components = [x, y], 
                                                    transformed = True)
                    conf[j, i] = np.diff(d[np.argsort(d)[-2:]])
                    delta[j, i] = self.analysis.classes[1, np.argmax(np.array(d))]
            
            conf = (conf - np.amin(conf)) / (np.amax(conf) - np.amin(conf))
            
            ax.imshow(delta, extent = [x_decision[0], x_decision[-1], 
                                       y_decision[0], y_decision[-1]], 
                      cmap = self.c_cmap, zorder = -2, origin = "lower", 
                      alpha = conf, interpolation = "hamming")            

    
    def biplot(self, 
               x=1, 
               y=2, 
               loading_scale=1, 
               FOI=None):
        """
        A powerful plot technique that shows the scores plot as well as the
        loading plot of the features of interest (FOIs) for the given
        components. The loading plot is scaled differently than the scores
        plot.

        Parameters
        ----------
         x : int, optional
            Defines the displayed component in x direction. The default is 1.
        y : int, optional
            Defines the displayed component in y direction. The default is 2.
        loading_scale : float, optional
            Scaling factor for the loading scale. The default is 1 and scales
            the loading axes such that the absolute loading maximum is as big 
            as the absolute score maximum.
        FOI : list, optional
            Mask that can highlight some relevant FOInts. The default is 
            'None'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        fig = plt.figure("Biplot")
        fig.clf()
        ax = fig.add_subplot(111)
        
        if not self.analysis._scores:
            self.analysis.get_scores()
        self.get_labels()
        
        x, y = x - 1, y - 1
        
        #plot scale
        max_score = np.amax(np.abs(self.analysis.scores[:,[x, 
                                                           y]]))
        max_loading = np.amax(np.abs(self.analysis.loadings[[x, 
                                                             y]]))
        max_value = max(max_score, max_loading)
        ax.set_xlim(-max_value * 1.1, max_value* 1.1)
        ax.set_ylim(-max_value * 1.1, max_value* 1.1)
        
        if self.component_type == "PC":
            self.scores_pca(ax, x, y)
        elif self.component_type == "LD":
            self.scores_lda(ax, x, y)  
            
        ax.axhline(0, color = "k", zorder = 0)
        ax.axvline(0, color = "k", zorder = 0)
        
        #scaling loadings on same scale as scores
        scale = (np.amax([np.abs(np.amin(self.analysis.scores)), 
                         np.amax(self.analysis.scores)]) 
                 / np.amax([np.abs(np.amin(self.analysis.loadings)), 
                            np.amax(self.analysis.loadings)]))
        scale = scale * loading_scale
        
        def scaling_down(x, scale = scale):
            return x * scale
        
        def scaling_up(x, scale = scale):
            return x / scale
        
        ax1 = ax.secondary_xaxis("top", functions=(scaling_up, scaling_down))
        ax2 = ax.secondary_yaxis("right", functions=(scaling_up, scaling_down))
        
        if FOI is None:
            FOI = self.analysis.features["name"]
        
        if not (isinstance(FOI, list) or isinstance(FOI, np.ndarray)):
            FOI = [FOI]
            
        for P in FOI:
            idx = np.argwhere(self.analysis.features["name"] == P)
            ax.annotate("", 
                        xy = (0, 0), 
                        xytext = (self.analysis.loadings[x, idx] * scale, 
                                  self.analysis.loadings[y, idx] * scale), 
                        va = "center", ha = "center", color = "k", 
                        arrowprops = dict(arrowstyle="<|-", shrinkA=0, 
                                          shrinkB=0, color = "k"))
            if self.analysis.loadings[x, idx] >= 0:
                halign = "left"
            else:
                halign = "right"
            if self.analysis.loadings[y, idx] >= 0:
                valign = "bottom"
            else:
                valign = "top"
            ax.text(self.analysis.loadings[x, idx] * scale, 
                    self.analysis.loadings[y, idx] * scale,
                    f"{P}", va = valign, ha = halign)
        ax1.set_xlabel(f"Loadings {self.components['name'][x]}")
        ax2.set_ylabel(f"Loadings {self.components['name'][y]}")
        
        ax.set_ylabel(self.labels["name"][y])
        ax.set_xlabel(self.labels["name"][x])
        
        ax.legend(loc = "upper right")
        ax.set_aspect("equal")
        ax.grid(True)
        fig.tight_layout()
    
    # def multiplot(self, visualization_type = "scores", n_components = None, label = None, FOInts_of_interest = None, n_features = None):
    #     if n_components == None:
    #         n_components = self.analysis.n_components
    #     figure_title = f"Multiplot of PCA {visualization_type}"
    #     fig, axs = plt.subplots(n_components - 1, n_components - 1)
    #     fig.canvas.set_window_title(figure_title)
    #     x, y = 1, 2
    #     while True:
    #         plt.sca(axs[x - 1, y - 2])
    #         if x >= y:
    #             fig.delaxes(axs[x - 1, y - 2])
    #         else:
    #             if visualization_type == "scores":
    #                 self.analysis.scores_plot(x, y, label = label, legend = False)
    #                 if y == n_components and x == 1:
    #                     plt.legend(bbox_to_anchor = (1.05, 1.05), loc = "upper right")
    #             elif visualization_type == "loadings":
    #                 self.analysis.loadings_plot(x, y, n_features = n_features)
    #         y += 1
    #         if y > n_components:
    #             y = 2
    #             x += 1
    #         if x >= n_components:
    #             break
    #     plt.tight_layout(0.5)