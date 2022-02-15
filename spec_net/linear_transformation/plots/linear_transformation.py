import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from pathlib import Path

from .base import Base
from .utils._visualization import confidence

# PLOT PROPERTIES:
# figure size:
full_width = 7.14
half_width = 3.5

class LinearTransformationVisualizer(Base):
  def __init__(self, container):
    """
    The base plot class for all implemented linear transformation algorithms
    (e.g. PCA and LDA).

    Parameters
    ----------
    container : class
      The chosen linear transformation class. Accepts 'PCA' and 'LDA'.

    Raises
    ------
    TypeError
      Error, if instantiated container class is not of kind 'PCA' or 'LDA'.

    Returns
    -------
    None.

    """
    super().__init__()
    self.container = container
    self.set_path(Path.cwd() / 'plots' / self.container.name /
                  self.container.timestamp)

    if "PCA" in repr(container):
      self.component_type = "PC"

    elif "LDA" in repr(container):
      self.component_type = "LD"

    else:
      raise TypeError("Plot function is not supported for other"
                      "container objects than 'PCA' or 'LDA'.")

    self.components = self._get_components()

  def _get_components(self):
    idx, components = [], []

    for i in range(self.container.params.build.n_components):
      idx.append(i)
      components.append(f"{self.component_type}$_{int(i+1)}$")

    components = {'idx': idx, 'name': components}
    return components

  def scree(self, type='bar', show_average=False):
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
    dark = self.c_cmap(16)
    bright1 = self.c_cmap(0)
    bright2 = self.c_cmap(2)

    self.container.get_eigs()

    fig = plt.figure("Scree Plot", figsize=(half_width, 2))
    fig.clf()

    ax = fig.add_subplot(111)
    ax.set_ylabel("variance explained [$\%$]")

    # gets oev and cev for the first n components:
    oev = self.container.data.oev[0, :self.container.params.build.n_components]
    cev = self.container.data.cev[0, :self.container.params.build.n_components]

    # data for cummulative explained variance:
    cev = np.insert(cev, 0, 0)

    # average variance and indices where average > oev:
    avg = np.sum(self.container.data.oev[0]
                     / len(self.container.data.oev[0]))
    idx = np.argwhere(oev < avg).flatten()

    ticklabels = self.components['name']
    ticks = np.array(self.components['idx'])

    if type == 'bar':
      ax.bar(ticks, oev, color=bright1)
      for i in ticks:
        ax.text(i, oev[i], np.round(oev[i], 1), ha='center', color=bright1)


    elif type == 'cummulative':
      ax.plot(np.arange(len(cev))-0.5, cev, color=dark,
              marker='.', label='cummulative')
      ax.set_xticks(ticks)
      ax.set_xticklabels(ticklabels)

    elif type == 'both':
      ax.bar(ticks, oev, color=bright1)
      ax.plot(np.arange(len(cev))-0.5, cev, color=dark,
              marker='.', label='cummulative')
      for i in ticks:
        ax.text(i, oev[i], np.round(oev[i], 1), ha='center', color=bright1)

    if show_average:
      ax.axhline(y=avg, color=dark, ls="-.",
                 label='average', zorder = 0)

      if type == 'bar' or type == 'both':
        ax.bar(ticks[idx], oev[idx], color=bright2)

      ax.legend()


    ax.set_xlabel('component')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_xlim(-0.5, len(oev)-0.5)

    ax.set_ylim(0, 105)

  def loadings_line(self, component=None):
    """
    The loadings line plot shows the correlation of the original features with
    the newly generated components (LDs or PCs). The importance of of each
    feature for the container can be extracted out of the line plot.

    Returns
    -------
    None.

    """

    self.container.get_loadings()

    fig = plt.figure("Loadings Lineplot", figsize=(full_width, 4))
    fig.clf()
    ax = fig.add_subplot(111)

    if component:
      idx = np.array(component, ndmin=1) - 1
    else:
      idx = self.components['idx']

    for i in idx:
      ax.plot(self.container.features, self.container.data.loadings[i],
              label=self.components['name'][i])

    # y-axis:
    ax.set_ylabel('loading value [-]')

    # x-axis:
    ax.set_xlabel('feature')
    ax.set_xlim(0, self.container.data.n_features-1)
    ax.set_xticks(self.container.features)
    ax.set_xticklabels(self.container.features, rotation=-45, ha="left")

    ax.legend(loc='upper center', ncol=len(idx))

  def contribution_bar(self, component=1, show_average=False):
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

    self.container.get_contribution()

    height = self.container.data.contribution[component] * 100
    idx = np.flip(np.argsort(height))
    height = height[idx]
    features = self.container.features[idx]
    ax.bar(features, height=height)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=-45, ha='left')

    if show_average:
      ax.axhline(np.mean(height), color='k', ls='-.', label='average',
                 zorder=0)
      ax.legend(loc='upper right')

    # y-axis:
    ax.set_ylim(0, height[0] * 1.05)
    ax.set_ylabel(f"conribution to {self.components['name'][component]} $[\%]$")

    # x-axis:
    ax.set_xlim(-0.5, len(height)-0.5)
    ax.set_xlabel('features')

  def contribution_bars(self, components=[1, 2], show_average=False):
    """
    The contribution bars plot is a visualization tool, that sorts the features
    according to their loading values for the given component and plots the
    features in descending order.
    It sums up all specified component contributions and sorts them according
    to their sum.

    Parameters
    ----------
    components : list, optional
      Defines the displayed components. The default is [1, 2].

    show_average : bool, optional
      If True, it shows the average contribution of all features. The default
      is 'False'.

    Returns
    -------
    None.

    """
    fig = plt.figure('Contribution bars', figsize=(half_width, half_width))
    fig.clf()
    ax = fig.add_subplot(111)

    components = np.array(components, ndmin=1) - 1
    width = 0.8 / len(components)

    self.container.get_contribution()

    heights = self.container.data.contribution[components]

    heights = heights / np.sum(heights) * 100
    height = np.sum(heights, axis=0)

    idx = np.flip(np.argsort(np.abs(height)))
    height = height[idx] / np.sum(height) * 100
    features = self.container.features[idx]
    x = np.arange(self.container.data.n_features)
    ax.bar(x, height=np.abs(height), color='k', alpha=0.25, edgecolor='k',
           label='sum')

    for i, h in enumerate(heights):
      offset = -1/2 * (0.8 + width) + (i + 1) / len(components)*0.8
      ax.bar(x + offset, height=heights[i, idx],
             label=self.components['name'][components[i]], width=width)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=-45, ha='left')

    if show_average:
      ax.axhline(np.mean(height), color='k', ls='-.', zorder=0)
      ax.legend(loc='upper right')

    ax.set_ylim(0, height[0] * 1.05)
    ax.set_ylabel("conribution to components [$\%$]")

    ax.set_xlim(-0.5, x[-1] - 0.5)
    ax.set_xlabel('features')

  def plot_features_of_interest(self, FOI, x, y):
    """
    Is looking for the feature of interest and plots it in a scatter plot in a
    lighter color.

    Parameters
    ----------
    FOI : str, int or list
      The feature of interest. Has to have the same formating as the features.
    x : int
      The component on the x-axis.
    y : int
      The component on the y-axis.

    Returns
    -------
    None.

    """

    if FOI is None:
      return

    FOI = np.array(FOI, ndmin=1)

    mask = np.argwhere(self.container.features == np.array(FOI))
    plt.scatter(self.container.data.loadings[x, mask],
                self.container.data.loadings[y, mask], color=self.c_cmap(2))

  def contribution_circle(self, component_x=1,component_y=2, FOI=None):
    """
    Similarily to the loadings lineplot, the contribution circle is a
    visualization tool to show how much each variable contributes to the
    corresponding component (PC or LD).
    It is only suitable for standardized datasets, since it uses an unit circle
    for comparison. For unstandardized data use '.loadings_plot()'.

    Parameters
    ----------
    component_x : int, optional
      Defines the displayed component in x direction. The default is 1.
    component_y : int, optional
      Defines the displayed component in y direction. The default is 2.
    FOI : list, optional
      Mask that can highlight some relevant points. The default is 'None'.

    Returns
    -------
    None.

    """

    fig = plt.figure('Contribution circle', figsize=(half_width, half_width))
    ax = fig.add_subplot(111)

    self.container.get_loadings()

    x, y = component_x - 1, component_y - 1

    ax.scatter(self.container.data.loadings[x],
               self.container.data.loadings[y])

    limits = 1.05

    self.plot_features_of_interest(FOI=FOI, x=x, y=y)

    # plot unit circle:
    circle = plt.Circle((0, 0), 1, color='k', ls='-.', fill=False)
    ax.add_patch(circle)

    # y-axis:
    ax.set_ylim(-limits, limits)
    ax.set_ylabel(self.components['name'][y])

    # x-axis:
    ax.set_xlim(-limits, limits)
    ax.set_xlabel(self.components['name'][x])

    ax.set_aspect('equal')

  def contribution_heatmap(self, idx=None, show_sum=False, weighted=True):
    """
    The contribution heatmap shows all the loading values of the p components
    (LD or PC) and visualizes their contribution with circles of different
    sizes. The bigger the circle, the more does this feature contribute to the
    corresponding component.

    Returns
    -------
    None.

    """
    fig = plt.figure("Contribution Heatmap", figsize=(half_width, 5))
    fig.clf()
    ax = fig.add_subplot(111)

    self.container.get_contribution()

    #mapping circles onto heatmap
    if not idx:
      n = self.container.params.build.n_components
      idx = np.arange(n)

    else:
      if isinstance(idx, int):
          idx = [idx]

      idx = np.array(idx) - 1
      n = len(idx)

    contribution = self.container.data.contribution[idx]

    if show_sum:
      weights = np.ones([n, 1]) / n
      if weighted:
        evals = self.container.data.evals[:, idx]
        weights = (evals/np.sum(evals)).T
      con_sum = np.sum(contribution*weights, axis=0, keepdims=True)
      n += 1 # for additional axis
      contribution = np.concatenate([contribution, con_sum],
                                    axis = 0)
    con_max = np.amax(contribution)
    con_min = np.amin(contribution)

    for i in range(self.container.data.n_features):
      for j in range(n):
        con = contribution[j, i]
        circle = plt.Circle((j, i),
                            radius=con/(2*con_max),
                            color=self.im_cmap(con/con_max))
        ax.add_patch(circle)

    #setup plot axes
    ax.set_xticks(np.arange(n))
    xticks = list(np.array(self.components['name'])[idx])
    if show_sum:
      if weighted:
        xticks.append('$\Sigma_w$')
      else:
        xticks.append('$\Sigma$')
    ax.set_xticklabels(xticks)
    ax.set_yticks(np.arange(self.container.data.n_features))
    ax.set_yticklabels(self.container.features)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, len(self.container.features) - 0.5)
    ax.set_aspect("equal")
    ax.grid(False)

    #rastering
    for i in range(len(self.container.features)):
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
    cbar.ax.set_ylabel("contribution [$\%$]")

  def loadings(self, component_x=1, component_y=2, FOI=None):
    """
    Similarily to the loadings lineplot, the loadings plot is a
    visualization tool to show how much each variable contributes to the
    corresponding component (PC or LD).

    Parameters
    ----------
    component_x : int, optional
      Defines the displayed component in x direction. The default is 1.
    component_y : int, optional
      Defines the displayed component in y direction. The default is 2.
    FOI : list, optional
      Mask that can highlight some relevant points. The default is 'None'.

    Returns
    -------
    None.

    """
    self.container.get_loadings()

    fig = plt.figure("Loadings Plot", figsize=(half_width, half_width))
    fig.clf()
    ax = fig.add_subplot(111)

    x, y = component_x - 1, component_y - 1

    ax.scatter(self.container.data.loadings[x],
               self.container.data.loadings[y])

    self.plot_features_of_interest(FOI=FOI, x=x, y=y)

    ax.set_ylabel(self.components["name"][y])
    ax.set_xlabel(self.components["name"][x])

    ax.set_aspect('equal', 'datalim')

    return ax

  # implement 3rd component for 3d plot:
  def scores(self, component_x=1, component_y=2,
             projection=False,
             decision_boundaries=False,
             confidence_interval=3):
    """
    The scores plot shows the samples in the new latent feature space defined
    by the given components.
    The scores plot is the probably most important plot in the analyses based
    on linear transformations, since it directly provides information about the
    ability to cluster the samples.

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
    self.container.get_scores()

    fig = plt.figure("Scores Plot", figsize=(half_width, half_width))
    fig.clf()

    if projection:
      gs = gridspec.GridSpec(2, 2, figure=fig,
                             width_ratios = (5,1),
                             height_ratios = (1,5))

      ax = fig.add_subplot(gs[1,0])
      ax.set_aspect('equal')

      ax1 = fig.add_subplot(gs[1,1], sharey=ax)
      ax1.tick_params(labelleft=False, labelbottom=False)
      # ax1.axis('off')

      ax2 = fig.add_subplot(gs[0,0], sharex=ax)
      ax2.tick_params(labelleft=False, labelbottom=False)
      # ax2.axis('off')

      ax3 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax2)
      # ax3.axis('off')

    else:
      ax = fig.add_subplot(111)

    if self.component_type == 'PC':
      if projection:
        self.scores_pca(ax, component_x, component_y,
                        projection=projection,
                        projection_axes=[ax1, ax2],
                        confidence_interval=confidence_interval)
        ax1.set_ylim(ax.get_ylim())
        ax2.set_xlim(ax.get_xlim())
      else:
        self.scores_pca(ax, component_x, component_y,
                        confidence_interval=confidence_interval)

    elif self.component_type == 'LD':
      if projection:
        self.scores_lda(ax, component_x, component_y,
                        projection=projection,
                        projection_axes=[ax1, ax2],
                        decision_boundaries=decision_boundaries,
                        confidence_interval=confidence_interval)
      else:
        self.scores_lda(ax, component_x, component_y,
                        projection=projection,
                        decision_boundaries=decision_boundaries,
                        confidence_interval=confidence_interval)

    ax.set_xlabel(self.components['name'][component_x - 1])
    ax.set_ylabel(self.components['name'][component_y - 1])

  def scores_pca(self, ax, component_x=1, component_y=2,
                  projection=False, projection_axes=[None, None],
                  confidence_interval=3):
      """
      The scores plot shows the samples in the new latent feature space
      defined by the given components.
      The scores plot is the probably most important plot in the analyses
      based on linear transformations, since it directly provides information
      about the ability to cluster the samples.

      Parameters
      ----------
      ax : axis
          Defines the current axis of the inserted plot.
      component_x : int, optional
          Defines the displayed component in x direction. The default is 1.
      component_y : int, optional
          Defines the displayed component in y direction. The default is 2.

      Returns
      -------
      ax : axis
        The axis with the scatter plot.

      """
      x, y = component_x - 1, component_y - 1
      limit = np.amax(np.abs(self.container.data.scores[:, [x, y]])) * 1.0

      if self.container.y is not None:
        for i, c in enumerate(self.container.data.classes):
          mask = np.argwhere(self.container.y == c).flatten()
          scores = self.container.data.scores[mask][:,np.array([x, y])]

          color = self.c_cmap(i * 4)
          cov = np.cov(scores.T)
          mu = np.average(scores, axis=0)

          if confidence_interval:
            confidence(ax, cov, mu, s=confidence_interval)

          ax.scatter(scores[:, 0], scores[:, 1], alpha=0.7, color=color,
                     label=c)

          ax.scatter(mu[0], mu[1], marker="X", color=color, edgecolors="k")

          # projection of scatter plot onto x- and y-axis
          if projection:
            X = np.linspace(-limit, limit, 100)
            exp_0 = np.exp(-(X - mu[0])**2 / (2 * cov[0, 0]))
            exp_1 = np.exp(-(X - mu[1])**2 / (2 * cov[1, 1]))
            scale_0 = 1 / np.sqrt(2 * np.pi * cov[0, 0])
            scale_1 = 1 / np.sqrt(2 * np.pi * cov[1, 1])
            projection_0 = scale_0 * exp_0
            projection_1 = scale_1 * exp_1
            projection_axes[1].plot(X, projection_0)
            projection_axes[0].plot(projection_1, X)

      else:
        ax.scatter(self.container.data.scores[:, x],
                   self.container.data.scores[:, y],
                   alpha=0.7)

      ax.set_ylim(-limit, limit)
      ax.set_xlim(-limit, limit)

      return ax

  def scores_lda(self, ax, x = 1, y = 2,
                  projection = False,
                  projection_axes = [None, None],
                  decision_boundaries = False,
                  confidence_interval = 3):
    """
    The scores plot shows the samples in the new latent feature space defined
    by the given components.
    The scores plot is the probably most important plot in the analyses based
    on linear transformations, since it directly provides information about the
    ability to cluster the samples.

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
    limit = np.amax(np.abs(self.container.scores[:, [x, y]]))
    for i in range(self.container.n_classes):
      scores = self.container.log[f"{self.container.classes[0][i]}"]["scores"].T[[x, y]]
      cov = self.container.data.covariance_matrix(scores)
      mu = (self.container.loadings[[x, y]] @ self.container.log[f"{self.container.classes[0][i]}"]["mu"]).T.squeeze()

      if confidence_interval:
        confidence(ax, cov, mu, s = confidence_interval)

      ax.scatter(scores[0], scores[1], alpha = 0.7, edgecolors = "k",
                 label = f"{self.container.classes[0][i]}")
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
          d = self.container.decision_rule(np.array([[X, Y]]).T,
                                          components = [x, y],
                                          transformed = True)
          conf[j, i] = np.diff(d[np.argsort(d)[-2:]])
          delta[j, i] = self.container.classes[1, np.argmax(np.array(d))]

      conf = (conf - np.amin(conf)) / (np.amax(conf) - np.amin(conf))

      ax.imshow(delta, extent = [x_decision[0], x_decision[-1],
                                  y_decision[0], y_decision[-1]],
                cmap = self.c_cmap, zorder = -2, origin = "lower",
                alpha = conf, interpolation = "hamming")


  def biplot(self, component_x=1, component_y=2, loading_scale=1, FOI=None):
    """
    A powerful plot technique that shows the scores plot as well as the
    loading plot of the features of interest (FOIs) for the given
    components. The loading plot is scaled differently than the scores
    plot.

    Parameters
    ----------
    component_x : int, optional
      Defines the displayed component in x direction. The default is 1.
    component_y : int, optional
      Defines the displayed component in y direction. The default is 2.
    loading_scale : float, optional
      Scaling factor for the loading scale. The default is 1 and scales the
      loading axes such that the absolute loading maximum is as big as the
      absolute score maximum.
    FOI : list, optional
      Mask that can highlight some relevant FOInts. The default is 'None'.

    Returns
    -------
    None.

    """
    self.container.get_scores()

    fig = plt.figure('Biplot', figsize=(half_width, half_width))
    fig.clf()
    ax = fig.add_subplot(111)

    x, y = component_x - 1, component_y - 1

    #plot scale

    if self.component_type == "PC":
      ax = self.scores_pca(ax, x, y)
    elif self.component_type == "LD":
      ax = self.scores_lda(ax, x, y)

    #scaling loadings on same scale as scores
    scale = (np.amax([np.abs(np.amin(self.container.data.scores)),
                      np.amax(self.container.data.scores)])
              / np.amax([np.abs(np.amin(self.container.data.loadings)),
                        np.amax(self.container.data.loadings)]))
    scale = scale * loading_scale

    def scaling_down(x, scale = scale):
      return x * scale

    def scaling_up(x, scale = scale):
      return x / scale

    ax1 = ax.secondary_xaxis("top", functions=(scaling_up, scaling_down))
    ax2 = ax.secondary_yaxis("right", functions=(scaling_up, scaling_down))

    if FOI:
      FOI = np.array(FOI, ndmin=1)

      for P in FOI:
        idx = np.argwhere(self.container.features == np.array(P))
        ax.annotate(P, xy = (0, 0),
                    xytext = (self.container.data.loadings[x, idx] * scale,
                              self.container.data.loadings[y, idx] * scale),
                    va = "center", ha = "center", color = "k",
                    arrowprops = dict(arrowstyle="<|-", shrinkA=0,
                                      shrinkB=0, color = "k"))

    # x-axis:
    ax.set_xlabel(f"score values {self.components['name'][x]}")
    ax1.set_xlabel(f"loading values {self.components['name'][x]}")

    # y-axis:
    ax.set_ylabel(f"score values {self.components['name'][y]}")
    ax2.set_ylabel(f"loading values {self.components['name'][y]}")

    ax.set_aspect('equal', 'datalim')

    if self.component_type == "LD":
      ax.legend(loc='upper right')


  # def multiplot(self, visualization_type = "scores", n_components = None, label = None, FOInts_of_interest = None, n_features = None):
  #     if n_components == None:
  #         n_components = self.container.n_components
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
  #                 self.container.scores_plot(x, y, label = label, legend = False)
  #                 if y == n_components and x == 1:
  #                     plt.legend(bbox_to_anchor = (1.05, 1.05), loc = "upper right")
  #             elif visualization_type == "loadings":
  #                 self.container.loadings_plot(x, y, n_features = n_features)
  #         y += 1
  #         if y > n_components:
  #             y = 2
  #             x += 1
  #         if x >= n_components:
  #             break
  #     plt.tight_layout(0.5)