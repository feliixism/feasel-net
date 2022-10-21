"""
feasel.plot.linear_transformation
=================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from matplotlib import colors
from pathlib import Path

from .base import Base
from .utils._visualization import confidence

# PLOT PROPERTIES:
# figure size:
full_width = 7.14
half_width = 3.5

class LinearTransformationVisualizer(Base):
  """
  The base plot class for all implemented linear transformation algorithms
  (e.g. PCA and LDA).

  Parameters
  ----------
  container : class
    The chosen linear transformation class with either
    :class:`feasel.linear.PCA` or :class:`feasel.linear.LDA`.

  data : class
    The data container stored in the model container specified by `container`.

  Raises
  ------
  TypeError
    Error, if instantiated container class is not of correct class type.

  """
  def __init__(self, container):
    super().__init__()
    self.container = container
    self.data = self.container.data

    self.set_path(Path.cwd() / 'plots' / self.container.name /
                  self.container.timestamp)

    if 'PCA' in repr(container):
      self.component_type = 'PC'

    elif 'LDA' in repr(container):
      self.component_type = 'LD'

    else:
      raise TypeError('Plot function is not supported for other'
                      'container objects than `PCA` or `LDA`.')

    self.components = self._get_components()

  def _get_components(self):
    """
    Provides a `dict` of indices and names of the first :math:`q` components.

    Returns
    -------
    components : dict
      A dictionary with indices and names of the components.

    """
    idx, components = [], []

    for i in range(self.container.params.build.n_components):
      idx.append(i)
      components.append(f'{self.component_type}$_{int(i+1)}$')

    components = {'idx': idx,
                  'name': components}

    return components

  def scree(self,
            type='bar',
            show_average=False):
    """
    The scree plot shows the individually contributed variance explained for
    each Principal Component (PC) or Linear Discriminant (LD) in sorted order.

    All :math:`n` contributions sum up to :math:`1`, whereas the scree plot
    only shows the variances for as much PCs or LDs as originally initialized
    in the class.

    Parameters
    ----------
    type : str, optional
      Determines the visualization type for the scree plot. Valid options are:
      `bar`, `line` or `both`. The default is `bar`.
    show_average : boolean, optional
      Displays average of all n contributions. The default is `False`.
    """
    dark = 'k'
    bright1 = self.c_cmap(0)
    bright2 = self.c_cmap(2)

    self.container.get_eigs()

    title = f'{self.component_type} Scree Plot'

    fig = plt.figure(title, figsize=(half_width, 2))
    fig.clf()

    ax = fig.add_subplot(111)
    ax.set_ylabel("explained variance [$\%$]")

    # gets oev and cev for the first n components:
    oev = self.data.oev[0, :self.container.params.build.n_components]
    cev = self.data.cev[0, :self.container.params.build.n_components]

    # data for cummulative explained variance:
    cev = np.insert(cev, 0, 0)

    # average variance and indices where average > oev:
    avg = np.sum(self.data.oev[0]
                     / len(self.data.oev[0]))
    idx1 = np.argwhere(oev < avg).flatten()
    idx2 = np.argwhere(oev > avg).flatten()

    ticklabels = self.components['name']
    ticks = np.array(self.components['idx'])

    if type == 'bar':
      ax.bar(ticks, oev, color=bright1)
      for i in ticks:
        ax.text(i, oev[i] + 1, np.round(oev[i], 1), ha='center', color=bright1)


    elif type == 'cummulative':
      ax.plot(np.arange(len(cev))-0.5, cev, color=dark,
              marker='.', label='cummulative')
      ax.set_xticks(ticks)
      ax.set_xticklabels(ticklabels)

    elif type == 'both':
      ax.bar(ticks, oev, color=bright1)
      ax.plot(np.arange(len(cev))-0.5, cev, color=dark,
              marker='.', label='cummulative')

    if show_average:
      ax.axhline(y=avg, color=self.c_cmap(16), ls="-.",
                 label='average', zorder = 0)

      if type == 'bar' or type == 'both':
        ax.bar(ticks[idx1], oev[idx1], color=bright2)
        for i in idx1:
          ax.text(ticks[i], oev[i] + 1, np.round(oev[i], 1), ha='center',
                  color=dark)
        for i in idx2:
          ax.text(ticks[i], oev[i] + 1, np.round(oev[i], 1), ha='center',
                  color=dark)
      else:
        for i in ticks:
          ax.text(i, oev[i] + 1, np.round(oev[i], 1), ha='center',
                  color=dark)

      ax.legend()

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_xlim(-0.5, len(oev)-0.5)

    ax.set_ylim(0, 105)

  def loadings(self, component, type='line', FOI=None):
    if type == 'bar':
      self._loadings_bar(component)
    elif type == 'line':
      self._loadings_line(component)
    elif type == 'scatter':
      component_x = component[0]
      component_y = component[1]
      self._loadings_scatter(component_x, component_y, FOI)

  def _loadings_line(self, component=None):
    """
    The loadings line plot shows the correlation of the original features with
    the newly generated components (LDs or PCs). The importance of of each
    feature for the container can be extracted out of the line plot.

    component : int, optional
      The component that shall be plotted. If 'None', all components are
      plotted. If it is a list of integers, it will plot all lines in one plot.
      The default is 'None'.

    Returns
    -------
    None.

    """
    title = f'{self.component_type} Loadings Plot: Components {component}'
    self.container.get_loadings()

    fig = plt.figure(title, figsize=(half_width, 2))
    fig.clf()
    ax = fig.add_subplot(111)

    if component:
      idx = np.array(component, ndmin=1) - 1
    else:
      idx = self.components['idx']

    for i in idx:
      ax.plot(self.data.features, self.data.loadings[i],
              label=self.components['name'][i])

    # y-axis:
    ax.set_ylabel('loading [-]')

    # x-axis:
    ax.set_xlabel('feature')
    ax.set_xlim(self.data.features[0],
                self.data.features[-1])

    if self.data.features.dtype.type == np.str_:
      ax.set_xticks(self.data.features)
      ax.set_xticklabels(self.data.features,
                              rotation=45, ha="right")

    ax.legend(loc='upper center', ncol=len(idx))

  def _loadings_bar(self, component=None):
    """
    The loadings line plot shows the correlation of the original features with
    the newly generated components (LDs or PCs). The importance of of each
    feature for the container can be extracted out of the line plot.

    component : int, optional
      The component that shall be plotted. If 'None', all components are
      plotted. If it is a list of integers, it will plot all bar plots in
      multiple subplots. The default is 'None'.

    Returns
    -------
    None.

    """
    title = f'{self.component_type} Loadings Plot: Components {component}'
    self.container.get_loadings()

    fig = plt.figure(title, figsize=(half_width, 2))

    if component:
      components = np.array(component, ndmin=1) - 1
    else:
      components = self.components['components']

    n = len(components)

    fig = plt.figure(title, figsize=(half_width, n*2))
    fig.clf()
    fig, axs = plt.subplots(n, 1, sharex=True, sharey=True,
                            num=title, figsize=(full_width, n*3))

    axs = np.array(axs, ndmin=1)
    for i, c in enumerate(components):
      axs[i].bar(self.data.features, self.data.loadings[c])

      # y-axis:
      axs[i].set_ylabel(f"{self.components['name'][c]} "
                        f"[{np.round(self.data.oev[0][c], 1)}\,\%]")

    # x-axis:
    # axs[-1].set_xlabel('feature')
    axs[-1].set_xlim(-0.5, self.data.n_features-0.5)

    if self.data.features.dtype.type == np.str_:
      axs[-1].set_xticks(self.data.features)
      axs[-1].set_xticklabels(self.data.features,
                              rotation=45, ha="right")

  def _loadings_scatter(self, component_x=1, component_y=2, FOI=None):
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
    title = (f'{self.component_type} Loadings Plot: ',
             'Components {component_x} and {component_y}')
    self.container.get_loadings()

    fig = plt.figure(title, figsize=(half_width, half_width))
    fig.clf()
    ax = fig.add_subplot(111)

    x, y = component_x - 1, component_y - 1

    limit = np.amax(np.abs(self.data.loadings[[x, y]])) * 1.05
    ax.set_ylim(-limit, limit)
    ax.set_xlim(-limit, limit)

    ax.scatter(self.data.loadings[x],
               self.data.loadings[y])

    self.plot_features_of_interest(FOI=FOI, x=x, y=y)

    ax.set_ylabel(self.components["name"][y])
    ax.set_xlabel(self.components["name"][x])

  def contribution_bar(self, component=1, show_average=False):
    """
    The contribution plot is a visualization tool, that sorts the features
    according to their loading values for the given component and plots
    the features in descending order.

    Parameters
    ----------
    component : int, optional
        Defines the displayed component. The default is 1.
    show_average : bool, optional
      If True, it shows the average contribution of all features. The default
      is 'False'.

    Returns
    -------
    None.

    """
    title = f'{self.component_type} Contribution Bar: Component {component}'
    fig = plt.figure(title, figsize=(half_width, 2))
    fig.clf()
    ax = fig.add_subplot(111)

    component = component - 1

    self.container.get_contribution()

    height = self.data.contribution[component] * 100
    idx = np.flip(np.argsort(height))
    height = height[idx]
    features = self.data.features[idx]
    ax.bar(np.arange(len(features)), height=height)
    ax.set_xticks(range(len(features)))

    if features.dtype.type == np.str_:
      ax.set_xticklabels(features, rotation=45, ha='right')
    else:
      ax.set_xticklabels(features, ha='center')

    if show_average:
      ax.axhline(np.mean(height),
                 color='k',
                 ls='-.',
                 label='average',
                 zorder=0)
      ax.legend(loc='upper right')

    # y-axis:
    ax.set_ylim(0, height[0] * 1.05)
    ax.set_ylabel(f"conribution to {self.components['name'][component]} "
                  "$[\%]$")
    ax.set_xlabel("feature")

    # x-axis:
    ax.set_xlim(-0.5, len(height)-0.5)

  def importance_bar(self, components=None, show_average=False, crop=None):
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
    title = (f"{self.component_type} Contribution Bars: Components: "
             f"{components}")
    dark = self.c_cmap(18)
    fig = plt.figure(title, figsize=(half_width, 2))
    fig.clf()
    ax = fig.add_subplot(111)
    if not components:
      components = np.arange(self.q)
    else:
      components = np.array(components, ndmin=1) - 1
    width = 0.8 / len(components)

    self.container.get_importance()

    heights = self.data.importance[components]
    height_sum = np.sum(heights, axis=0)

    idx = np.flip(np.argsort(np.abs(height_sum)))

    if crop:
      idx = idx[:crop]

    height = height_sum[idx] / np.sum(height_sum) * 100
    heights = heights / np.sum(height_sum) * 100
    features = self.data.features[idx]
    x = np.arange(len(idx))
    ax.bar(x, height=np.abs(height), color=dark, alpha=1, edgecolor=dark,
           label='sum')

    for i, h in enumerate(heights):
      offset = -1/2 * (0.8 + width) + (i + 1) / len(components)*0.8
      ax.bar(x + offset, height=heights[i, idx],
             label=self.components['name'][components[i]], width=width,
             color=self.c_cmap(i))

    ax.set_xticks(x)

    if features.dtype.type == np.str_:
      ax.set_xticklabels(features, rotation=45, ha='right')
    else:
      ax.set_xticklabels(features, ha='center')

    if show_average:
      ax.axhline(np.mean(height), color='k', ls='-.', zorder=0)

    ax.legend(loc='upper right')

    ax.set_ylabel("feature importance")
    ax.set_xlabel("features")
    ax.set_ylim(np.amin(np.abs(height))*0.95, np.amax(np.abs(height))*1.05)

    ax.set_xlim(-0.5, x[-1] + 0.5)

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

    if (FOI is None) or (FOI not in self.data.features):
      return

    FOI = np.array(FOI, ndmin=1)

    mask = np.argwhere(self.data.features == np.array(FOI))
    plt.scatter(self.data.loadings[x, mask],
                self.data.loadings[y, mask], color=self.c_cmap(2))

  def contribution_circle(self,
                          component_x=1,
                          component_y=2,
                          FOI=None):
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
    title = (f'{self.component_type} Contribution circle: Component '
             f'{component_x} and {component_y}')
    fig = plt.figure(title, figsize=(half_width, half_width))
    ax = fig.add_subplot(111)

    self.container.get_contribution()

    x, y = component_x-1, component_y-1

    ax.scatter(self.data.contribution[x],
               self.data.contribution[y])

    limits = (ax.get_xlim(), ax.get_ylim())
    self.plot_features_of_interest(FOI=FOI, x=x, y=y)

    # plot unit circle:
    circle = plt.Circle((0, 0), 1, color='k', ls='-.', fill=False)
    ax.add_patch(circle)

    # y-axis:
    ax.set_ylabel(self.components['name'][y])
    ax.set_xlim(limits[0])
    # x-axis:
    ax.set_xlabel(self.components['name'][x])
    ax.set_ylim(limits[1])

  def contribution_heatmap(self,
                           component=None,
                           show_sum=False,
                           weighted=True):
    """
    The contribution heatmap shows all the loading values of the p components
    (LD or PC) and visualizes their contribution with circles of different
    sizes. The bigger the circle, the more does this feature contribute to the
    corresponding component.

    component : int, optional
      Defines the components to plot. If 'None', it will plot all components.
      The default is 'None'.
    show_sum : bool, optional
      Defines whether to plot the sum of all contributions. The default is
      'False'.
    weighted : bool, optional
      Defines whether the sum is weighted according their loadings. The default
      is 'True'.

    Returns
    -------
    None.

    """
    title = (f"{self.component_type} Contribution Heatmap: Component ",
             f"{component}")
    fig = plt.figure(title, figsize=(half_width, 5))
    fig.clf()
    ax = fig.add_subplot(111)

    self.container.get_contribution()

    #mapping circles onto heatmap
    if not component:
      n = self.container.params.build.n_components
      component = np.arange(n)

    else:
      if isinstance(component, int):
          component = [component]

      component = np.array(component) - 1
      n = len(component)

    contribution = self.data.contribution[component]

    # summation of all contributions:
    if show_sum:
      weights = np.ones([n, 1]) / n

      if weighted:
        evals = self.data.evals[:, component]
        weights = (evals/np.sum(evals)).T

      con_sum = np.sum(contribution*weights, axis=0, keepdims=True)
      n += 1 # for additional axis
      contribution = np.concatenate([contribution, con_sum],
                                    axis = 0)

    con_max = np.amax(contribution)

    # plot of contributions:
    for i in range(self.data.n_features):
      for j in range(n):
        con = contribution[j, i]
        circle = plt.Circle((j, i),
                            radius=con/(2*con_max),
                            color=self.im_cmap(con/con_max))
        ax.add_patch(circle)

    # x-axis:
    ax.set_xticks(np.arange(n))
    xticks = list(np.array(self.components['name'])[component])

    if show_sum:
      if weighted:
        xticks.append('$\Sigma_w$')

      else:
        xticks.append('$\Sigma$')

    ax.set_xticklabels(xticks)
    ax.set_xlim(-0.5, n - 0.5)

    # y-axis:
    ax.set_yticks(np.arange(self.data.n_features))
    ax.set_yticklabels(self.data.features)
    ax.set_ylim(-0.5, len(self.data.features) - 0.5)

    ax.set_aspect("equal")

    ax.grid(False)

    #rastering
    for i in range(len(self.data.features)):
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

  # implement 3rd component for 3d plot:
  def scores(self, component_x=1, component_y=2, X=None, y=None,
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
    component_x : int, optional
      Defines the displayed component in x direction. The default is 1.
    component_y : int, optional
      Defines the displayed component in y direction. The default is 2.
    projection : bool, optional
      Defines whether a projection of the scores onto the x- and y-axis is
      plotted. It requires the definition of the axes in the list specified
      in 'projection_axes'. The default is 'False'.
    decision_boundaries : bool, optional
      Defines whether the decision boundaries for LDAs is plotted or not. The
      default is 'False'.
    confidence_interval : int, optional
      Sets the value for a confidence interval around the center of center of
      data. If 'None', it will not plot the interval. The default is 'None'.

    Returns
    -------
    None.

    """
    if X is not None:
      X, target = X, y
    else:
      X, target = self.data.X_train, self.data.y_train

    scores = self.container.get_scores(X)

    title = (f'{self.component_type} Scores Plot: '
             f'Component {component_x} and {component_y}')

    fig = plt.figure(title, figsize=(half_width, half_width))
    fig.clf()

    if projection:
      gs = gridspec.GridSpec(2, 2, figure=fig,
                             width_ratios=(5,1),
                             height_ratios=(1,5))

      ax = fig.add_subplot(gs[1,0])

      ax2 = fig.add_subplot(gs[0,0], sharex=ax)
      ax2.tick_params(labelleft=False, labelbottom=False)
      ax2.axis('off')

      ax1 = fig.add_subplot(gs[1,1], sharey=ax)
      ax1.tick_params(labelleft=False, labelbottom=False)
      ax1.axis('off')

      ax3 = fig.add_subplot(gs[0,1])
      ax3.tick_params(labelleft=False, labelbottom=False)
      ax3.axis('off')
      projection_axes=[ax1, ax2]

    else:
      ax = fig.add_subplot(111)
      projection_axes=None

    if self.component_type == 'PC':
      self.scores_pca(ax, scores, target, component_x, component_y,
                      projection_axes=projection_axes,
                      confidence_interval=confidence_interval)

    elif self.component_type == 'LD':
      self.scores_lda(ax, scores, target, component_x, component_y,
                      projection_axes=projection_axes,
                      decision_boundaries=decision_boundaries,
                      confidence_interval=confidence_interval)

    idx_x, idx_y = component_x-1, component_y-1
    ax.set_xlabel(f"{self.components['name'][idx_x]} "
                  f" [{np.round(self.data.oev[0][idx_x], 1)}\,\%]")
    ax.set_ylabel(f"{self.components['name'][idx_y]} "
                  f" [{np.round(self.data.oev[0][idx_y], 1)}\,\%]")

  def scores_pca(self, ax, scores, target,
                 component_x=1,
                 component_y=2,
                 projection_axes=None,
                 confidence_interval=None):
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
      projection_axes : bool, optional
        Defines whether and where a projection of the scores onto the x- and
        y-axis is plotted. If the projection is to be plotted, a list with two
        `matplotlib.pyplot.axes` objects are needed. The default is 'None'.
      confidence_interval : int, optional
        Sets the value for a confidence interval around the center of center of
        data. If 'None', it will not plot the interval. The default is 'None'.

      Returns
      -------
      None.

      """
      if isinstance(scores, type(None)):
        scores = self.data.scores

      x, y = component_x-1, component_y-1
      scores = scores[:, [x, y]]

      if not isinstance(target, type(None)):
        for i, c in enumerate(self.data.classes):
          mask = target == c
          S = scores[mask]

          color = self.c_cmap(i * 4)
          cov = np.cov(S.T)
          mu = np.average(S, axis=0)

          if confidence_interval:
            confidence(ax, cov, mu, s=confidence_interval)

          ax.scatter(S[:, 0], S[:, 1],
                     alpha=0.7, color=color, label=c)

          ax.scatter(mu[0], mu[1], marker='X', color=color, edgecolors="k")

        ax.legend(loc='upper right')

      else:
        ax.scatter(scores[:, x], scores[:, y], alpha=0.7)

      # projection of scatter plot onto x- and y-axis:
      if projection_axes:
        if not isinstance(target, type(None)):
          for i, c in enumerate(self.data.classes):
            mask = target == c
            S = scores[mask]
            self._projection(S, target, ax, projection_axes, i)
        else:
          self._projection(scores, target, ax, projection_axes, i)

  def scores_lda(self, ax, scores, target, component_x=1, component_y=2,
                 projection=False,
                 projection_axes=[None, None],
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
    if isinstance(scores, type(None)):
      scores = self.data.scores

    x, y = component_x-1, component_y-1
    colors = []

    for i, c in enumerate(self.data.classes):
      mask = target == c
      S = scores[mask][:, np.array([x, y])]

      color = self.c_cmap(i * 4)
      cov = np.cov(S.T)
      mu = np.average(S, axis=0)

      if confidence_interval:
        confidence(ax, cov, mu, s=confidence_interval)

      ax.scatter(S[:, 0], S[:, 1],
                 alpha=0.7, color=color, label=c)

      ax.scatter(mu[0], mu[1], marker='X', color=color, edgecolors="k")
      colors.append(color)

      # projection of scatter plot onto x- and y-axis:
    if projection_axes:
      for i, c in enumerate(self.data.classes):
        mask = target == c
        S = scores[mask]
        self._projection(S, target, ax, projection_axes, i)

    # get decision boundary and confidence levels
    if decision_boundaries:
      cmap = LinearSegmentedColormap.from_list('im_lda',
                                               colors,
                                               N=len(colors))
      cmap = cmap.reversed()
      sampling = 100
      x_lims = ax.get_xlim()
      y_lims = ax.get_ylim()
      X = np.linspace(x_lims[0], x_lims[1], sampling)
      Y = np.linspace(y_lims[0], y_lims[1], sampling)

      conf = np.empty([sampling, sampling])
      delta = np.empty([sampling, sampling])

      for i, x_ in enumerate(X):
        for j, y_ in enumerate(Y):
          d = self.container.decision_rule(score=np.array([x_, y_], ndmin=2),
                                           components=[x, y])

          conf[i, j] = np.diff(d[:, np.argsort(d)[0,-2:]])
          delta[i, j] = np.argmax(np.array(d))

      conf = (conf - np.amin(conf)) / (np.amax(conf) - np.amin(conf))

      ax.imshow(delta.T, extent=[X[-1], X[0], Y[0], Y[-1]],
                cmap=cmap, zorder=-2, origin='lower',
                alpha=conf.T, interpolation='hamming',
                aspect=(np.abs(x_lims[0]-x_lims[-1])
                        /np.abs(y_lims[0]-y_lims[-1])))

      ax.set_xlim(x_lims)
      ax.set_ylim(y_lims)

    ax.legend(loc='upper right')

  def _projection(self, scores, target, ax, projection_axes, i):
    X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
    Y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 1000)
    projection_axes[1].set_xlim(ax.get_xlim())
    projection_axes[0].set_ylim(ax.get_ylim())

    if not isinstance(target, type(None)):
      color = self.c_cmap(i * 4)
      cov = np.cov(scores.T)
      mu = np.average(scores, axis=0)

      exp_x = np.exp(-(X - mu[0])**2 / (2 * cov[0, 0]))
      exp_y = np.exp(-(Y - mu[1])**2 / (2 * cov[1, 1]))

      scale_x = 1 / np.sqrt(2 * np.pi * cov[0, 0])
      scale_y = 1 / np.sqrt(2 * np.pi * cov[1, 1])

      projection_x = scale_x * exp_x
      projection_y = scale_y * exp_y

      projection_axes[1].plot(X, projection_x, color=color)
      projection_axes[0].plot(projection_y, Y, color=color)

  def biplot(self, FOI=None, component_x=1, component_y=2, loading_scale=1):
    """
    A powerful plot technique that shows the scores plot as well as the
    loading plot of the features of interest (FOIs) for the given components.
    The loading plot is scaled differently than the scores plot.

    Parameters
    ----------
    FOI : list
      Mask that can highlight some relevant FOInts.
    component_x : int, optional
      Defines the displayed component in x direction. The default is 1.
    component_y : int, optional
      Defines the displayed component in y direction. The default is 2.
    loading_scale : float, optional
      Scaling factor for the loading scale. The default is 1 and scales the
      loading axes such that the absolute loading maximum is as big as the
      absolute score maximum.

    Returns
    -------
    None.

    """
    self.container.get_scores()

    x, y = component_x - 1, component_y - 1
    limit = np.amax(np.abs(self.data.scores[:, [x, y]])) * 1.05

    fig = plt.figure('Biplot', figsize=(half_width, 3.1))
    fig.clf()
    ax = fig.add_subplot(111)

    # plot scores:
    if self.component_type == 'PC':
      self.scores_pca(ax, scores=None,
                      target=self.data.y_train,
                      component_x=component_x, component_y=component_y)
    elif self.component_type == 'LD':
      self.scores_lda(ax, scores=None,
                      target=self.data.y_train,
                      component_x=component_x, component_y=component_y)

    # plot loadings as arrows:
    # scaling loadings on same scale as scores
    scale = (np.amax(np.abs(self.data.scores))
             / np.amax(np.abs(self.data.loadings)))
    scale = scale * loading_scale

    def scaling_down(x, scale = scale):
      return x * scale

    def scaling_up(x, scale = scale):
      return x / scale

    ax1 = ax.secondary_xaxis('top', functions=(scaling_up, scaling_down))
    ax2 = ax.secondary_yaxis('right', functions=(scaling_up, scaling_down))

    if FOI:
      FOI = np.array(FOI, ndmin=1)

      for P in FOI:
        if not P in self.data.features:
          continue
        idx = np.argwhere(self.data.features == np.array(P))

        if self.data.loadings[x, idx] > 0:
          ha = 'left'
        else:
          ha = 'right'

        if self.data.loadings[y, idx] > 0:
          va = 'bottom'
        else:
          va = 'top'

        ax.annotate(P, xy=(0, 0),
                    xytext=(self.data.loadings[x, idx] * scale,
                            self.data.loadings[y, idx] * scale),
                    va=va, ha=ha, color='k',
                    arrowprops=dict(arrowstyle='<|-', shrinkA=0,
                                    shrinkB=0, color='k'))

    # x-axis:
    ax.set_xlabel(f"score {self.components['name'][x]}")
    ax1.set_xlabel(f"loading {self.components['name'][x]}")
    ax.set_xlim(-limit, limit)

    # y-axis:
    ax.set_ylabel(f"score {self.components['name'][y]}")
    ax2.set_ylabel(f"loading {self.components['name'][y]}")
    ax.set_ylim(-limit, limit)

    if self.component_type == "LD":
      ax.legend(loc='upper right')

  # still to be investigated:
  def decode(self):
    if self._reduced == False:
        self.get_reduced_X()

    n_pixels = self.n_samples * self.n_features
    reduction = np.round((1 - (self.scores.size
                               + self.evecs[:self.n_components].size)
                          / n_pixels) * 100, 2)
    mse = np.sum((self.reduced_X - self.preprocessed_X)**2) / n_pixels
    psnr = -np.log10(mse / (np.amax(self.preprocessed_X)
                            + np.abs(np.amin(self.preprocessed_X)))**2)
    fig, axs = plt.subplots(2,1)
    axs[0].imshow(self.X.T)
    axs[0].title.set_text("Original")
    axs[1].imshow(self.reduced_X.T)
    axs[1].title.set_text(f"Reduction: {reduction} \%; PSNR: {psnr:0.2f} dB")