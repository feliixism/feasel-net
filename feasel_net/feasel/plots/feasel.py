import numpy as np

# matplotlib imports
from matplotlib import cm
import matplotlib.pyplot as plt

from .neural_network import NeuralNetworkVisualizer

# PLOT PROPERTIES:
# figure size:
full_width = 7.14
half_width = 3.5

class FeaselVisualizer(NeuralNetworkVisualizer):
  def __init__(self, model_container):
    super().__init__(model_container)

  @property
  def _pruned(self):
    return self._is_pruned()

  @property
  def log(self):
    return self.container.callback.log

  def _is_pruned(self):
    """
    Checks whether feature selection algorithm already pruned a feature or not.
    If not, the class will  not plot anything, since the necessary information
    is not provided.

    Returns
    -------
    _pruned : bool
      True, if network specified in the container is pruned at least once.

    Raises
    ------
    NotImplementedError
      If the feature selection algorithm did not prune anything.

    """
    _pruned = self.log._pruned
    if not _pruned:
      raise NotImplementedError("The feature selection did not return "
                                "anything printworthy.")
    return _pruned

  def pruning_history(self):
    """
    Plots the evaluation history of the feature selection algorithm and marks
    the epochs where features have been pruned with vertical lines.

    Returns
    -------
    None.

    """
    self._pruned
    fig = plt.figure('Pruning History', figsize=(half_width, 2))
    plt.clf()

    ax = fig.add_subplot(111)
    ax = self._pruning_history(ax)

  def _pruning_history(self, ax):
    """
    Plots the pruning history of the feature selection algorithm where the
    number of features is plotted over the epochs.

    Parameters
    ----------
    ax : axes
      The axes where the pruning history is plotted.

    Returns
    -------
    ax : axes
      The axes where the pruning history is plotted.

    """
    # plot content and informative variables

    n_prunes, n_features = self.log.m_k.shape
    F = self.log.f_n.astype(int)

    E = self.log.e_prune

    # plotting settings for the pruning history
    ax.plot(E, F, color = "k", marker = ".")

    ax.set_xlabel("epoch")
    ax.set_ylabel("number of features")

    y_max = int(F[0]) * 1.05

    if self.log.e_stop:
      np.append(E, self.log.e_stop)

    if E.size > 2:
      # first prune
      first = int(E[1])
      ax.axvline(first, color = "k", ls = "-.")
      ax.text(E[1], y_max/2,
              f"first prune: $e={first}$",
              rotation = "vertical",
              horizontalalignment = "center",
              verticalalignment = "center", color = "k",
              bbox=dict(facecolor='w'))

      # last prune
      if not self.container.callback.trigger._converged:
        last = int(E[-1])
      else:
        last = int(E[-2])
      ax.axvline(last, color = "k", ls = "-.")
      ax.text(last, y_max / 2,
              f"last prune: $e={last}$",
              rotation = "vertical",
              horizontalalignment = "center",
              verticalalignment = "center", color = "k",
              bbox=dict(facecolor='w'))

    elif E.size == 2:
      # only prune
      only = int(E[1])
      ax.axvline(only, color = "k", ls = "-.")
      ax.text(only, y_max / 2,
              f"only prune: $e={only}$",
              rotation = "vertical",
              horizontalalignment = "center",
              verticalalignment = "center", color = "k",
              bbox=dict(facecolor='w'))

    ax.set_ylim(0, F[0] * 1.05)
    ax.set_xlim(0, E[-1])

    ax.grid(True)

    return ax

  def mask_history(self, highlight=False):
    """
    Provides a colorized history of the masks where brighter colors indicate
    features that were pruned earlier in the training and darker colors
    features pruned later.

    Parameters
    ----------
    highlight : bool
      If True, it will highlight the remaining features. The default is False.

    Returns
    -------
    None.

    """
    self._pruned
    fig = plt.figure('Mask History', figsize=(half_width, 2))
    plt.clf()

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 20])

    # mask history:
    ax1 = fig.add_subplot(gs[0, 1])
    self._mask_history(ax1, highlight)

    # colorbar:
    ax2 = fig.add_subplot(gs[0, 0])
    self._colorbar(fig, ax2)

  def _mask_history(self, ax, highlight):
    """
    Provides a colorized history of the masks where brighter colors indicate
    features that were pruned earlier in the training and darker colors
    features pruned later.

    Parameters
    ----------
    ax : axes
      The axes where the mask history is plotted.
    highlight : bool
      If True, it will highlight the remaining features.

    Returns
    -------
    ax : axes
      The axes where the mask history is plotted.

    """
    cmap = self.default.im_cmap.reversed() # if reversed cmap: .reversed()

    color = self.c_cmap(4) # orange color

    weights = self.log.m_k
    n_prunes, n_features = weights.shape

    # plotting settings for the mask history
    masks = np.sum(weights, axis = 0, keepdims=True)
    masks = np.amax(masks) - masks
    last = np.array(weights[-1], ndmin=2)

    if self.container.features is not None:
      features = self.container.features

      # pcolormesh is used to plot mask history if features are linearly
      # distributed, i.e. equally distributed numbers that can be rescaled:
      if self._check_linearity(features):
        features_cmesh = np.linspace(features[0], features[-1],
                                     len(features)+1)
        ax.pcolormesh(features_cmesh, np.arange(2), masks, cmap=cmap)
        ax.set_xlim(features[0], features[-1])
        # plot of the remaining features

        if highlight:
          ratio = len(features)/len(features_cmesh)
          ax.bar(features_cmesh[:-1] + 0.5 * ratio, last[0] * 0.5,
                 width=ratio,
                 edgecolor=color, facecolor=color,
                 ls='-', lw=1, label='remaining features')

        # inversion of the x-axis if required.
        if features[-1] < features[0]:
          ax.invert_xaxis()

      else:
        # plots an image with summed up weight matrices:
        ax.imshow(masks, cmap=cmap, aspect="auto", vmin=0, vmax=n_prunes)
        # plot of the remaining features

        if highlight:
          ax.bar(features, last[0]*0.5, width=1, edgecolor=color, facecolor=color,
                 ls='-', lw=1, label='remaining features')

        # enables 10 ticks in x-axis with strings
        ratio = int(n_features / 10)

        ax.set_xticks(np.arange(len(features))[::ratio])
        ax.set_xticklabels(features[::ratio], rotation=-45, ha='left')

    else:
      features = np.arange(len(masks[0]))
      ax.imshow(masks, cmap = cmap, aspect = "auto")
      # plot of the remaining features
      if highlight:
        ax.bar(features, last[0]*0.5, width=1, edgecolor=color, facecolor=color,
               ls='-', lw=1, label='remaining features')
        ax.legend(loc='upper right')

    # y-axis
    ax.set_yticks([])
    ax.set_ylim(0, 0.5)

    ax.grid(False)
    return ax

  def _check_linearity(self, array):
    """
    Checks the linearity of an array, to ensure correct xtick labels.

    Parameters
    ----------
    array : nd-array
        The xtick-array.

    Returns
    -------
    bool
        True if xticks array is linear.

    """
    try:
      comparison_array = np.linspace(array[0], array[-1], len(array))
      return np.array_equal(array, comparison_array)

    except:
      return False


  def _colorbar(self, fig, ax):
    """
    Generates a colorbar that maps the feature masks onto the epoch that the
    specific feature has been pruned.

    Parameters
    ----------
    fig : figure
      The figure where the colorbar is plotted.
    ax : axes
      The axes where the colorbar is plotted.

    Returns
    -------
    colorbar : colorbar
      The colorbar representing the epochs where features have been pruned.

    """
    cmap = self.default.im_cmap.reversed()

    n_prunes, n_features = self.log.m_k.shape
    weight_sum = np.sum(self.log.m_k, axis=0)

    if n_prunes > 6:
      n_ticks = 6

    elif n_prunes < 2:
      n_ticks = 2

    else:
      n_ticks = n_prunes

    ticks = np.linspace(0, 1, n_ticks, endpoint=True)
    n_masked = np.round(np.linspace(0, np.amax(weight_sum), n_ticks), 1)

    colorbar = fig.colorbar(cm.ScalarMappable(cmap=cmap),
                            ticks=ticks, cax=ax, aspect=20)

    # y axis:
    colorbar.ax.set_yticklabels(n_masked)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.invert_yaxis()
    colorbar.set_label("times masked $[\mathrm{I/O}]$")

    return colorbar

  def input_reduction(self, plot='both', highlight=False):
    """
    Plots the reduction of the input nodes over the epoch and a color gradient
    that indicates the importance of features regarding their mask history.
    Darker colors indicate more important features whereas brighter colors
    stand for features that have been pruned early and are not that important.

    Parameters
    ----------
    plot : str, optional
      Defines which plot is shown. Possible arguments are 'mask', 'prune' or
      'both'. The default is 'both'.
    highlight : bool, optional
      Is only needed for the mask history. If True, it highlights the remaining
      features. The default is False.

    Raises
    ------
    NameError
      If argument in 'plot' is not valid.

    Returns
    -------
    None.

    """
    self._pruned

    fig = plt.figure("Input Reduction", figsize=(half_width, 4))
    plt.clf()

    if plot == "both":
      gs = fig.add_gridspec(2, 2, width_ratios = [1, 20])
      ax1 = fig.add_subplot(gs[0, :])
      ax1 = self._pruning_history(ax1)

      ax2 = fig.add_subplot(gs[1, 0])
      ax2 = self._colorbar(fig, ax2)

      ax3 = fig.add_subplot(gs[1, 1])
      ax3 = self._mask_history(ax3, highlight)

    elif plot == 'mask':
      plt.close()
      self.mask_history()
      fig = plt.gcf()
      fig.canvas.set_window_title("Input Reduction")

    elif plot == 'prune':
      plt.close()
      self.pruning_history()
      fig = plt.gcf()
      fig.canvas.set_window_title("Input Reduction")

    else:
      raise NameError(f"'{plot}' is not a valid argument for 'plot'. "
                      "Try 'mask', 'prune' or 'both' instead.")

  def feature_history(self):
    """
    Plots the history of the evaluation metric used for the feature selection.
    Each pruning epoch is marked by a vertical line.

    Returns
    -------
    None.

    """
    self._pruned
    log = self.container.callback.log
    trigger = self.container.callback.trigger
    params = self.container.callback.feasel_params

    colors = [self.default.c_cmap(0), # used for accuracy-based FS
              self.default.c_cmap(4), # used for loss-based FS
              self.default.c_cmap(8), # used for the gradient of the loss
              self.default.c_cmap(16)] # used for the threshold

    if params.eval_metric in ['loss', 'val_loss']:
      fig = plt.figure('Pruning History', figsize=(half_width, 4))
      plt.clf()

      # two subplots are needed since the evaluation is based on the gradient
      # of the loss and not on the loss itself
      ax1 = fig.add_subplot(211)
      ax1.plot(trigger.value_l, color=colors[1], label='loss')
      ax1.tick_params(labelbottom=False)

      ax2 = fig.add_subplot(212, sharex=ax1)
      ax2.plot(trigger.gradient_l, color=colors[2], label='gradient')
      ax2.plot(trigger.thresh_l, color=colors[3], label='threshold')
      ax2.set_ylabel('gradient')
      ax2.set_xlabel('epoch')
      ax2.legend(loc='lower right')

      for ax in [ax1, ax2]:
        for i in log.pruning_epochs:
          ax.axvline(i, c='k', lw=0.5)

    else:
      fig = plt.figure('Pruning History', figsize=(half_width, 2))
      plt.clf()
      # only one axis is needed for the history
      ax1 = fig.add_subplot(111)
      ax1.plot(trigger.value_l, color=colors[0], label='accuracy')
      ax1.plot(trigger.thresh_l, color=colors[3], label='threshold')
      ax1.legend(loc='lower right')
      ax1.set_xlabel('epoch')

      for i in log.pruning_epochs:
        ax1.axvline(i, c='k', lw=0.5)

    x = np.arange(len(trigger.value_l))
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylabel(f'{params.eval_metric}')

  def feature_entropy(self, pruning_step=None):
    """
    Generates a bar and box plot where the decision metrics for the feature
    pruning are shown.

    Parameters
    ----------
    pruning_step : int, optional
      The pruning step that shall be investigated. A list of integers is also
      allowed. If None, it will show all feature pruning steps. The default is
      None.

    Returns
    -------
    None.

    """
    self._pruned
    metric = self.container.params.callback.eval_metric
    norm = self.container.params.callback.eval_normalization

    if metric:
      label = f'{norm} {metric}'
    else:
      label = f'{metric}'

    log = self.container.callback.log

    if pruning_step:
      pruning_step = np.array([pruning_step-1])

    else:
      pruning_step = np.arange(len(log.f_loss))

    for i in pruning_step:
      # the masks for each pruning step (applied on second subplot):
      try:
        mask_k = log.m_k[i+1]
        mask_p = log.m_p[i+1]

      except:
        return

      colors = [self.default.c_cmap(0), self.default.c_cmap(19)]

      fig = plt.figure(f'Feature evaluation losses at {i+1}. Prune '
                       f'(Epoch: {log.e_prune[i+1]})',
                       figsize=(half_width, 4))
      plt.clf()

      # first subplot: box plot for the entropy all samples with each masked
      # feature
      ax1 = fig.add_subplot(211)
      ax1.boxplot(log.f_loss[i].T,
                  labels=np.arange(self.container.n_in))
      ax1.set_ylabel(label)
      ax1.set_xlim(0.5, self.container.n_in+0.5)
      ax1.tick_params(labelbottom=False)

      # second subplot: bar plot with feature selection metric and masks
      ax2 = fig.add_subplot(212, sharex=ax1)
      ax2.bar(np.argwhere(mask_k).squeeze()+1,
              log.f_eval[i][mask_k],
              color=colors[0], label='keep')
      ax2.bar(np.argwhere(mask_p).squeeze()+1,
              log.f_eval[i][mask_p],
              color=colors[1], label='prune')
      ax2.set_ylabel(f'{self.container.params.callback.decision_metric} '
                     f'{label}')
      ax2.set_xlabel('feature')
      ax2.legend(loc='upper right')