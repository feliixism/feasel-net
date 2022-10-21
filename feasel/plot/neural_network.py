"""
feasel.plot.neural_networks
===========================
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from scipy.optimize import curve_fit
from tensorflow.keras.utils import plot_model

# matplotlib imports
import matplotlib.pyplot as plt

from .base import Base
from ..data import metrics

# PLOT PROPERTIES:
# figure size:
full_width = 7.14
half_width = 3.5

class NeuralNetworkVisualizer(Base):
  def __init__(self, model_container):
    super().__init__()
    self.container = model_container
    self.model = self.container.model
    self.set_path(Path.cwd() / 'plots' / self.container.name /
                  self.container.timestamp)

  @property
  def _trained(self):
    return self._is_trained()

  def _is_trained(self):
    """
    Checks whether network is already trained or not. If not, the class will
    not plot anything, since the necessary information is not provided.

    Returns
    -------
    _trained : bool
      True, if network specified in the container is trained.

    """
    _trained = self.container.params.train._trained
    return _trained

  # SETTERS:
  def set_modelname(self, modelname=None):
    """
    Sets the modelname for saving the plots.

    Parameters
    ----------
    modelname : str, optional
      The model name for saving the plots. If None, it will automatically use
      the name specified in the container. The default is None.

    Returns
    -------
    modelname : str
      A string of the model.

    """
    if not modelname:
      modelname = f"{self.container.name}"

    else:
      modelname = modelname

    return modelname

  def _get_subplot_array(self, n_plots):
    """
    Calculates a subplot array from a given number of plots.

    Parameters
    ----------
    n_plots : int
      Number of plots.

    Returns
    -------
    rows : int
      Number of rows.
    cols : int
      Number of columns.

    """
    rows = np.ceil(np.sqrt(n_plots)).astype(int)
    cols = np.floor(np.sqrt(n_plots)).astype(int)
    return rows, cols

  # LAYER HELPERS:
  def _get_layers(self):
    """
    Provides a dictionary with all layers and their corresponding positions.

    Returns
    -------
    layer_dict : TYPE
      DESCRIPTION.

    """
    layers = []
    for layer in self.model.layers:
      layers.append(layer)
    return layers

  def _get_layer_weights(self, layername, type=None):
    """
    Provides weights and biases of the specified layer.

    Parameters
    ----------
    layername : str
      A string representing the layer name.
    type : str, optional
      The weight type that shall be returned. If 'bias' or 'weights', it will
      return the chosen type. If 'None', it will return both in a tuple. The
      default is None.

    Raises
    ------
    NameError
      If weights cannot be found.

    Returns
    -------
    TYPE
      DESCRIPTION.

    """
    try:
      weights = self.container.get_weights(layername, type)
      return weights

    except:
      raise NameError(f"Layer '{layername}' does not provide weights and/or "
                      "biases.")

  def history(self, metric=None):
    """
    Plots the history for the metrics that have been logged, i.e. 'accuracy' or
    'loss'.

    Parameters
    ----------
    metric : str
      Specifies the metric that is plotted. The default is None.

    Returns
    -------
    None.

    """
    self._trained
    history = self.container.history.history
    metric = [metric]

    if not metric[0]:
      fig = plt.figure('Training History', figsize=(half_width, 4))
      fig.clf()
      ax1 = fig.add_subplot(211)
      ax2 = fig.add_subplot(212, sharex=ax1)
      metric = ['accuracy', 'loss']

    elif metric[0] == 'accuracy':
      fig = plt.figure('Training History', figsize=(half_width, 2))
      fig.clf()
      ax1 = fig.add_subplot(111)

    elif metric[0] == 'loss':
      fig = plt.figure('Training History', figsize=(half_width, 2))
      fig.clf()
      ax2 = fig.add_subplot(111)

    # data
    x = np.arange(len(history['accuracy'])) + 1
    if 'val_accuracy' in history:
      accuracy = np.array([history['accuracy'], history['val_accuracy']]) * 100
      labels_acc = ['accuracy', 'val_accuracy']
      colors_acc = self.c_cmap([0, 2])
      loss = np.array([history['loss'], history['val_loss']])
      labels_loss = ['loss', 'val_loss']
      colors_loss = self.c_cmap([4, 6])

    else:
      accuracy = np.array(history['accuracy']) * 100
      labels_acc = ['accuracy']
      colors_acc = self.c_cmap([0])
      loss = np.array(history['loss'])
      labels_loss = ['loss']
      colors_acc = self.c_cmap([4])

    for i in range(len(labels_acc)):
      if 'accuracy' in metric:
        ax1.plot(x, accuracy[i], c=colors_acc[i], label=labels_acc[i])
        ax1.set_ylabel('accuracy $[\%]$')
        ax1.legend(loc='lower right')

      if 'loss' in metric:
        ax2.plot(x, loss[i], c=colors_loss[i], label=labels_loss[i])
        ax2.set_ylabel('loss')
        ax2.legend(loc='upper right')

    for ax in fig.axes:
      ax.set_xlim(x[0], x[-1])
      ax.set_xlabel('epoch')

    if len(fig.axes) == 2:
      ax1.set_xlabel(None)
      ax1.tick_params(labelbottom=False)

  def model(self, directory=None, modelname=None):
    """
    Saves an image of the visualization of the model's summary using a function
    from keras.

    Parameters
    ----------
    directory : str, optional
      The directory where the plots are saved. The default is None.
    modelname : str, optional
      The modelname of the plots. It generates another folder for it. The
      default is None.

    Returns
    -------
    None.

    """
    # The option "layer_range=None" has not been implemented in keras yet.
    path = self.set_path(directory, modelname)
    filename = path / Path("model" + self.container.time + ".png")
    plot_model(self.model,
               filename,
               show_shapes=True,
               dpi=100)

  def feature_maps(self, X, layername):
    """
    Plots feature maps for given input X at the specified layer.

    Parameters
    ----------
    X : ndarray
      Defines the input data. It is only allowed to use one sample up to now.
    layer_name : str
      Defines name of the layer from which the feature map is extracted.

    """
    self._trained

    # provide model with an output at desired layer:
    feature_maps = self.container.get_feature_maps(X, layername)
    n_feature_maps = len(feature_maps[0])

    rows, cols = self._get_subplot_array(n_feature_maps)

    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True,
                            figsize=(half_width*cols, 2*rows),
                            num=f'Feature Maps at {layername}')

    axs = np.array(axs, ndmin=2)
    idx = 0

    for i in range(rows):
      for j in range(cols):
        if feature_maps[0][idx].ndim == 1:
          x = np.arange(len(feature_maps[0][idx]))
          axs[i, j].plot(feature_maps[0][idx])
          axs[i, j].set_xlim(x[0], x[-1])

        elif feature_maps[0][idx].ndim == 2:
          axs[i, j].imshow(feature_maps[0][idx])
        idx += 1

        if idx == n_feature_maps:
          break

    return feature_maps

  # analyzers
  def predict(self, X, y=None, model=None):
    """
    Predicts a single sample and plots the probability for a correct
    classification in each class. The ground truth will be plotted as a dashed
    bar plot.

    Parameters
    ----------
    X : ndarray
      A single sample input array.
    y : ndarray, optional
      A single target array. The default is None.
    model : keras.Model, optional
      A keras model that replaces the model instantiated in the container
      class. If None, it will use the container model. The default is None.

    Returns
    -------
    None.

    """
    self._trained
    y_pred, y_true = self.container.test(X, y, model=model)

    fig = plt.figure('Prediction of a sample', figsize=(half_width, 2))
    fig.clf()
    ax = fig.add_subplot(111)
    classes = np.arange(self.container.data.n_classes)
    # plot data
    ax.bar(classes, y_pred[0]*100, color=self.default.c_cmap(17),
          label='prediction')
    if y_true is not None:
      ax.bar(classes, y_true[0]*100, edgecolor=self.default.c_cmap(18), lw=1,
             ls='--', fill=False, label='ground truth')

    for i in range(len(classes)):
      if y_pred[0, i] < 0.05:
        ax.text(i, y_pred[0, i]*100, s=np.round(y_pred[0, i]*100,1),
                ha='center', va='bottom', color='k')
      else:
        ax.text(i, y_pred[0, i]*100/2, s=np.round(y_pred[0, i]*100,1),
                ha='center', va='center', color='k')
    # x-axis:
    ax.set_xticks(classes)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_xlabel('classes')
    # y-axis:
    ax.set_ylabel('probability $[\%]$')
    ax.set_ylim(0, 105)
    # legend:
    ax.legend(loc='upper right')

  def predict_set(self, X, y,
                 metric='accuracy',
                 model=None):
    self._trained
    y_pred, y_true = self.container.test(X, y, model=model)

    METRIC = {'accuracy': self._accuracy,
              'sensitivity': self._sensitivity,
              'specificity': self._specificity,
              'f1-score': self._f1_score,
              'precision': self._precision}

    y_pred = np.array(y_pred.max(axis=1, keepdims=1) == y_pred).astype(int)
    y_pred = self.container.data.one_hot_labels(y_pred)
    y_true = self.container.data.one_hot_labels(y_true)

    if isinstance(metric, list):
      fig = plt.figure('Test prediction', figsize=(half_width, 2))
      fig.clf()
      ax = fig.add_subplot(111)
      self._multiple_metrics(ax, y_true, y_pred, metric)

    else:
      fig = plt.figure(f'Test {metric}', figsize=(half_width, 2))
      fig.clf()
      ax = fig.add_subplot(111)

      METRIC[metric](ax, y_true, y_pred)

    return fig

  def _multiple_metrics(self, ax, y_true, y_pred, metric=None):
    METRICS = {'accuracy': metrics.accuracy,
               'sensitivity': metrics.sensitivity,
               'specificity': metrics.specificity,
               'precision': metrics.precision,
               'f1-score': metrics.f1_score}

    values = []
    for met in metric:
      try:
        values.append(METRICS[met](y_pred, y_true))
      except:
        raise KeyError(f"{met} is not a valid key for 'metrics'.")

    n_metrics = len(values)
    n_classes = len(self.container.data.classes)
    x = np.arange(n_classes)

    width = 0.8

    for i, val in enumerate(values):
      ax.bar(x + i / n_metrics * width, val*100,
             width=1/n_metrics * width, label=metric[i])

    # x-axis:
    ax.set_xticks(x+width/n_metrics)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_xlabel('class')

    # y-label:
    ax.set_ylabel('percentage [\%]')
    ax.set_ylim(0, 105)

    ax.legend(loc='lower right')


  def _accuracy(self, ax, y_true, y_pred, color=None):
    if not color:
      color = self.default.c_cmap(13)

    ACC = metrics.accuracy(y_pred, y_true)
    x = np.arange(len(ACC))

    for i, acc in enumerate(ACC):
      if acc < 0.05:
        ax.text(i, acc*100, s=np.round(acc*100, 1),
                ha='center', va='bottom', color='k')
      else:
        ax.text(i, acc*100/2, s=np.round(acc*100, 1),
                ha='center', va='center', color='k')

    ax.bar(x, ACC*100, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_xlabel('class')
    ax.set_ylim(0,105)
    ax.set_ylabel('accuracy [\%]')

    return ax

  def _sensitivity(self, ax, y_true, y_pred, color=None):
    if not color:
      color = self.default.c_cmap(13)

    TPR = metrics.sensitivity(y_pred, y_true)
    x = np.arange(len(TPR))

    for i, tpr in enumerate(TPR):
      if tpr < 0.05:
        ax.text(i, tpr*100, s=np.round(tpr*100, 1),
                ha='center', va='bottom', color='k')
      else:
        ax.text(i, tpr*100/2, s=np.round(tpr*100, 1),
                ha='center', va='center', color='k')

    ax.bar(x, TPR*100, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_xlabel('class')
    ax.set_ylim(0,105)
    ax.set_ylabel('sensitivity [\%]')

    return ax

  def _specificity(self, ax, y_true, y_pred, color=None):
    if not color:
      color = self.default.c_cmap(13)

    TNR = metrics.specificity(y_pred, y_true)
    x = np.arange(len(TNR))

    for i, tnr in enumerate(TNR):
      if tnr < 0.05:
        ax.text(i, tnr*100, s=np.round(tnr*100, 1),
                ha='center', va='bottom', color='k')
      else:
        ax.text(i, tnr*100/2, s=np.round(tnr*100, 1),
                ha='center', va='center', color='k')

    ax.bar(x, TNR*100, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_xlabel('class')
    ax.set_ylim(0,105)
    ax.set_ylabel('specificity [\%]')

    return ax

  def _precision(self, ax, y_true, y_pred, color=None):
    if not color:
      color = self.default.c_cmap(13)
    PPV = metrics.precision(y_pred, y_true)
    x = np.arange(len(PPV))

    for i, ppv in enumerate(PPV):
      if ppv < 0.05:
        ax.text(i, ppv*100, s=np.round(ppv*100, 1),
                ha='center', va='bottom', color='k')
      else:
        ax.text(i, ppv*100/2, s=np.round(ppv*100, 1),
                ha='center', va='center', color='k')

    ax.bar(x, PPV*100, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_xlabel('class')
    ax.set_ylim(0,105)
    ax.set_ylabel('precision [\%]')

    return ax

  def _f1_score(self, ax, y_true, y_pred, color=None):
    if not color:
      color = self.default.c_cmap(13)
    F1 = metrics.f1_score(y_pred, y_true)
    x = np.arange(len(F1))

    for i, f1 in enumerate(F1):
      if f1 < 0.05:
        ax.text(i, f1*100, s=np.round(f1*100, 1),
                ha='center', va='bottom', color='k')
      else:
        ax.text(i, f1*100/2, s=np.round(f1*100, 1),
                ha='center', va='center', color='k')

    ax.bar(x, F1*100, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_xlabel('class')
    ax.set_ylim(0,105)
    ax.set_ylabel('f1-score [\%]')

    return ax

  def confusion_matrix(self, X, y, normalize=True):
    """
    Plots the confusion matrix of a test dataset.

    Parameters
    ----------
    X : ndarray
      Test input data array.
    y : ndarray
      Target array.
    normalize : bool
      Defines whether the confusion matrix is normalized. The default is True.

    Returns
    -------
    cm : ndarray
      The confusion matrix.

    """
    self._trained

    if normalize is True:
      normalize = 'true'
    else:
      normalize = None

    # get true and predicted targets:
    y_pred, y_true = self.container.test(X, y)

    # sk-learn only accepts categorical classes: transformation to categorical
    y_pred = np.argmax(y_pred, axis=1)
    y_true = self.container.data.categorical_labels(y_true)

    # normalized confusion matrix:
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    if normalize:
      cm = cm * 100

    classes = self.container.data.classes

    # start of plot:
    fig = plt.figure("Confusion Matrix", figsize=(2, 2))
    plt.clf()

    ax = fig.add_subplot(111)
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
        val = np.round(cm[i, j], 1)
        if val < 50:
          ax.text(j, i, f'{val}', color='k', ha='center', va='center',
                  size='small')
        else:
          ax.text(j, i, f'{val}', color='white', ha='center', va='center',
                  size='small')

    # x-axis:
    ax.set_xticks(range(len(classes)), classes,
                  rotation = 45, horizontalalignment = "right")
    ax.set_xlabel('prediction')
    ax.xaxis.set_label_position('top')

    # y-axis:
    ax.set_yticks(range(len(classes)), classes)
    ax.set_ylabel('ground truth')
    ax.yaxis.set_label_position('right')

    ax.grid(False)

    return cm

  def ROC(self, X, y, fit=False):
    """
    The receiver operating characteristic (ROC) curve is usually applied on a
    binary classifier that varies the thresholds and thus the weights for the
    prediction. To handle multi-class predictions, a different metric is used
    that only regards the TPRs and FPRs of each class. Therefore a subplot of a
    ROC for each class is shown.

    Parameters
    ----------
    X : ndarray
      Input test data.
    y : ndarray
      Target array.
    fit : bool
      If True, it applies a curve fitting function from the scipy.optimizer
      module. The default is False.

    Returns
    -------
    None.

    """

    # get true and predicted targets:
    y_pred, y_true = self.container.test(X, y)
    TPR, FPR = metrics.ROC(y_pred, y_true)

    fig = plt.figure(num='Receiver operating characteristic curve',
                     figsize=(half_width, 2))
    fig.clf()
    ax = fig.add_subplot(111)

    def func(x, a, b):
      """
      A weibull function fits well for the ROC curve.

      Parameters
      ----------
      x : ndarray
        X data.
      a : float
        Scale value.
      b : float
        Exponent value.

      Returns
      -------
      y : ndarray
        The output array.

      """
      y = 1 - np.e**(-(a * x)**b)
      return y

    ax.plot([0, 1.05], [0, 1.05], color=self.c_cmap(16), ls='--',
            label='worst classifier')

    if fit:
      x = np.linspace(0, 1.05, 1000)
      popt, pcov = curve_fit(func, FPR, TPR)
      y = func(x, *popt)
      ax.plot(x, y, label='weibull fit')

    else:
      idx = np.argsort(FPR)
      x = np.insert(np.insert(FPR[idx], 0, 0), -1, 1)
      y = np.insert(np.insert(TPR[idx], 0, 0), -1, 1)
      ax.plot(x, y, label='interpolation')

    ax.scatter(FPR, TPR, s=8)

    # x-axis:
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('false positive rate')
    # y-axis:
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.set_ylabel('true positive rate')

    ax.set_aspect(1)

  def weights(self, type='weights', layernames=None):
    """
    Plots weights values.

    Parameters
    ----------
    type : str
      The weight type accepts 'weights', 'bias' or 'both'. The default is
      "weights".
    layernames : str (or: list of str)
        Defines name of layer(s). If None, it will plot all layers. The default
        is None.
    """
    self._trained

    if type == 'weights':
      i_l = [0]
    elif type == 'bias':
      i_l = [1]
    elif type == 'both':
      i_l = [0, 1]
    else:
      raise TypeError(f"'{type}' is invalid for 'type'. Try ''bias' or "
                      "'weight'.")

    if not layernames:
      layers = self._get_layers()
      layernames = []

      for layer in layers:
        layernames.append(layer.name)

    weights = {}

    for name in layernames:
      weights[name] = self._get_layer_weights(name)

    for name in weights:
      w = weights[name]

      for i in i_l:
        i_map = {0: 'weights',
                 1: 'bias'}

        if (i == 1 and len(w) < 2) or (len(w) == 0):
          print(f"Layer '{name}' does not have any {i_map[i]} to plot.")

        else:
          fig = plt.figure(f"Layer '{name}' {i_map[i]}.",
                           figsize=(2, 2))
          fig.clf()
          if w[i].ndim == 1:
            ax = fig.add_subplot(111)
            ax.plot(w[i])
            ax.set_xlim(0, len(w[i])-1)
            ax.set_xlabel('nodes $n^{(l)}$')

          elif w[i].ndim == 2:
            ax = fig.add_subplot(111)
            ax.imshow(w[i], cmap=self.im_cmap)
            ax.set_xlabel('nodes $n^{(l)}$')
            ax.set_ylabel('nodes $n^{(l-1)}$')
            ax.grid(False)

          elif w[i].ndim == 3:
            rows, cols = self._get_subplot_array()
            fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
            fig.set_size_inches(2*cols, 2*rows)
            j=0

            for m in range(rows):
              for n in range(rows):
                axs[n,m].imshow(w[i][j], cmap=self.im_cmap)
                j += 1
                axs[n,m].set_title(f'{j+1}. kernel')
                axs[n,m].grid(False)

          else:
            raise ValueError(f"Dimensionality of layer '{name}' {i_map[i]} to "
                             "high.")