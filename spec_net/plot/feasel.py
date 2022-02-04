import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model

# matplotlib imports
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from .utils import _default

full_width = 7.14
half_width = 3.5

class FeaselVisualizer:
  def __init__(self, model_container):
    self.container = model_container
    self.default = _default.AdvancedPlot()
    self.default.set_default()
    self.default.enable_tex()
    self._model = self.container.model
    self._directory = self.set_directory()

  @property
  def _trained(self):
    return self._trained()

  def _trained(self):
    return self.container.params.train._trained

  @property
  def _pruned(self):
    return self._pruned()

  def _pruned(self):
    return self.container.callback.log._pruned

  def _get_layer_positions(self):
    layer_dict = {}
    for i in range(len(self._model.layers)):
      layer_dict[f"{self.layers[i].name}"] = i
    return layer_dict

  def _get_layer_name(self):
    layer_names = []
    for i in range(len(self._model.layers)):
      layer_names.append(self._model.layers[i].name)
    return layer_names

  def _get_layer_information(self, layer_names, information_type):
    layer_values = []
    names = []
    for i in range(len(layer_names)):
      try:
        weights, bias = self.container.get_weights(layer_names[i])
        names.append(layer_names[i])
        if information_type == "bias":
          layer_values.append(bias)
        elif information_type == "weights":
          layer_values.append(weights)
      except:
        pass
    return layer_values, names

  def set_directory(self, directory=None):
    if not directory:
      directory = (Path.cwd() / "plots")
    else:
      directory = Path(directory)
    return directory

  def _aspect_ratio(n_plots, max_vert = 4):

    hor = np.ceil(n_plots / max_vert).astype(int)
    vert = np.ceil(n_plots / hor)

    return vert, hor

  def set_modelname(self, modelname=None):
    if not modelname:
      modelname = f"{self.container.name}"
    else:
      modelname = modelname
    return modelname

  def set_path(self, directory=None, modelname=None):
    path = self.set_directory(directory) / self.set_modelname(modelname)
    path.mkdir(parents=True, exist_ok=True)
    return path

  def model(self, directory=None, modelname=None):
    # The option "layer_range=None" has not been implemented in keras yet.
    path = self.set_path(directory, modelname)
    filename = path / Path("model" + self.container.time + ".png")
    plot_model(self._model,
               filename,
               show_shapes=True,
               dpi=100)

  def history(self, *metrics, **kwargs):
    history = self.container.history.history

    colors = [self.default.c_cmap(0), self.default.c_cmap(2),
              self.default.c_cmap(4), self.default.c_cmap(6)]

    x = np.arange(len(history['accuracy']))

    # plotting figure
    fig = plt.figure('Training History', figsize=(half_width, 4))
    ax1 = fig.add_subplot(211)
    ax1.plot(np.array(history['accuracy']) * 100,
             color=colors[0], label='accuracy')
    ax1.plot(np.array(history['val_accuracy']) * 100,
             color=colors[1], label='val_accuracy')
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylabel('accuracy [$\%$]')
    ax1.tick_params(labelbottom=False)
    ax1.legend(loc='lower right')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(history['loss'] * 100, color=colors[2], label='loss')
    ax2.plot(history['val_loss'] * 100, color=colors[3],
             label='val_loss')
    ax2.set_xlim(x[0], x[-1])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(loc='upper right')

  def bias(self, *layer_names):
    """
    Plots bias values.

    Parameters
    ----------
    layer_names : str (or: list of str)
        Defines name of layer(s).
    """
    if len(layer_names) == 0:
      layer_names = self._get_layer_name(self)

    biases, names = self._get_layer_information(self, layer_names, information_type = "bias")

    n_biases = len(biases)

    vert, hor = self._aspect_ratio(n_biases)

    fig = plt.figure()
    fig.canvas.set_window_title("Biases")
    for i in range(n_biases - 1):
      i += 1
      ax = fig.add_subplot(vert, hor, i)
      plt.sca(ax)
      x = np.round(np.arange(0, biases[i].shape[0]), 0)
      plt.plot(x, biases[i], label = f"{names[i]}")
      plt.xticks(np.arange(0, len(biases[i]) + 1, len(biases[i]) / 5).astype(int))
      plt.xlim(0, len(biases[i]))
      plt.figaspect(fmt)
      plt.legend(loc = 4)
      plt.grid(True)
    fig.show()

  def _batch_norm(self, layers):
    for layer in layers:
      current_layer = self._model.layers[layer]
      gamma = current_layer._trainable_weights[0].numpy()
      beta = current_layer._trainable_weights[1].numpy()

      mean = current_layer._non_trainable_weights[0].numpy()
      variance = current_layer._non_trainable_weights[1].numpy()
      stdev = np.sqrt(variance)

      upper = mean + stdev
      lower = mean - stdev

      fig = plt.figure()
      fig.canvas.set_window_title("Parameters of BatchNorm-Layer: "
                                  f",{current_layer.name}")
      fig.clf()

      ax = fig.add_subplot(1, 1, 1)
      plt.sca(ax)

      x = np.arange(len(gamma))
      plt.plot(gamma, label = "$\\gamma$")
      plt.plot(beta, label = "$\\beta$")
      plt.plot(mean, label = "$\\bar{x}$", color = "k")
      plt.plot(upper, color = "gray", alpha = 0.2)
      plt.plot(lower, color = "gray", alpha = 0.2)
      plt.fill_between(x, upper, lower, color = "gray", alpha = 0.2)
      plt.grid(True)
      plt.legend(loc = "lower center", ncol = 3)
      plt.xlim(0, len(x) - 1)
      plt.xlabel("Nodes [-]")

  def _dense(self, layers):
    for layer in layers:
      current_layer = self._model.layers[layer]
      weights = current_layer._trainable_weights[0].numpy()

      fig = plt.figure()
      fig.canvas.set_window_title("First Four Node Parameters of ",
                                  f"Dense-Layer: {current_layer.name}")
      fig.clf()

      vert, hor = self._aspect_ratio(4)

      node = 0

      for node in range(4):
        node += 1
        ax = fig.add_subplot(vert, hor, node)
        plt.sca(ax)
        plt.title(f"Node {node}")
        plt.plot(weights[node])
        plt.grid(True)
        plt.xlim(0, len(weights[node]) - 1)
        plt.xlabel("Nodes [-]")

  def _conv1d(self, layers):
    for layer in layers:
      current_layer = self._model.layers[layer]
      conv_weights = current_layer._trainable_weights[0].numpy()
      fig = plt.figure()
      fig.canvas.set_window_title("Parameters of Conv-Layer: "
                                  "f{current_layer.name}")
      fig.clf()

      vert, hor = self._aspect_ratio(conv_weights.shape[-1])

      kernel = 0

      for kernel in range(conv_weights.shape[-1]):
        kernel += 1
        ax = fig.add_subplot(vert, hor, kernel)
        plt.sca(ax)
        plt.title(f"Kernel {kernel}")

        for n_FeatureMap in range(conv_weights.shape[1]):
          plt.plot(conv_weights[:, n_FeatureMap, kernel - 1])

        plt.grid(True)
        plt.xlim(0, len(conv_weights) - 1)
        plt.xlabel("Nodes [-]")

  def has_callback(self):
    if not hasattr(self.container, "callback"):
      raise NameError("Could not find the callback object. "
                      "Please ensure that it is instantiated correctly.")
    elif len(self.container.callback.log.weights) == 1:
      raise NotImplementedError("The feature selection did not return "
                                "anything printworthy.")

  def pruning_history(self):
    fig = plt.figure('Pruning History', figsize=(half_width, 2))
    plt.clf()
    ax = fig.add_subplot(111)
    ax = self._pruning_history(ax)

  def _pruning_history(self, ax):

    self.has_callback() # ensure that there is a callback to work with

    # plot content and informative variables
    log = self.container.callback.log

    weights = np.array(log.weights)
    n_kills, n_features = weights.shape
    n_features = np.array(log.n_features).astype(int)
    epochs = np.array(log.pruning_epochs)

    # plotting settings for the pruning history
    ax.plot(epochs, n_features, color = "k", marker = ".")
    ax.set_xlabel("pruning epochs $e_p$")
    ax.set_ylabel("number of features $n_f$")

    if epochs[1] != epochs[-2]:
      # first prune
      ax.axvline(epochs[1], color = "k", ls = "-.")
      ax.text(epochs[1], n_features[0] / 2,
              f"first prune: {epochs[1]}",
              rotation = "vertical",
              horizontalalignment = "right",
              verticalalignment = "center", color = "k")

      # last prune
      ax.axvline(epochs[-2], color = "k", ls = "-.")
      ax.text(epochs[-2], n_features[0] / 2,
              f"last prune: {epochs[-2]}",
              rotation = "vertical",
              horizontalalignment = "right",
              verticalalignment = "center", color = "k")

    else:
      # first prune
      ax.axvline(epochs[1], color = "k", ls = "-.")
      ax.text(epochs[1], n_features[0] / 2,
              f"only prune: {epochs[1]}",
              rotation = "vertical",
              horizontalalignment = "right",
              verticalalignment = "center", color = "k", size="x-small")

    ax.set_ylim(0, n_features[0])
    ax.set_xlim(0, epochs[-1])

    ax.grid(True)

    return ax

  def mask_history(self, highlight=False):
    fig = plt.figure('Mask History')
    plt.clf()
    gs = fig.add_gridspec(1, 2, figure=fig,
                          width_ratios=[1, 20],
                          wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 1])
    self._mask_history(ax1, highlight)
    ax2 = fig.add_subplot(gs[0, 0])
    self._colorbar(fig, ax2)

  def _mask_history(self, ax, highlight):
    self.has_callback() # ensure that there is a callback to work with
    cmap1 = self.default.im_cmap
    # cmap1 = cmap1.reversed() <--depends on cmap (start w/ lighter colors)
    cmap2 = plt.get_cmap("Wistia")
    c = cmap2(255) # orange color of Wistia cmap

    # plot content and informative variables
    log = self.container.callback.log
    weights = np.array(log.weights)
    n_kills, n_features = weights.shape

    # plotting settings for the mask history
    masks = np.array(np.sum(weights, axis = 0), ndmin = 2)
    last = np.array(weights[-1], ndmin=2)

    if self.container.features is not None:
      features = self.container.features

      if self._check_linearity(features):
        ax.pcolormesh(features, np.arange(2), masks, cmap=cmap1)

        # single pcolormeshes for the plot of the remaining features
        if highlight:
          ax.bar(features + 0.5,
                 weights[-1],
                 width=1,
                 color=c)
        ax.set_xlim(features[0], features[-1])

        # inversion of the x-axis if required
        if features[-1] < features[0]:
          ax.invert_xaxis()

      else:
        ax.imshow(masks, cmap = cmap1, aspect = "auto")
        # plot of the remaining features
        if highlight:
          ax.imshow(last, cmap=cmap2, alpha=last, aspect = "auto")
        features = self.default.replace_char(features, '_', ' ')

        # enables 10 ticks in x-axis with string or
        ratio = int(len(features) / 10)
        ax.set_xticks(np.arange(len(features))[::ratio])
        ax.set_xticklabels(features[::ratio], rotation=45, ha='right')

    else:
      ax.imshow(masks, cmap = cmap1, aspect = "auto")
      # plot of the remaining features
      if highlight:
        ax.imshow(last, cmap=cmap2, alpha=last, aspect = "auto")

    ax.set_xlabel("features")
    ax.grid(False)
    ax.set_yticks([])
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
    self.has_callback() # ensure that there is a callback to work with
    cmap = self.default.im_cmap#.reversed()

    # plot content and informative variables
    log = self.container.callback.log
    weights = np.array(log.weights)
    n_kills, n_features = weights.shape
    n_nodes = np.array(log.n_features).astype(int)
    ratio = int(len(n_nodes) / 6) # secure 6 elements
    if ratio:
      n_nodes = n_nodes[::ratio]

    ticks = np.linspace(0, 1, len(n_nodes), endpoint=True)
    colorbar = fig.colorbar(cm.ScalarMappable(cmap = cmap),
                            ticks = ticks, cax = ax, aspect = 20)
    colorbar.ax.set_yticklabels(n_nodes)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.invert_yaxis()
    colorbar.set_label("mask $[I/O]$")

    return colorbar

  def input_reduction(self, plot='both', highlight=False):

    self.has_callback() # ensure that there is a callback to work with

    fig = plt.figure("Input Reduction")
    plt.clf()

    if plot == "both":
      gs = fig.add_gridspec(2, 2, width_ratios = [1, 20], wspace = 0.05)
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

  def weights(self, *layer_names):
    """
    Plots weights values.

    Parameters
    ----------
    layer_names : str (or: list of str)
        Defines name of layer(s).
    """

    executed = False
    if len(layer_names) == 0:
      layer_names = self._get_layer_name()

    weights, names = self._get_layer_information(layer_names=layer_names,
                                                 information_type="weights")

    n_weights = len(weights)
    positions = {"BatchNorm": [],
                 "Dense": [],
                 "Conv1D": [],
                 "QuantizedDense": []}
    for i in range(n_weights):
      for pos in range(len(self._model.layers)):
        if self._model.layers[pos].name == names[i]:
          break
      layer_type = self._model.layers[pos].__class__.__name__
      if layer_type == "BatchNormalization":
          positions["BatchNorm"].append(pos)
      elif layer_type == "Dense":
          positions["Dense"].append(pos)
      elif layer_type == "Conv1D":
          positions["Conv1D"].append(pos)
      elif layer_type == "QuantizedDense":
          positions["QuantizedDense"].append(pos)

    if len(positions["BatchNorm"]) != 0:
      self._batch_norm(positions["BatchNorm"])
      executed = True
    if len(positions["Dense"]) != 0:
      self._dense(positions["Dense"])
      executed = True
    if len(positions["Conv1D"]) != 0:
      self._conv1d(positions["Conv1D"])
      executed = True
    if len(positions["QuantizedDense"]) != 0:
      self._dense(positions["QuantizedDense"])
      executed = True
    if executed == False:
      print("Could not plot anything.\nTry one of these layers to plot: \n",
            self._get_layer_positions())

  def feature_maps(self, x, layer_name,
                 lower_boundary = 4000, upper_boundary = 400):
    """
    Plots feature maps for given input x.

    Parameters
    ----------
    layer_name : str
        Defines name of layer.

    """
    plt.figure("Masked Input")
    test_model = self.container(inputs = self._model.inputs,
                                      outputs = self._model.get_layer(layer_name).output)
    sample_points = test_model.layers[0].input_shape[1]
    x_ = np.linspace(lower_boundary, upper_boundary, sample_points)
    # except:
    #     raise NameError(f"Layer '{layer_name}' is not a part of self.container '{self.container}'")
    feature_maps = self.container.test_model(x, model = test_model)

    n_feature_maps = len(feature_maps[0])

    # if kwargs["n_hor_plots"] is None:
    #     n_vert_plots = kwargs["n_vert_plots"]
    #     n_hor_plots = int(np.ceil(n_feature_maps / n_vert_plots))
    # else:
    #     n_hor_plots = kwargs["n_hor_plots"]
    #     n_vert_plots = int(np.ceil(n_feature_maps / n_hor_plots))
    # if len(feature_maps) >= 3:
    #     i = 1
    #     for _ in range(n_vert_plots):
    #         for _ in range(n_hor_plots):
    #             try:
    #                 ax = plt.subplot(n_vert_plots, n_hor_plots, i)
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])
    #                 plt.plot(feature_maps[0, ..., i - 1])
    #             except:
    #                 pass
    #             i += 1
    # else:
    #     plt.plot(feature_maps[0])
    plt.plot(x_, x[0])
    plt.plot(x_, feature_maps[0])
    plt.xlabel("Wavenumbers [cm$^-1$]")
    # plt.ylabel("Rel. Scattering Intensity")
    plt.xlim(x_[0], x_[-1])
    plt.grid(True)
    return feature_maps

  # analyzers
  def predict(self, x, y=None, model=None):
    if not self._trained:
      return
    y_pred, y_true = self.container.test_model(x, y, model=model)

    fig = plt.figure('Prediction', figsize=(half_width, 2))
    ax = fig.add_subplot(111)
    classes = np.arange(len(y_pred))
    ax.bar(classes, y_pred, color=self.default.c_cmap(12),
           label='prediction')
    ax.bar(classes, y_true, edgecolor=self.default.c_cmap(16),
           color=None, label='ground truth')
    ax.set_xticks(classes)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_xlabel('classes')
    ax.set_ylabel('probability $[\%]$')

  def predict_set(self, X, y,
                 metric='accuracy',
                 model=None):
    if not self._trained:
      return
    y_pred, y_true = self.container.test_model(X, y, model=model)

    METRIC = {'accuracy': self._accuracy,
              'recall': self._recall,
              'precision': self._precision,
              'all': self._all}

    if metric != 'all':
      fig = plt.fig('Prediction', figsize=(half_width, 2))
      ax = fig.add_subplot(111)
    else:
      fig, ax = plt.subplots(3, 1, num='Prediction', figsize=(half_width, 6),
                             sharex=True)

    METRIC[metric](ax, y_true, y_pred)

    return fig

  def _all(self, ax, y_true, y_pred):
    axs = ax
    metrics = (self._accuracy, self._recall, self._precision)
    colors = (self.default.c_cmap(12),
              self.default.c_cmap(13),
              self.default.c_cmap(14))
    for i, ax in enumerate(axs):
      metrics[i](ax, y_true, y_pred, colors[i])
    ax.set_xlabel('class')

  def _accuracy(self, ax, y_true, y_pred, color=None):
    if not color:
      color = self.default.c_cmap(12)
    c_pred = self.container.data.one_hot_labels(np.argmax(y_pred, axis=1))
    c_true = self.container.data.one_hot_labels(np.argmax(y_true, axis=1))
    acc_l = []
    classes = np.arange(len(self.container.data.classes))
    for i in range(c_true.shape[1]):
      pred = c_pred[:,i]
      true = c_true[:,i]
      diff = pred-true
      TPnTN = len(np.argwhere(diff==0))
      TPnTNnFPnFN = len(pred)
      acc = TPnTN / TPnTNnFPnFN
      acc_l.append(acc)
      ax.text(i, acc*100/2, s=np.round(acc*100,1),
              ha='center', va='center', color='w')
    acc_l = np.array(acc_l)
    ax.bar(classes, acc_l*100,
           color=color)
    ax.set_xticks(classes)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_ylabel('accuracy [\%]')
    return ax

  def _recall(self, ax, y_true, y_pred, color=None):
    if not color:
      color = self.default.c_cmap(12)
    c_pred = self.container.data.one_hot_labels(np.argmax(y_pred, axis=1))
    c_true = self.container.data.one_hot_labels(np.argmax(y_true, axis=1))
    recall_l = []
    classes = np.arange(len(self.container.data.classes))
    for i in range(c_true.shape[1]):
      pred = c_pred[:,i]
      true = c_true[:,i]
      idx = np.argwhere(true==1)
      diff = pred[idx]-true[idx]
      TP = len(np.argwhere(diff==0))
      TPnFN = len(idx)
      recall = TP / TPnFN
      recall_l.append(recall)
      ax.text(i, recall*100/2, s=np.round(recall*100,1),
              ha='center', va='center', color='w')
    recall_l = np.array(recall_l)
    ax.bar(classes,
           recall_l*100,
           color=color)
    ax.set_xticks(classes)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_ylabel('recall [\%]')

    return ax

  def _precision(self, ax, y_true, y_pred, color=None):
    if not color:
      color = self.default.c_cmap(12)
    c_pred = self.container.data.one_hot_labels(np.argmax(y_pred, axis=1))
    c_true = self.container.data.one_hot_labels(np.argmax(y_true, axis=1))
    classes = np.arange(len(self.container.data.classes))
    precision_l = []
    for i in range(c_true.shape[1]):
      pred = c_pred[:,i]
      idx = np.argwhere(pred==1)
      true = c_true[:,i]
      diff = pred[idx]-true[idx]
      TP = len(np.argwhere(diff==0))
      TPnFP = len(idx)
      precision = TP / TPnFP
      precision_l.append(precision)
      ax.text(i, precision*100/2, s=np.round(precision*100,1),
              ha='center', va='center', color='w')
    precision_l = np.array(precision_l)
    ax.bar(classes, precision_l*100,
           color=color)
    ax.set_xticks(classes)
    ax.set_xticklabels(self.container.data.classes)
    ax.set_ylabel('precision [\%]')

    return ax

  def confusion_matrix(self, X, y):
    if not self._trained:
      return
    y_pred, y_true = self.container.test_model(X, y)
    # sk-learn only accepts categorical classes:
    y_pred = np.argmax(y_pred, axis=1)
    y_true = self.container.data.categorical_labels(y_true)

    cm = confusion_matrix(y_true, y_pred, normalize = "true")

    classes = self.container.data.classes
    fig = plt.figure("Confusion Matrix", figsize=(2,2))
    ax = fig.add_subplot()
    ax.imshow(cm, cmap=self.default.im_cmap)

    ax.set_xticks(range(len(classes)), classes,
                  rotation = 45, horizontalalignment = "right")
    ax.set_xlabel('prediction $y$')
    ax.xaxis.set_label_position('top')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_ylabel('ground truth $\hat{y}$')
    ax.yaxis.set_label_position('right')

  def output(self, x, layer = None):
    x = self.container.data.X_train
    if not sample:
      sample = np.random.randint(0, len(x))

    plt.figure(f"Predicted Output {sample}")
    array = np.array(x[sample], ndmin = 2)
    feature_map = self._model.predict(array)
    pred = feature_map[0]
    try:
      plt.plot([0, self.container.y_train.shape[1]],
               [self.container.threshold,
                self.container.threshold])
    except:
      pass
    plt.plot(pred)
    plt.grid(True)
    plt.show()

  def feature_history(self):
    if not self._pruned:
      return
    log = self.container.callback.log
    trigger = self.container.callback.trigger
    params = self.container.callback.params

    colors = [self.default.c_cmap(0),
              self.default.c_cmap(4),
              self.default.c_cmap(8),
              self.default.c_cmap(16)]

    if params.metric in ['loss', 'val_loss']:
      fig = plt.figure('Pruning History', figsize=(half_width, 4))
      plt.clf()
      ax1 = fig.add_subplot(211)
      ax1.plot(trigger.value_l, color=colors[1], label='loss')
      ax1.tick_params(labelbottom=False)

      ax2 = fig.add_subplot(212, sharex=ax1)
      ax2.plot(trigger.gradient_l, color=colors[2], label='gradient')
      ax2.set_ylabel('gradient [-]')
      ax2.set_xlabel('epoch [-]')
      ax2.plot(trigger.thresh_l, color=colors[3], label='threshold')
      ax2.legend(loc='lower right')
      for ax in [ax1, ax2]:
        for i in log.pruning_epochs:
          ax.axvline(i, c='k', lw=0.5)

    else:
      fig = plt.figure('Pruning History', figsize=(half_width, 2))
      plt.clf()
      ax1 = fig.add_subplot(111)
      ax1.plot(trigger.value_l, color=colors[0], label='accuracy')
      ax1.plot(trigger.thresh_l, color=colors[3], label='threshold')
      ax1.legend(loc='lower right')
      ax1.set_xlabel('epoch [-]')
      for i in log.pruning_epochs:
        ax1.axvline(i, c='k', lw=0.5)

    x = np.arange(len(trigger.value_l))
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylabel(f'{params.metric} [-]')

  def feature_entropy(self, pruning_step=None):
    if not self._pruned:
      return

    log = self.container.callback.log

    if pruning_step:
      pruning_step = np.array([pruning_step])
    else:
      pruning_step = np.arange(len(log.loss_f))

    for i in pruning_step:
      mask_k = log.mask_k[i+1]
      mask_p = log.mask_p[i+1]

      colors = [self.default.c_cmap(0), self.default.c_cmap(19)]

      fig = plt.figure(f'Feature Evaluation Losses at {i+1}. Prune '
                       f'(Epoch: {log.pruning_epochs[i+1]})',
                       figsize=(full_width, 5))
      plt.clf()
      ax1 = fig.add_subplot(211)
      ax1.boxplot(log.loss_f[i].T, labels=np.arange(self.container.n_in))
      ax1.set_ylabel('normalized cross-entropy [-]')
      ax1.set_xlim(0.5, self.container.n_in+0.5)
      ax1.tick_params(labelbottom=False)

      ax2 = fig.add_subplot(212)
      ax2.bar(np.argwhere(mask_k).squeeze(),
              np.average(log.loss_f[i][mask_k], axis=1),
              color=colors[0], label='keep')
      ax2.bar(np.argwhere(mask_p).squeeze(),
              np.average(log.loss_f[i][mask_p],axis=1),
              color=colors[1], label='prune')
      ax2.set_ylabel('average normalized cross-entropy [-]')
      ax2.set_xlabel('feature')
      ax2.set_xlim(-0.5, self.container.n_in-0.5)
      ax2.legend(loc='upper right')