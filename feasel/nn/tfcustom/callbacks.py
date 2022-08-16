# TODOs:
# - Implementation of the ability for using callback in conv layers

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback # inherits from keras callbacks
from tensorflow.keras.utils import Progbar

# callback information variables in different containers:
from ...parameters import callback # parameter container
from ...data import preprocess as prep
from ...data import normalize
from ...utils import time

# log and trigger container:
from .utils.callback import CallbackLog, CallbackTrigger

class FeatureSelection(Callback):
  def __init__(self, evaluation_data, layer_name, **kwargs):
    """
    This class is a customized callback function to iteratively delete all the
    unnecassary input nodes (n_in) of a 'LinearPass' layer. The number of nodes
    is reduced successively with respect to the optimum of the given metric.

    The pruning or killing process is triggered whenever the threshold is
    surpassed for a given interval (e.g. 90 % accuracy for at least 15 epochs).

    The pruning function itself is an exponential or linear approximation
    towards the desired number of leftover nodes (n_features):

    Exponential:
      n_features = n_in * (1-r_p)**n_i

    Linear:
      n_features = n_in - n_p*n_i

    At each step there is a fixed percentage r_p or number n_p of nodes with
    irrelevant data that is to be deleted.

    Parameters
    ----------
    evaluation_data : tuple
      A tuple with evaluation data and evaluation labels.
    layer_name : str
      Name of layer with class type 'LinearPass'. Specifies the layer where
      irrelevant nodes are being deleted.
    n_features : float, optional
      Number of leftover optimized relevant nodes. If 'None', the node killing
      lasts until threshold cannot be surpassed anymore. The default is None.
    callback : dict
      A dictionary with all the feature selection callback parameters. Use
      get_callback_keys() method to get an overview on the valid keywords.
      The default is None.

    Returns
    -------
    None.

    """
    super().__init__()
    self._get_params(**kwargs)
    self.evaluation_data = self.get_evaluation_subset(evaluation_data)
    self.layer = layer_name
    self.I = self.get_identity(self._params.loocv)
    self.log = None

  def __repr__(self):
    return 'Feature Selection Callback'

  def on_epoch_end(self, epoch, logs={}):
    """
    Actual tensorflow callback function that is called after every epoch by
    default. It is expanded by several methods, parameters and log functions to
    suit the needs of a feature selection algorithm.

    The procedure is a s follows:

      1. Initialize:
        Store variables in self.logs (log for all iterations of the callback)
        and self.trigger (updates of each trigger variable after each epoch).

      2. Update of trigger and stopping criteria:
        Decision whether to pause training for an evaluation of feature
        importance (step 3) AND decision whether to stop training and feature
        selection or not.

      3. Update of weights:
        If callback is not triggered, the algorithm will not update anything.
        If it is triggered, it chooses the most uninformative features to be
        pruned.

    Parameters
    ----------
    epoch : int
      The current training epoch.
    logs : dict, optional
      A dictionary of the current epoch's metric. The default is {}.

    Returns
    -------
    None.

    """

    if not self.log: # checks whether it is initiallized or not
        self.initialize(logs)

    # updates the trigger parameters that decide whether it should be pruned:
    self.trigger.update(logs, self.log.f_n[-1])

    if self.trigger._prune:
      t0 = time.time_ns()
      self._update_weights(epoch, logs)
      t1 = time.time_ns()
      self.log.t.append(t1 - t0)

    if self.trigger._stop:
      self.early_stopping_criterion(epoch, logs)

  def _get_params(self, **kwargs):
    """
    Automatically checks the kwargs for valid parameters for the callback
    parameter object and updates them.

    Parameters
    ----------
    **kwargs : kwargs
      The parameter keywords.

    Raises
    ------
    KeyError
      If kwarg cannot be assigned to any parameter object.

    Returns
    -------
    None.

    """
    # parameter container with train, build and data parameters
    self._params = callback.NN()

    for key in kwargs:
      containers = {'callback': self._params}

      # updates all values that are summarized in an extra container:
      if key in containers:
        for sub_key in kwargs[key]:
          if sub_key in containers[key].__dict__.keys():
            containers[key].update(sub_key)(kwargs[key][sub_key])

      # updates keys if they are not defined in an extra container:
      elif key in self._params.__dict__.keys():
        self._params.update(key)(kwargs[key])

      else:
        raise KeyError(f"'{key}' is not a valid key for the feature selection "
                       "callback instantiation. Use show_params() to get an "
                       "overview about valid keys.")

  def show_params(self, type=None):
    """
    Overview of all possible parameters for the instantiation of the 'Base'
    class.

    Parameters
    ----------
    type : str, optional
      Specifies the parameter group that is inspected. 'Build', 'data'
      and 'train' are the possible options. If None, all parameter
      groups are shown. The default is None.

    Raises
    ------
    NameError
      If the specified 'type' is not one of the aforementioned groups.

    Returns
    -------
    params : dict
      A dictionary of valid parameters.

    """
    if type is None:
      return self._params
    else:
      try:
        return self._params[f"{type}"]
      except:
        raise NameError(f"'{type}' is an invalid argument for 'type'."
                        " Try 'train', 'build' or 'data' instead.")

  def initialize(self, logs):
    """
    Instantiates the remaining callback containers: Log and Trigger.

    Parameters
    ----------
    logs : dict
      Contains information on the training results (accuracy and loss values).

    Returns
    -------
    None.

    """
    self.log = CallbackLog(self.model, self.layer, self._params)
    self.log.initialize(logs)

    self.trigger = CallbackTrigger(self.model, self.layer, self._params)
    self.trigger.initialize(logs)

    self.n_in = self.log.f_n[0]

    # embed FeaSel parameters into keras callback params:
    self.params['FeaSel'] = self._params

  # internal functions to generate the evaluation data set for the
  # feature evaluation task
  def get_identity(self, inv):
    """
    Provides an identity matrix or bitwise inverted identity matrix,
    respectively. The matrix also considers the already pruned nodes. The
    inverted identity matrix corresponds to the LOOCV approach and is just
    masking one node at a time. The uninverted matrix behaves like a mask that
    only allows one node at each evaluation step at all.

    Parameters
    ----------
    inv : bool
      Defines whether the identity matrix has to be inverted bitwise. If True,
      it is inverted.

    Returns
    -------
    I : ndarray
        The resulting identity matrix.

    """
    I = np.identity(self.evaluation_data[0].shape[1])

    if inv: # bitwise inversion
      I = np.ones(I.shape) - I

    return I

  def get_evaluation_subset(self, evaluation_data):
    """
    Extracts an arbitrary subset from the evaluation data as an input for
    the feature selection algorithm.

    Parameters
    ----------
    n_samples : int, optional
      The number of samples per class that are evaluated by the feature
      selection algorithm.

    Returns
    -------
    evaluation_data : tuple
      The evaluation data array and the corresponding labels.

    """
    data, labels = evaluation_data

    n_samples = self._params.n_samples

    if isinstance(n_samples, type(None)):
      self.data_tf = tf.convert_to_tensor(data)
      return (data, labels)

    elif (n_samples > 0):
      classes = self.get_n_classes(labels)
      indices = []

      for c in range(classes):
        indices.append(np.argwhere(labels[:, c] == 1)[:n_samples, 0])

      indices = np.array(indices).flatten()

      data, labels = data[indices], labels[indices]

    else:
      self._params.set_loocv(False)

    return (data, labels)

  def get_n_classes(self, labels):
    """
    Returns number of unique classes given in an array.

    Parameters
    ----------
    labels : ndarray
      The array with labels.

    Returns
    -------
    classes : int
      Number of classes.

    """
    funcs = {'cross-entropy': labels.shape[-1],
             'sparse categorical-crossentropy': len(np.unique(labels))}

    classes = funcs[f'{self._params.eval_metric}']

    return classes

  def _get_n_features(self):
    """
    Calculates the number of nodes or features n_features at each iteration i.
    The applied function depends on the pruning type and its specific
    parameters (rate r_p, number n_p), the number of weights or initial number
    of features n_in and the number of times n_i that the features already have
    been pruned.

    Exponential:
      n_features = n_in * (1-r_p)**n_i

    Linear:
      n_features = n_in - n_p*n_i

    Returns
    -------
    n_features : int
      Number of features that are still active after the pruning.

    Raises
    ------
    NameError
      Raised if the FS pruning type is not valid and there is no corresponding
      function for it.

    """
    if self._params.pruning_type == "exponential":
        n_features = int(self.n_in * (1 - self._params.pruning_rate)
                         ** len(self.log.e_prune))

        # if the same int is calculated twice, it will be subtracted by one
        if n_features == self.log.f_n[-1]:
            n_features -= 1

    elif self._params.pruning_type == "linear":
        n_features = int(self.n_in - (self._params.n_prune
                                      * len(self.log.pruning_epochs)))

    else:
        raise NameError(f"'{self.trigger_params.pruning_type}' is not "
                        "valid as 'pruning_type'. Try 'exponential' or "
                        "'linear' instead.")

    # securing minimum number of features
    if n_features <= self._params.n_features:
        n_features = self._params.n_features

    return n_features

  def get_callback_keys(self):
    """
    Returns a list of possible callback parameter keywords.

    Returns
    -------
    keys : list
      List of possible callback parameter keywords.

    """
    keys = self._params.__dict__.keys()
    return keys

  def map_evaluation_data(self):
    """
    Maps the evaluation data into a one-hot encoded dataset with the desired
    amount of evaluation samples for the feature selection process (specified
    by n_samples). The already pruned features are being filtered during the
    mapping.

    Returns
    -------
    evaluation_data : ndarray
      The evaluation array for the feature selection process.
    evaluation_labels : ndarray
      The corresponding label array.

    """
    data, labels = self.evaluation_data
    n_samples, n_in = data.shape

    # one-hot-encoding function:
    labels_one_hot = prep.one_hot(labels)

    # masked loocv array with masking indices from previous pruning epoch:
    I = self.log.m_k[-1] * self.I
    idx = np.argwhere(self.log.m_k[-1]==True)

    if self._params.accelerate:
      idx = self.accelerate(idx)

    H = np.zeros(self.evaluation_data[0].shape).T
    if self._params.n_samples != 0:
      print(f'Evaluation of {len(idx)} features:')
      pbar = Progbar(target=len(idx), width=30, unit_name='Evaluated samples')
      for i, j in enumerate(idx):
        pbar.update(i + 1)
        evaluation_data = np.array(data)
        evaluation_data[:, j] = 0
        evaluation_labels = labels_one_hot
        H[j] = self.evaluate_feature(evaluation_data, evaluation_labels)

    else:
      for i, j in enumerate(idx):
        P = np.array(I[j], ndmin=2)
        Q = None
        H[j] = self.calculate_loss(P, Q)

    return H

  def accelerate(self, idx, n_max=1000):
    if len(idx) <= n_max:
      idx_new = idx

    else:
      t = int(np.ceil(len(idx) / n_max))
      idx_new = []
      i = 0

      while i < len(idx):
        idx_new.append(idx[i:i+t].squeeze())
        i += t

      if (len(idx) - i) > 0:
        idx_new.append(idx[i:-1].squeeze())

    return idx_new

  def evaluate_feature(self, data, labels):
    P = labels
    Q = self.model.predict(data, batch_size=len(P), verbose=False)
    Q = np.where(Q > 1e-10, Q, 1e-10)

    H = np.array(self.calculate_loss(P, Q), ndmin=2)

    return H

  def merge_losses(self, H):
    """
    Merges current and previous loss values according to the previous mask.

    Returns
    -------
    None.

    """
    if self.log.f_loss:
      H_previous = np.array(self.log.f_loss[-1])

      H_merged = H_previous
      H_merged[self.log.m_k[-1]] = H[self.log.m_k[-1]]
      self.log.f_loss.append(H_merged)

    else:
      H_merged = H
      self.log.f_loss = [H_merged]

    return H_merged

  #check function: stop model?
  def early_stopping_criterion(self, epoch, logs):
    """
    Checks whether stopping conditions are met.

    Parameters
    ----------
    epoch : int
      Current training epoch.
    logs : dict
      Contains information on the training results (accuracy and loss values).

    Returns
    -------
    None.

    """

    if self.trigger._epoch_max and self.trigger._success:
      self.log.update(epoch, self.log.f_eval[-1])
      self.trigger._converged = True
      self.model.stop_training = True
      print(f"Epoch {epoch} - Successful Feature Selection:\n"
            f"Stopped training with '{self._params.eval_type}' of "
            f"{logs[self._params.eval_type]} using "
            f"{int(self.log.f_n[-1])} features as input.")

    elif self.trigger._epoch_max:
      self.log.update(epoch, self.log.f_eval[-1])
      self.model.stop_training = True
      print(f"Epoch {epoch} - Non Convergent:\n"
            "The optimizer did not converge. Please adjust the feature "
            "selection and/or model parameters and try again.\n\nStopped "
            f"training with '{self._params.eval_type}' of "
            f"{logs[self._params.eval_type]} using "
            f"{int(self.log.f_n[-1])} features as input.")

  #update functions: weights and stopping criteria
  def _update_weights(self, epoch, logs):
    """
    Updates the weights in the 'LinearPass' layer.

    Parameters
    ----------
    epoch : int
      Current training epoch.
    logs : dict
      Contains information on the training results (accuracy and loss values).

    Returns
    -------
    None.

    """

    layer = self.model.get_layer(self.layer)

    weights, loss = self._prune_weights(epoch) # pruned weights

    if np.array_equal(weights, self.log.m[-1]):
      return None

    layer.set_weights([weights]) # update of the weights in 'Linear' layer
    self.log.update(epoch, loss)

    if self._params.reset_weights:
      for l in self.model.layers:
        if l.name not in ['Linear', 'Input']:
          old = l.get_weights()
          l.set_weights([l.kernel_initializer(shape=old[0].shape),
                         l.bias_initializer(shape=old[1].shape)])

    print(f"Epoch {epoch} - Weight Update:\n"
          f"Pruned {int(len(weights) - np.sum(weights))} feature(s). "
          f"Left with {int(self.log.f_n[-1])} feature(s).\n")



  def _get_indices(self, H_features, n_features):
    """
    Get the features with the most important information.

    Parameters
    ----------
    H_features : ndarray
        Loss matrix of the features.
    n_features : int
        Number of features that are to be kept.

    Returns
    -------
    index : ndarray
        Indices of the most important weights.

    """

    if n_features == int(self.log.f_n[-1]):
      mask = self.log.m_k[-1]

    # highest H_features shall endure, since the masked features cause the
    # biggest increase of the entropy and are thus more important
    elif self._params.loocv:
      prune_index = np.argsort(H_features)[:-n_features]

      # previously pruned features are not released:
      if not self._params.release:
        idx_p = np.array(np.argwhere(self.log.m_p[-1]).squeeze(), ndmin=1)
        idx_k = np.argwhere(self.log.m_k[0]).squeeze()[self.log.m_k[-1]]
        idx_sort = np.argsort(H_features[self.log.m_k[-1]])[:-n_features]
        idx = np.array(idx_k[idx_sort], ndmin=1)
        prune_index = np.concatenate([idx_p, idx])
      mask = np.array(self.log.m_k[0], ndmin=1)
      mask[prune_index] = False

    # lowest H_features shall endure
    else:
      prune_index = np.argsort(H_features)[n_features:]
      mask = np.array(self.log.m_k[0])
      mask[prune_index] = False

    return mask

  def calculate_loss(self, P, Q):
    """
    Calculates loss values according to the currently used metric.

    Parameters
    ----------
    P : ndarray
      Target values.
    Q : ndarray
      Predicted values.

    Returns
    -------
    loss : ndarray
      Array with loss as its entries.

    """
    loss = self._params._calculate_loss(P, Q)
    return loss

  def _get_information_richness(self):
    """
    We use the cross-entropy H that measures uncertainty of possible outcomes
    and is the best suited loss metric for multiclass classification tasks.
    It has several calculation steps, where the entropy for the whole
    evaluation with multiple replications H_s is divided into the corresponding
    features H_f_s. Those feature entropies are then averaged by using the
    metric in 'metric'.

    Returns
    -------
    H_f : ndarray
      The summarized cross-entropy for each feature.

    """

    # get prediction data for the measure
    H = self.map_evaluation_data()

    H = self.merge_losses(H)

    if self._params.remove_outliers:
        H = self._remove_ouliers(H)

    if self._params.normalization == 'min-max':
      H = normalize.min_max(H, axis=0)

    elif self._params.normalization == 'standardize':
      H = normalize.standardize(H, axis=0)

    # applies decsion metric of all entropies as loss
    METRIC = {'average': np.mean,
              'median': np.median}

    H_f = METRIC[self._params.decision_metric](H, axis=1)

    self.log.f_eval.append(H_f)

    return H_f

  def _remove_ouliers(self, X, factor=1):
    quant3, quant1 = np.percentile(X, [95, 5], axis=1)
    iqr = quant3 - quant1
    iqr_sigma = iqr/1.34896
    med_data = np.median(X, axis=1)
    for i in range(X.shape[0]):
      X[i] = np.where(X[i] > med_data[i] - factor*iqr_sigma[i],
                      X[i], med_data[i])
      X[i] = np.where(X[i] < med_data[i] + factor*iqr_sigma[i],
                      X[i], med_data[i])
    return X

  def _prune_weights(self, epoch):
    """
    Class method to calculate new weight matrix for the 'LinearPass' layer.

    Parameters
    ----------
    weights : float
      Previously used weight matrix. Initiallized with all weights being 1.
    n_min : int
      Minimum number of nodes to be leftover. If 'None', the weights are
      updated until the threshold condition cannot be met anymore.

    Returns
    -------
    weights : ndarray
      Calculated new sparse weight matrix.
    H_features : ndarray
      The loss values for each feature.

    """
    # calculate the amount of features after the pruning iteration
    H_features = self._get_information_richness()
    n_features = self._get_n_features()

    # get indices that shall be kept
    index = self._get_indices(H_features, n_features)

    weights = np.zeros(self.log.m[0].shape)
    weights[index] = 1

    return weights, H_features