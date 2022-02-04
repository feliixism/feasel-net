# TODOs:
# - Implementation of the ability for using callback in conv layers

import numpy as np
from tensorflow.keras.callbacks import Callback # inherits from keras callbacks

# callback information variables in different containers:
from ..parameters import CallbackParams # parameter container
from ..data import preprocess as prep
# log and trigger container
from .utils.callback import CallbackLog, CallbackTrigger

class FeatureSelection(Callback):
  def __init__(self,
               evaluation_data,
               layer_name,
               n_features=None,
               callback=None):
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
    self.layer_name = layer_name
    self.n_features = n_features

    self.set_params(params=callback)

    self.evaluation_data = self.get_evaluation_subset(evaluation_data,
                                                      self.params.n_samples)

    self.I = self.get_identity(self.params.loocv)

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

    if not hasattr(self, 'log'): # checks whether it is initiallized or not
        self.initialize(logs)

    # updates the trigger parameters that decide whether it should be pruned:
    self.trigger.update(logs, self.log.n_features[-1])

    if self.trigger._prune:
      self._update_weights(epoch, logs)

    if self.trigger._stop:
      self._stop_model(epoch, logs)

  def set_params(self, params):
    """
    Overwrites the original set_params() in Keras' Callback class and sets
    a parameter class for this specific callback, where all the necessary
    parameters are saved in the class object of type CallbackParams.

    Parameters
    ----------
    params : dict
      The callback dictionary with all the paramter information for the
      instantiation of CallbackParams class.

    Returns
    -------
    None.

    """
    if params is None:
      self.params = CallbackParams()

    elif 'callback' in params.keys():
      self.params = CallbackParams(n_features=self.n_features,
                                   **params['callback'])

    else:
      return

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
    self.log = CallbackLog(self.params, self.model, self.layer_name)
    self.log.initialize(logs)

    self.trigger = CallbackTrigger(self.params, self.model, self.layer_name)
    self.trigger.initialize(logs)

    self.n_in = self.log.n_features[0]

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

  def get_evaluation_subset(self, evaluation_data, n_samples):
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
    n_samples : int
      The evaluation subset label array.

    """
    data, labels = evaluation_data

    if n_samples:
      classes = self.get_n_classes(labels)
      indices = []

      for c in range(classes):
        indices.append(np.argwhere(labels[:, c]==1).flatten()[:n_samples])

      indices = np.array(indices).flatten()

      data, labels = data[indices], labels[indices]

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
    funcs = {'cross_entropy': labels.shape[-1],
             'sparse_categorical_crossentropy': len(np.unique(labels))}

    classes = funcs[f'{self.params.loss}']

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
    if self.params.pruning_type == "exponential":
        n_features = int(self.n_in * (1 - self.params.pruning_rate)
                         ** len(self.log.pruning_epochs))

        # if the same int is calculated twice, it will be subtracted by one
        if n_features == self.log.n_features[-1]:
            n_features -= 1

    elif self.params.pruning_type == "linear":
        n_features = int(self.n_in - (self.params.n_prune
                                      * len(self.log.pruning_epochs)))

    else:
        raise NameError(f"'{self.trigger_params.pruning_type}' is not "
                        "valid as 'pruning_type'. Try 'exponential' or "
                        "'linear' instead.")

    # securing minimum number of features
    if n_features <= self.n_features:
        n_features = self.n_features

    return n_features

  def get_callback_keys(self):
    """
    Returns a list of possible callback parameter keywords.

    Returns
    -------
    keys : list
      List of possible callback parameter keywords.

    """
    keys = self.params.__dict__.keys()
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
    I = self.log.mask_k[-1] * self.I
    idx = np.argwhere(self.log.mask_k[-1]==True)
    for i, j in enumerate(idx):
      if i == 0:
        evaluation_data = I[j] * data
        evaluation_labels = labels_one_hot
      else:
        evaluation_data = np.append(evaluation_data, I[j] * data, axis=0)
        evaluation_labels = np.append(evaluation_labels,
                                      labels_one_hot, axis=0)

    return evaluation_data, evaluation_labels

  #check function: stop model?
  def _stop_model(self, epoch, logs):
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
      weights, loss = self._prune_weights(epoch)
      self.log.update(epoch, loss)
      self.trigger.converged = True
      self.model.stop_training = True
      print(f"Epoch {epoch} - Successful Feature Selection:\n"
            f"Stopped training with '{self.params.metric}' of "
            f"{logs[self.params.metric]} using "
            f"{self.log.n_features[-1]} nodes as input features.")

    elif self.trigger._epoch_max:
      weights, loss = self._prune_weights(epoch)
      self.log.update(epoch, loss)
      self.model.stop_training = True
      print(f"Epoch {epoch} - Non Convergent:\n"
            "The optimizer did not converge. Please adjust the feature "
            "selection and/or model parameters and try again.\n\nStopped "
            f"training with '{self.params.metric}' of "
            f"{logs[self.params.metric]} using "
            f"{self.log.n_features[-1]} nodes as input features.")

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

    layer = self.model.get_layer(self.layer_name)

    weights, loss = self._prune_weights(epoch) # pruned weights

    if np.array_equal(weights, self.log.weights[-1]):
      return None

    layer.set_weights([weights]) # update of the weights in 'Linear' layer
    self.log.update(epoch, loss)

    print(f"Epoch {epoch} - Weight Update:\n"
          f"Pruned {int(len(weights) - np.sum(weights))} feature(s). "
          f"Left with {self.log.n_features[-1]} feature(s).\n")



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
    # CONVENIENCE FACTOR: not used
    # the convenience factor is an adaptive factor that shall encourage
    # the feature selection callback to trigger after the second stage
    # pruning trigger is not pulled in the first few attempts
    if self.trigger.d >= self.params.d_min:
        convenience_factor = self.params.d_min / self.trigger.d
    else:
        convenience_factor = 1.0

    if n_features == self.log.n_features[-1]:
      index = self.log.index_k[-1]

    elif self.params.loocv: # highest H_features shall endure
      prune_index = np.argsort(H_features)[:-n_features]
      # prune = H_features[prune_index]
      keep_index = np.argsort(H_features)[-n_features:]
      # keep = H_features[keep_index]

      # prune_min, prune_max = np.amin(prune), np.amax(prune)
      # keep_min, keep_max = np.amin(keep), np.amax(keep)

      # # the difference between kept and pruned features shall be at
      # # least 10 % of the maximum loss value
      # # if (keep_min - prune_max) >= (self.params.d_loss_ratio
      # #                               * convenience_factor
      # #                               * (keep_max - prune_min)):
      # # index = self.log.index_k[-1][keep_index]
      mask = np.array(self.log.mask_k[0])
      mask[prune_index] = False
      # else:
      #   index = self.log.index_k[-1]


    else: # lowest H_features shall endure
      prune_index = np.argsort(H_features)[n_features:]
      # prune = H_features[prune_index]
      keep_index = np.argsort(H_features)[:n_features]
      # keep = H_features[keep_index]

      # prune_min, prune_max = np.amin(prune), np.amax(prune)
      # keep_min, keep_max = np.amin(keep), np.amax(keep)
      # # the difference between kept and pruned features shall be at
      # # least 10 % of the minimum loss value
      # # if (keep_max - prune_min) <= (self.params.d_loss_ratio
      # #                               * convenience_factor
      # #                               * (keep_min - prune_max)):
      # # index = self.log.index_k[-1][keep_index]
      mask = np.array(self.log.mask_k[0])
      mask[prune_index] = False
      # else:
      #   index = self.log.index_k[-1]

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
    loss = self.params._calculate_loss(P, Q)
    return loss

  def _get_information_richness(self, metric='mean', scale=True):
    """
    We use the cross-entropy H that measures uncertainty of possible outcomes
    and is the best suited loss metric for multiclass classification tasks.
    It has several calculation steps, where the entropy for the whole
    evaluation with multiple replications H_s is divided into the corresponding
    features H_f_s. Those feature entropies are then averaged by using the
    metric in 'metric'.

    Parameters
    ----------
    metric : str, optional
      The type of metric to combine all individual loss values to one
      meaningful and comparable value for each feature. Possible options are
      'median' or 'mean'. The default is 'mean'.
    scale : bool, optional
      If True, the feature losses are scaled that they fit in between 0 and 1.
      The default is True.

    Returns
    -------
    H_f : ndarray
      The summarized cross-entropy for each feature.

    """

    # get prediction data for the measure
    data, labels = self.map_evaluation_data()
    n_f = int(self.log.n_features[-1])

    P = labels
    P_o = self.evaluation_data[1] # unmasked datset labels in one-hot encoding

    # prediction of mapped data:
    Q = self.model.predict(data)

    # prediction of unmapped evaluation data (for comparison purposes):
    Q_o = self.model.predict(self.evaluation_data[0])

    # prevents zero division:
    Q, Q_o = np.where(Q > 1e-8, Q, 1e-8), np.where(Q_o > 1e-8, Q_o, 1e-8)

    # for a comparison - unmasked values:
    H_o = self.calculate_loss(P_o, Q_o)

    # loss for all samples s:
    H_s = self.calculate_loss(P, Q)

    # splits cross-entropy array into feature batches f:
    H_f_s = np.array(np.array_split(H_s, n_f))

    # # loss values severly depend on samples --> scaling features:
    if scale:
      H_f_s = prep.min_max(H_f_s, axis=0)

    # log the loss values for each feature and sample:
    if self.log.loss_f:
      H_f_s_ = np.array(self.log.loss_f[-1])
      H_f_s_[self.log.mask_k[-1]] = H_f_s
      H_f_s = H_f_s_
      self.log.loss_f.append(H_f_s)
      self.log.loss_o.append(H_o)

    else:
      self.log.loss_f = [H_f_s]
      self.log.loss_o = [H_o]

    # calculates mean of all entropies as loss
    METRIC = {'mean': np.mean,
              'median': np.median}

    H_f = METRIC[metric](H_f_s, axis = 1)

    return H_f

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

    weights = np.zeros(self.log.weights[0].shape)
    weights[index] = 1

    return weights, H_features