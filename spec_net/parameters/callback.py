from .base import BaseParams
from ..data.metrics import cross_entropy, entropy

# mapping for the different functionalities of the feature selection callback
_METRIC_MAP = {'accuracy': {'thresh': 0.9},
               'loss': {'thresh': -1e-3},
               'val_accuracy': {'thresh': 0.9},
               'val_loss': {'thresh': -1e-3}}

_METRIC_TYPES = {'accuracy': ['accuracy', 'acc', 'acc.'],
                 'loss': ['loss'],
                 'val_accuracy': ['val_accuracy', 'val_acc'],
                 'val_loss': ['val_loss']}

_NORM_TYPES = {'min_max': ['min_max', 'min-max', 'min max', 'min-max scale'],
               'standardize': ['standardize', 'standard',
                               'z score', 'z-score']}

_LOSS_MAP = {'cross_entropy': cross_entropy,
             'entropy': entropy}

_LOSS_TYPES = {'cross_entropy': ['cross_entropy', 'CE'],
               'entropy': ['entropy']}

_PRUNING_MAP = {'exponential': {'pruning_rate': 0.2,
                                'n_prune': None},
                'linear': {'pruning_rate': None,
                           'n_prune': 1}}

_PRUNING_TYPES = {'exponential': ['exponential', 'exp', 'exp.', 'ex', 'ex.'],
                  'linear': ['linear', 'lin', 'lin.']}

class CallbackParams(BaseParams):
  def __init__(self,
               thresh=None,
               decay=0.,
               d_min=20,
               d_max=500,
               eval_metric='accuracy',
               loss='cross_entropy',
               pruning_type='exponential',
               pruning_rate=None,
               eval_normalization=None,
               n_features=None,
               n_prune=None,
               n_samples=None,
               loss_ratio=0.1,
               loocv=True):
    """
    Parameter class for the trigger control of the leave-one-out
    cross-validation (LOOCV) based feature selection callback.

    Parameters
    ----------
    thresh : float, optional
      The threshold value that has to be surpassed in order to trigger the
      callback. If the metric 'accuracy' or 'val_accuracy' is chosen, it is set
      to 0.95 by default. The corresponding default value for the loss metrics
      is 0.1.
    decay : float
      Sets an adaptive thresholding value. If set, it decreases the threshold
      (thresh) by this value until it meets trigger conditions. The threshold
      will be set to its original state after the trigger is being pulled
      again. The deafult is None.
    d_min : int, optional
      The number of epochs in which the trigger conditions have to be met
      consistently. The default is 20.
    d_max : int, optional
      The maximum number of epochs in which the optimizer tries to meet the
      trigger conditions. If None, it will try to reach the thresholds until
      the end of the actual training process set by the number of epochs in the
      keras model.fit() method. The default is None.
    eval_metric : str, optional
      The metric that is monitored. See _METRIC_TYPES for the possible
      arguments. The type of metric determines the default arguments of
      'thresh' and 'grad'. The default is "accuracy".
    loss : str, optional
      The loss function that is exectuted for the evaluation of the features'
      importances. See _LOSS_TYPES for the possible arguments. The default is
      "cross_entropy".
    pruning_type : str, optional
      The type of node reduction for the recursive feature elimination. See
      _PRUNING_TYPES for the possible arguments. The type of pruning determines
      the default arguments of 'pruning_rate' and 'n_prune'. The default is
      "exponential".
    pruning_rate : float, optional
      The percentage of nodes being eliminated after the update method is
      triggered. The rate is only necessary if 'exponential' is chosen as
      'pruning_type' and it is set to None otherwise. The default is 0.2.
    eval_normalization : str, optional
      The evaluated losses per feature can prominently vary over samples.
      Therefore it can be normalized. The default is None.
    picking_metric : str, optional
      The metric used for choosing the relevant features. The default is
      "mean".
    n_prune : int, optional
      The absolute number of nodes that are eliminated at each iteration. The
      number is only necessary if 'linear' is chosen as 'pruning_type'. The
      default is None.
    n_samples : int, optional
      The number of validation samples that are chosen for the evaluation of
      the most important features. If None, it will not use any validation
      data, but only evaluates the weights in the model without any sample
      data. This implies a significantly faster evaluation method on the one
      hand and worse consistency in the feature selection results on the other
      hand. The default is None.
    loss_ratio : float, optional
      The minimum ratio of the difference of the first pruned and last kept
      feature and the best overall feature loss that has to be obtained in
      order to trigger pruning (2nd pruning stage). The default is 0.1.
    loocv : bool, optional
      If True, it will apply the principle of the LOOCV and mask only one
      feature per classification. If False, it will use the inverse LOOCV and
      mask every feature but one. The default is True.

    Returns
    -------
    None.

    """
    self.set_metric(eval_metric)
    self.set_loss(loss)
    self.set_pruning_type(pruning_type)
    self.set_threshold(thresh)
    self.set_d_min(d_min)
    self.set_d_max(d_max)
    self.set_pruning_rate(pruning_rate)
    self.set_n_features(n_features)
    self.set_n_prune(n_prune)
    self.set_n_samples(n_samples)
    self.set_loss_ratio(loss_ratio)
    self.set_loocv(loocv)
    self.set_decay(decay)
    self.set_normalization(eval_normalization)

    self._MAP = {'eval_metric': self.set_metric,
                 'loss': self.set_loss,
                 'pruning_type': self.set_pruning_type,
                 'thresh': self.set_threshold,
                 'd_min': self.set_d_min,
                 'd_max': self.set_d_max,
                 'pruning_rate': self.set_pruning_rate,
                 'n_features': self.set_n_features,
                 'n_prune': self.set_n_prune,
                 'n_samples': self.set_n_samples,
                 'loss_ratio': self.set_loss_ratio,
                 'loocv': self.set_loocv,
                 'decay': self.set_decay,
                 'eval_normalization': self.set_normalization}

  def __repr__(self):
    return ('Parmeter container for the Feature Selection Callback\n'
            f'{self.__dict__}')

  # SETTERS
  def set_metric(self, metric):
    self.metric = self._get_metric(metric)

  def set_loss(self, loss):
    self.loss = self._get_loss(loss)

  def set_pruning_type(self, pruning_type):
    self.pruning_type = self._get_pruning_type(pruning_type)

  def set_threshold(self, thresh):
    self.thresh = self._get_thresh(thresh)


  def set_d_min(self, d_min):
    self.d_min = d_min

  def set_d_max(self, d_max):
    self.d_max = d_max

  def set_pruning_rate(self, pruning_rate):
    self.pruning_rate = self._get_pruning_rate(pruning_rate)

  def set_n_features(self, n_features):
    self.n_features = n_features

  def set_n_prune(self, n_prune):
    self.n_prune = self._get_n_prune(n_prune)

  def set_n_samples(self, n_samples):
    self.n_samples = n_samples

  def set_loss_ratio(self, loss_ratio):
    self.loss_ratio = loss_ratio

  def set_loocv(self, loocv):
    self.loocv = loocv

  def set_decay(self, decay):
    self.decay = decay

  def set_normalization(self, normalization):
    self.normalization = normalization

  # HELPER FUNCTIONS:
  def _get_metric(self, metric):
    """
    Provides different aliases for all possible metrices and searches for
    the corresponding metric (e.g. 'acc' --> 'accuracy').

    Parameters
    ----------
    metric : str
      Metric alias.

    Raises
    ------
    NameError
      If metric is not a valid alias.

    Returns
    -------
    METRIC : str
      The proper metric used for the following operations inside the class.

    """
    for METRIC in _METRIC_TYPES:
      if metric in _METRIC_TYPES[f'{METRIC}']:
        return metric
    raise NameError(f"'{metric}' is not a valid metric.")

  def _get_normalization(self, normalization):
    """
    Provides different aliases for all possible noramlizations and searches for
    the corresponding normalization (e.g. 'min max' --> 'min_max').

    Parameters
    ----------
    normalization : str
      Normalization alias.

    Raises
    ------
    NameError
      If normalization is not a valid alias.

    Returns
    -------
    NORM : str
      The proper normalization used for the following operations inside the
      class.

    """
    for NORM in _NORM_TYPES:
      if normalization in _NORM_TYPES[f'{NORM}']:
        return NORM
    raise NameError(f"'{normalization}' is not a valid normalization.")

  def _get_loss(self, loss):
      """
      Provides different aliases for all possible losses and searches for
      the corresponding loss (e.g. 'CE' --> 'cross_entropy').

      Parameters
      ----------
      loss : str
        Loss alias.

      Raises
      ------
      NameError
        If loss is not a valid alias.

      Returns
      -------
      LOSS : str
        The proper loss used for the following operations inside the class.

      """
      for LOSS in _LOSS_TYPES:
        if loss in _LOSS_TYPES[f'{LOSS}']:
          return LOSS
      raise NameError(f"'{loss}' is not a valid loss.")

  def _get_pruning_type(self, pruning_type):
    """
    Provides different aliases for all possible pruning types and searches
    for the corresponding type (e.g. 'exp' --> 'exponential').

    Parameters
    ----------
    pruning_type : str
      Pruning type alias.

    Raises
    ------
    NameError
    If pruning type is not a valid alias.

    Returns
    -------
    PRUNING_TYPE : str
      The proper type used for the following operations inside the class.

    """
    for PRUNING_TYPE in _PRUNING_TYPES:
      if pruning_type in _PRUNING_TYPES[f'{PRUNING_TYPE}']:
        return PRUNING_TYPE
    raise NameError(f"'{pruning_type}' is not a valid pruning type.")

  def _get_thresh(self, thresh):
    """
    Looks up the metric map for the default threshold value if no other
    value is given.

    Parameters
    ----------
    thresh : float
      Threshold value for the feature selection callback.

    Returns
    -------
    thresh : float
      Threshold value for the feature selection callback.

    """
    if not thresh:
      thresh = _METRIC_MAP[f'{self.metric}']['thresh']
    return thresh

  def _get_pruning_rate(self, pruning_rate):
    """
    Looks up the pruning map for the default pruning rate value if no other
    value is given.

    Parameters
    ----------
    pruning_rate : float
      Pruning rate value for the feature selection callback.

    Returns
    -------
    pruning_rate : float
      Pruning rate value for the feature selection callback.

    """
    if not pruning_rate:
      pruning_rate = _PRUNING_MAP[f'{self.pruning_type}']['pruning_rate']
    return pruning_rate

  def _get_n_prune(self, n_prune):
    """
    Looks up the pruning map for the default pruning number if no other
    value is given.

    Parameters
    ----------
    n_prune : float
      Pruning number for the feature selection callback.

    Returns
    -------
    n_prune : float
      Pruning number for the feature selection callback.

    """
    if not n_prune:
      n_prune = _PRUNING_MAP[f'{self.pruning_type}']['n_prune']
    return n_prune

  def _get_map(self, type):
    """
    Returns the specified map with all possible options.

    Parameters
    ----------
    type : str
      Name of the map. Possible options are 'metric', 'loss' and 'pruning'.

    Raises
    ------
    NameError
      If wrong type is given.

    Returns
    -------
    map : dict
      The specific map dict.

    """
    if type == 'metric':
      return _METRIC_MAP

    elif type == 'loss':
      return _LOSS_MAP

    elif type == 'pruning':
      return _PRUNING_MAP

    else:
      raise NameError(f"Unable to find a map for '{type}' type (valid types "
                      "are: metric, loss and pruning.")


  # probably not needed: deletion or relocation?
  def _calculate_loss(self, P, Q):
    """
    Calculates the loss between predicted Q and actual P class.

    Parameters
    ----------
    P : ndarray
      The actual data.
    Q : ndarray
      The predicted data.

    Returns
    -------
    loss : ndarray
      The losses between each data set.

    """
    loss = _LOSS_MAP[f'{self.loss}'](P,Q)
    return loss



