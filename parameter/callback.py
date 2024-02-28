"""
feasel.parameter.callback
=========================
"""

from .base import BaseParams
from ..data.metrics import cross_entropy, entropy # metrics for evaluation

class CallbackParamsNN(BaseParams):
  """
  Parameter class for the trigger control of the leave-one-out
  cross-validation (LOOCV) based feature selection callback.

  Attributes
  ----------
  n_features : int
    The number of features :math:`n_f` that the callback tries to extract. If
    the number is not specified, it will use the ``compression_rate`` to
    calculate the number of features. The default is `None`.

  compression_rate : float
    The compression rate :math:`\\rho` that is tried to obtain after the
    callback has been applied. It is calculated by the ratio of the number of
    features :math:`n_f` after the FeaSel-Net algorithm and the original number
    of features :math:`p` and is only needed, when ``n_features`` is not
    specified. The default is :math:`0.1`.

    .. math::

      \\rho = \\frac{n_f}{p}

  thresh : float
    The threshold value :math:`\\tau` that has to be surpassed in order to
    trigger the callback. If the ``eval_metric`` `accuracy` or `val_accuracy`
    is chosen, it is set to :math:`\\tau_{acc}=0.9` by default. The
    corresponding default value for the loss-based metrics is
    :math:`\\tau_g=-0.001`.

  decay : float
    The decay :math:`\\delta` provides the possibility of an adaptive
    thresholding value. If set, it decreases ``thresh`` by this value every
    epoch :math:`e` that the callback has not been triggered since the last
    prune :math:`e_p` plus the required waiting time ``d_min``. This is done
    until it meets the trigger conditions again. The threshold will be set to
    its original state after the trigger is being pulled. The default is
    `None`.

    .. math::

      \\tau' = \\tau-\\left((e-d_{min}-e_p)\\cdot \\delta\\right)

  d_min : int
    The number of epochs :math:`e` in which the trigger conditions have to be
    met consistently. The default is :math:`20`.

  d_max : int
    The maximum number of epochs :math:`e` in which the optimizer tries to meet
    the trigger conditions. If `None`, it will try to reach the thresholds
    until the end of the actual training process set by the number of epochs in
    the keras ``model.fit()`` method. The default is `None`.

  pruning_type : str
    The type of node reduction for the recursive feature elimination. The type
    of pruning determines the default arguments of ``pruning_rate`` and
    ``n_prune``. The default is `exponential`.

    Possible arguments are:
      - `exponential`: The number of features :math:`n_f` is exponentially
        decreased and pruned with increasing amount of trigger processes
        :math:`n_t` depending on an initial pruning rate :math:`\\rho`.
      - `linear`: The number of features :math:`n_f` is linearly
        decreased and pruned with increasing amount of trigger processes
        :math:`n_t` depending on an initial number of features to prune
        :math:`n_p`.

    .. list-table:: Pruning Types Aliases
      :widths: 25 50
      :header-rows: 1

      * - Pruning Types
        - Aliases
      * - `exponential`
        - `exponential`, `exp`, `exp.`, `ex`, or `ex.`
      * - `linear`
        - `linear`, `lin`, or `lin.`

  pruning_rate : float
    The percentage of nodes :math:`\\rho` being eliminated after the update
    method is triggered. The rate is only necessary if `exponential` is chosen
    as ``pruning_type`` and it is set to `None` otherwise. The default is
    :math:`0.2`.

  n_prune : int, optional
    The absolute number of nodes that are eliminated at each iteration. The
    number is only necessary if `linear` is chosen as ``pruning_type``. The
    default is :math:`0`.

  eval_type : str
    The evaluation type that is monitored during training. Its value will
    decide whether and when the callback is executed.

    Possible arguments are:
      - `accuracy`: The `accuracy` history is monitored during training.
      - `loss`: The `loss` history, or its gradient to be precise, is monitored
        during training.
      - `val_accuracy`: The `validation accuracy` history is monitored during
        training.
      - `val_loss`: The `validation loss` history, or its gradient to be
        precise, is monitored during training.

    .. list-table:: Evaluation Types Aliases
      :widths: 25 50
      :header-rows: 1

      * - Evaluation Type
        - Aliases
      * - `accuracy`
        - `accuracy`, `acc`, or `acc.`
      * - `loss`
        - `loss`
      * - `val_accuracy`
        - `val_accuracy`, or `val_acc`
      * - `val_loss`
        - `val_loss`

  eval_metric : str
    The metric function that is exectuted for the evaluation of the feature
    importances :math:`\\mathcal{I}_f`.

    Possible arguments are:
      - `cross-entropy`: The algorithm will use the cross-entropy metric for
        the evaluation of the feature importance.
      - `entropy`: The algorithm will use the entropy metric for the evaluation
        of the feature importance.

    .. list-table:: Evaluation Metrics Aliases
      :widths: 25 50
      :header-rows: 1

      * - Evaluation Metric
        - Aliases
      * - `cross-entropy`
        - `cross-entropy`, `cross_entropy`, or `CE`
      * - `entropy`
        - `entropy`

  callback_norm : str
    The evaluated losses per feature can prominently vary over samples.
    Therefore it can be normalized. The default is `None`.

    Possible arguments are:
      - `min-max`: The algorithm will scale all features to fit into the range
        :math:`0-1`.
      - `standardize`: The algorithm will scale all features to have
        :math:`\\sigma=1` and :math:`\\mu=0`.

    .. list-table:: Evaluation Metrics Aliases
      :widths: 25 50
      :header-rows: 1

      * - Normalization
        - Aliases
      * - `min-max`
        - `min_max`, `min-max`, `min max`, or `min-max scale`
      * - `standardize`
        - `standardize`, `standard`, `z score`, or `z-score`

  n_samples : int
    The number of evaluation samples :math:`n_s^{(e)}` that are chosen for the
    evaluation of the most important features. If `None`, it will not use any
    validation data, but only evaluates the weights in the model without any
    sample data. This implies a significantly faster evaluation method on the
    one hand and worse consistency in the feature selection results on the
    other hand. The default is `None`.

  loss_ratio : float
    The minimum ratio of the difference of the first pruned and last kept
    feature and the best overall feature loss that has to be obtained in
    order to trigger pruning (2nd pruning stage). The default is :math:`0.1`.

  decision_metric : str
    Sets the metric for the decision, which features are to be pruned. The
    possible options are `median` or `average`. The default is `average`.

    Possible arguments are:
      - `median`: The features are evaluated using the median of all losses.
      - `average`: The features are evaluated using the average of all losses.

    .. list-table:: Decision Metrics Aliases
      :widths: 25 50

      * - Decision Metrics
        - Aliases
      * - `median`
        - `median` or `med`
      * - `average`
        - `average`, `mean`, `avrg`, or `mu`

  remove_outliers : bool
    If `True`, it will remove all outliers from the evaluation data. The
    default is `False`.

  reset_weights : bool
    If `True`, the weights will be resetted and initialized anew at every
    pruning epoch such that the algorithm is unbiased from previous learning
    and pruning steps for the next pruning epoch. The default is `False`.

    The :math:`3\\times 3` weight matrix :math:`W` for a `fully connected`
    layer may look something like this after training at the first pruning
    step:

    .. math::

      W = \\begin{bmatrix}
            -0.1 & 3.9 & 0.2 \\\\
            -0.2 & 0.0 & 0.1 \\\\
            5.2 & 4.1 & 0.2
          \\end{bmatrix}

    For the next pruning it is biased towards the higher weighted entries. To
    get rid of this bias, the matrix gets initialized again:

    .. math::

      W = \\begin{bmatrix}
            -0.2 & 0.3 & -0.2 \\\\
            -0.2 & 0.2 & -0.1 \\\\
            0.2 & 0.0 & 0.1
          \\end{bmatrix}

  loocv : bool
    If `True`, it will apply the principle of the leave-one-out
    cross-validation (LOOCV) and mask only one feature per classification. If
    `False`, it will use the inverse LOOCV and mask every feature but one. The
    default is `True`.

  accelerate : bool
    If `True`, it will use a maximum number of :math:`1000` features at the
    same time. This is especially helpful for huge datasets. Features will be
    grouped with other features in their direct vicinity. The default is
    `False`.

  release : bool
    If `True`, it releases all previously pruned features from the definitive
    state of being pruned. The features are available again and if some other
    feature performs worse than previously pruned features, they will be
    deleted instead. The default is `False`.

  """
  def __init__(self,
               # monitoring parameters:
               n_features=None,
               compression_rate=None,
               thresh=None,
               decay=0.,
               d_min=20,
               d_max=500,
               # evaluation parameters:
               pruning_type='exponential',
               pruning_rate=0.2,
               n_prune=1,
               eval_type='accuracy',
               eval_metric='cross_entropy',
               callback_norm=None,
               n_samples=None,
               loss_ratio=0.1,
               decision_metric='average',
               remove_outliers=False,
               reset_weights=False,
               loocv=True,
               accelerate=False,
               release=False):
    self.set_n_features(n_features)
    self.set_compression_rate(compression_rate)
    self.set_d_min(d_min)
    self.set_d_max(d_max)
    self.set_pruning_type(pruning_type)
    self.set_pruning_rate(pruning_rate)
    self.set_n_prune(n_prune)
    self.set_eval_type(eval_type)
    self.set_eval_metric(eval_metric)
    self.set_threshold(thresh)
    self.set_decay(decay)
    self.set_callback_norm(callback_norm)
    self.set_n_samples(n_samples)
    self.set_loss_ratio(loss_ratio)
    self.set_loocv(loocv)
    self.set_decision_metric(decision_metric)
    self.set_remove_outliers(remove_outliers)
    self.set_reset_weights(reset_weights)
    self.set_accelerate(accelerate)
    self.set_release(release)

    self._MAP = {'n_features': self.set_n_features,
                 'compression_rate': self.set_compression_rate,
                 'thresh': self.set_threshold,
                 'decay': self.set_decay,
                 'd_min': self.set_d_min,
                 'd_max': self.set_d_max,
                 'pruning_type': self.set_pruning_type,
                 'pruning_rate': self.set_pruning_rate,
                 'n_prune': self.set_n_prune,
                 'eval_type': self.set_eval_type,
                 'eval_metric': self.set_eval_metric,
                 'callback_norm': self.set_callback_norm,
                 'n_samples': self.set_n_samples,
                 'loss_ratio': self.set_loss_ratio,
                 'loocv': self.set_loocv,
                 'decision_metric': self.set_decision_metric,
                 'remove_outliers': self.set_remove_outliers,
                 'reset_weights': self.set_reset_weights,
                 'accelerate': self.set_accelerate,
                 'release': self.set_release}

  def __repr__(self):
    return ('Parmeter container for the Feature Selection Callback')

  @property
  def settings(self):
    """
    The dictionary with all attributes belonging to the
    :class:`.CallbackParamsNN` class except for the excluded ``n_features``
    attribute.
    """
    return self._settings(['n_features'])

  def _settings(self, exceptions):
    """
    Provides a dictionary with all class attributes that do not begin with an
    underscore (underscore variables are no class parameters).

    Returns
    -------
    new_dict : dict
      The new dictionary with all attributes belonging to the
      :class:`.CallbackParamsNN` class except for the excluded attributes.

    """
    old_dict = vars(self)
    new_dict = dict(old_dict)

    for key in old_dict:
      if (key[0] == '_') or (key in exceptions):
        del new_dict[key]

    return new_dict

  def set_n_features(self, n_features):
    """
    The number of features that the callback tries to extract.

    Parameters
    ----------
    n_features : int
      The number of features.
    """
    self.n_features = n_features

  def set_compression_rate(self, compression_rate):
    """
    The compression rate that is tried to obtain after the callback has been
    applied. It is calculated by the ratio of the number of features after the
    FeaSel-Net algorithm and the original number of features and is only
    needed, when ``n_features`` is not specified. The default is :math:`0.1`.

    Parameters
    ----------
    compression_rate : float
      The compression rate that is tried to obtain after the callback has been
      applied.
    """
    if isinstance(self.n_features, type(None)):
      if compression_rate:
        self.compression_rate = compression_rate
      else:
        self.compression_rate = 0.1 # 10% by default
    else:
      self.compression_rate = None

  def set_threshold(self, thresh):
    """
    Sets the threshold value :math:`\\tau` that has to be surpassed in order to
    trigger the callback. If the ``eval_metric`` `accuracy` or `val_accuracy`
    is chosen, a value of :math:`\\tau_{acc}=0.9` is recommended. The
    corresponding recommendation value for the loss-based metrics is
    :math:`\\tau_g=-0.001`.

    Parameters
    ----------
    thresh : float
      The threshold value.
    """
    self.thresh = self._get_thresh(thresh)

  def set_decay(self, decay):
    """
    Sets the decay value and enables an adaptive thresholding value. If set, it
    decreases ``thresh`` by this value every that the callback has not been
    triggered since the last prune plus the required waiting time ``d_min``.
    This is done until it meets the trigger conditions again. The threshold
    will be set to its original state after the trigger is being pulled. The
    default is `None`.

    Parameters
    ----------
    decay : float
      The decay value.
    """
    self.decay = decay

  def set_d_min(self, d_min):
    """
    Sets the number of epochs in which the trigger conditions have to be met
    consistently.
    Parameters
    ----------
    d_min : int
      The minimum number of epochs.
    """
    self.d_min = d_min

  def set_d_max(self, d_max):
    """
    Sets the maximum number of epochs in which the optimizer tries to meet the
    trigger conditions. If it does not prune in :math:`d_{max}` epochs, the
    algorithm will stop.

    Parameters
    ----------
    d_max : int
      The maximum number of epochs.
    """
    self.d_max = d_max

  def set_pruning_type(self, pruning_type):
    """
    Sets the type of node reduction for the recursive feature elimination. The
    type of pruning determines the default arguments of ``pruning_rate`` and
    ``n_prune``.

    Possible arguments are:
      - `exponential`: The number of features is exponentially decreased and
        pruned with increasing amount of trigger processes depending on an
        initial pruning rate.
      - `linear`: The number of features is linearly decreased and pruned with
        increasing amount of trigger processes depending on an initial number
        of features to prune`.

    Parameters
    ----------
    pruning_type : str
      The pruning type.
    """
    self.pruning_type = self._get_pruning_type(pruning_type)

  def set_pruning_rate(self, pruning_rate):
    """
    Sets the percentage of nodes being eliminated after the update method is
    triggered. The rate is only necessary if `exponential` is chosen as
    ``pruning_type`` and is set to `None` otherwise. It has to be in the range
    of :math:`0-1`.

    Parameters
    ----------
    pruning_rate : float
      The pruning rate.
    """
    if pruning_rate > 1:
      pruning_rate = 1
    elif pruning_rate < 0:
      pruning_rate = 0
    self.pruning_rate = self._get_pruning_rate(pruning_rate)

  def set_n_prune(self, n_prune):
    """
    Sets the absolute number of nodes that are eliminated at each iteration.
    The number is only necessary if `linear` is chosen as ``pruning_type``.

    Parameters
    ----------
    n_prune : int
      The number of nodes to be pruned.
    """
    self.n_prune = self._get_n_prune(n_prune)

  def set_eval_type(self, eval_type):
    """
    Sets the evaluation type that is monitored during training. Its value will
    decide whether and when the callback is executed.

    Possible arguments are:
      - `accuracy`: The `accuracy` history is monitored during training.
      - `loss`: The `loss` history, or its gradient to be precise, is monitored
        during training.
      - `val_accuracy`: The `validation accuracy` history is monitored during
        training.
      - `val_loss`: The `validation loss` history, or its gradient to be
        precise, is monitored during training.

    Parameters
    ----------
    eval_type : str
      The evaluation type that is monitored during training. Its value will
      decide whether and when the callback is executed.
    """
    self.eval_type = self._get_eval_type(eval_type)

  def set_eval_metric(self, eval_metric):
    """
    Sets the metric function that is exectuted for the evaluation of the
    feature importances :math:`\\mathcal{I}_f`.

    Possible arguments are:
      - `cross-entropy`: The algorithm will use the cross-entropy metric for
        the evaluation of the feature importance.
      - `entropy`: The algorithm will use the entropy metric for the evaluation
        of the feature importance.

    Parameters
    ----------
    metric : str
      The metric function that is exectuted for the evaluation of the feature
      importances.
    """
    self.eval_metric = self._get_eval_metric(eval_metric)

  def set_callback_norm(self, callback_norm):
    """
    Sets the evaluated losses per feature can prominently vary over samples.
    Therefore it can be normalized.

    Possible arguments are:
      - `min-max`: The algorithm will scale all features to fit into the range
        :math:`0-1`.
      - `standardize`: The algorithm will scale all features to have
        :math:`\\sigma=1` and :math:`\\mu=0`.

    Parameters
    ----------
    callback_norm : str
      The normalization type.
    """
    if callback_norm:
      self.callback_norm = self._get_callback_norm(callback_norm)
    else:
      self.callback_norm = None



  def set_n_samples(self, n_samples):
    """
    Sets the number of evaluation samples that are chosen for the evaluation of
    the most important features. If `None`, it will not use any validation
    data, but only evaluates the weights in the model without any sample data.
    This implies a significantly faster evaluation method on the one hand and
    worse consistency in the feature selection results on the other hand.

    Parameters
    ----------
    n_samples : int
      The number of samples.
    """
    self.n_samples = n_samples

  def set_loss_ratio(self, loss_ratio):
    """
    Sets the minimum ratio of the difference of the first pruned and last kept
    feature and the best overall feature loss that has to be obtained in
    order to trigger pruning (2nd pruning stage).

    Parameters
    ----------
    loss_ratio : float
      The ratio between last kept and first disposed feature.
    """
    self.loss_ratio = loss_ratio

  def set_decision_metric(self, decision_metric):
    """
    Sets the metric for the decision, which features are to be pruned. The
    possible options are `median` or `average`. The default is `average`.

    Possible arguments are:
      - `median`: The features are evaluated using the median of all losses.
      - `average`: The features are evaluated using the average of all losses.

    Parameters
    ----------
    decision_metric : str
      The desision metric.
    """
    self.decision_metric = self._get_decision_metric(decision_metric)

  def set_remove_outliers(self, remove_outliers):
    """
    Sets the ``remove_outliers`` parameter. If `True`, it will remove all
    outliers from the evaluation data.

    Parameters
    ----------
    remove_outliers : bool
      Decides whether to remove outliers.
    """
    self.remove_outliers = remove_outliers

  def set_reset_weights(self, reset_weights):
    """
    Sets the ``reset_weights`` parameter. If `True`, the weights will be
    resetted and initialized anew at every pruning epoch such that the
    algorithm is unbiased from previous learning and pruning steps for the
    next pruning epoch.

    Parameters
    ----------
    reset_weights : bool
      Decides whether to reset all weights after every prune.
    """
    self.reset_weights = reset_weights

  def set_loocv(self, loocv):
    """
    Sets the ``loocv`` parameter. If `True`, it will apply the principle of the
    leave-one-out cross-validation (LOOCV) and mask only one feature per
    classification. If `False`, it will use the inverse LOOCV and mask every
    feature but one.

    Parameters
    ----------
    loocv : bool
      Decides whether to use LOOCV (`True`) or inverted LOOCV.
    """
    self.loocv = loocv

  def set_accelerate(self, accelerate):
    """
    Sets the acceleration mode. If `True`, it will use a maximum number of
    :math:`1000` features at the same time. This is especially helpful for
    huge datasets. Features will be grouped with other features in their direct
    vicinity.

    Parameters
    ----------
    accelerate : bool
      Decides whether to accelerate the algorithm or not.
    """
    self.accelerate = accelerate

  def set_release(self, release):
    """
    Sets the release mode. If `True`, it releases all previously pruned
    features from the definitive state of being pruned. The features are
    available again and if some other feature performs worse than previously
    pruned features, they will be deleted instead.

    Parameters
    ----------
    release : bool
      Decides whether to release and evaluate features every pruning epoch.
    """
    self.release = release

  # HELPER FUNCTIONS:
  def _get_eval_type(self, eval_type):
    """
    Provides different aliases for all possible evaluation type and searches
    for the corresponding metric (e.g. 'acc' --> 'accuracy').

    Parameters
    ----------
    eval_type : str
      Type alias.

    Raises
    ------
    NameError
      If type is not a valid alias.

    Returns
    -------
    EVAL_TYPE : str
      The proper metric used for the following operations inside the class.

    """
    _EVAL_TYPES = {'accuracy': ['accuracy', 'acc', 'acc.'],
                   'loss': ['loss'],
                   'val_accuracy': ['val_accuracy', 'val_acc'],
                   'val_loss': ['val_loss']}
    for EVAL_TYPE in _EVAL_TYPES:
      if eval_type in _EVAL_TYPES[f'{EVAL_TYPE}']:
        return EVAL_TYPE
    raise NameError(f"'{eval_type}' is not a valid evaluation type.")

  def _get_decision_metric(self, decision_metric):
    """
    Provides different aliases for all possible decision metrics and searches
    for the corresponding metric (e.g. 'mean' --> 'average').

    Parameters
    ----------
    decision_metric : str
      Decision metric alias.

    Raises
    ------
    NameError
      If decision metric is not a valid alias.

    Returns
    -------
    DECISION_METRIC : str
      The proper decision metric used for the following operations inside the
      class.

    """
    _DECISION_TYPES = {'average': ['average', 'mean', 'avrg', 'mu'],
                       'median': ['median', 'med']}
    for DECISION_METRIC in _DECISION_TYPES:
      if decision_metric in _DECISION_TYPES[f'{DECISION_METRIC}']:
        return DECISION_METRIC
    raise NameError(f"'{decision_metric}' is not a valid decision metric.")

  def _get_callback_norm(self, callback_norm):
    """
    Provides different aliases for all possible noramlizations and searches for
    the corresponding normalization (e.g. 'min max' --> 'min_max').

    Parameters
    ----------
    callback_norm : str
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
    _NORM_TYPES = {'min-max': ['min_max', 'min-max',
                               'min max', 'min-max scale'],
                   'standardize': ['standardize', 'standard',
                                   'z score', 'z-score']}
    for NORM in _NORM_TYPES:
      if callback_norm in _NORM_TYPES[f'{NORM}']:
        return NORM
    raise NameError(f"'{callback_norm}' is not a valid normalization.")

  def _get_eval_metric(self, eval_metric):
      """
      Provides different aliases for all possible metrics and searches for
      the corresponding metric (e.g. 'CE' --> 'cross-entropy').

      Parameters
      ----------
      eval_metric : str
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
      _METRIC_TYPES = {'cross-entropy': ['cross-entropy',
                                         'cross_entropy', 'CE'],
                       'entropy': ['entropy']}
      for METRIC in _METRIC_TYPES:
        if eval_metric in _METRIC_TYPES[f'{METRIC}']:
          return METRIC
      raise NameError(f"'{eval_metric}' is not a valid metric.")

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
    _PRUNING_TYPES = {'exponential': ['exponential', 'exp',
                                      'exp.', 'ex', 'ex.'],
                      'linear': ['linear', 'lin', 'lin.']}
    for PRUNING_TYPE in _PRUNING_TYPES:
      if pruning_type in _PRUNING_TYPES[f'{PRUNING_TYPE}']:
        return PRUNING_TYPE
    raise NameError(f"'{pruning_type}' is not a valid pruning type.")

  def _get_thresh(self, thresh):
    """
    Looks up the evaluation map for the default threshold value if no other
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
    _EVAL_MAP = {'accuracy': {'thresh': 0.9},
                 'loss': {'thresh': -1e-3},
                 'val_accuracy': {'thresh': 0.9},
                 'val_loss': {'thresh': -1e-3}}
    if not thresh:
      thresh = _EVAL_MAP[f'{self.eval_type}']['thresh']
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
    _PRUNING_MAP = {'exponential': {'pruning_rate': 0.2,
                                    'n_prune': None},
                    'linear': {'pruning_rate': None,
                               'n_prune': 1}}
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
    _PRUNING_MAP = {'exponential': {'pruning_rate': 0.2,
                                    'n_prune': None},
                    'linear': {'pruning_rate': None,
                               'n_prune': 1}}
    if not n_prune:
      n_prune = _PRUNING_MAP[f'{self.pruning_type}']['n_prune']
    return n_prune

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
    _METRIC_MAP = {'cross-entropy': cross_entropy,
                   'entropy': entropy}
    if not isinstance(P, type(None)):
      loss = _METRIC_MAP[f'{self.eval_metric}'](P, Q)

    else:
      loss = _METRIC_MAP['entropy'](P, Q)

    return loss