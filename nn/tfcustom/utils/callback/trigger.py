"""
feasel.nn.tfcustom.utils.callback.trigger
=========================================
"""

from .log import CallbackLog

class CallbackTrigger(CallbackLog):
  def __init__(self, model, layer, params):
    """
    A class that keeps track of the current state of all relevant trigger
    variables. The class variables decide, whether to prune features or
    not. It is a subclass of the CallbackLog class since both need the
    same information.

    Parameters
    ----------
    params : CallbackParams
      The parameter object for the feature selection algorithm. It contains all
      relevant initializing parameters.
    model : keras.engine.training.Model
      The neural network architecture plus the pre-trained weights.
    layer_name : str
      The layer where the feature selection is applied.

    Returns
    -------
    None.

    """
    super().__init__(model, layer, params)

  def __repr__(self):
    return ('Trigger container for the Feature Selection Callback\n'
            f'{self.__dict__}')

  def initialize(self, logs):
    """
    Initializes the Log class with all the values after the very first training
    epoch. This method has to be called after the class instantiation because
    it needs the model information that, at the time of instantiation, has not
    been instantiated yet and thus cannot be referred to.

    Initializes the following parameters:
      - value_l: A list of the spectated values (loss or classification
          accuracy).
      - value: The current value of the spectated loss or classification
          accuracy.
      - gradient_l: A list of the spectated gradients of the corresponding loss
          values.
      - gradient: The gradient of the loss value at the current epoch.
      - window: The window size for the Differentiation Quotient (DQ)
          calculation.
      - thresh_l: A list of the threshold for the accuracy or gradient values
          that have to be surpassed (acc and gradient > thresh).
      - thresh: The threshold for the accuracy or gradient values that have to
          be surpassed (acc and gradient > thresh). It adaptively changes over
          time, if the actual threshold cannot be reached.
      - epochs: Epoch counter for the consistency criterion.
      - d: The number of epochs since the first time the first trigger
          condition (surpassing the threshold value several times in a row) has
          been met in a row.
      - d_hit: The number of epochs since the last time the first trigger
          condition has been met.
      - d_prune: The number of epochs since the last time the feature
          selection algorithm was triggered.
      - _epoch_max: Bool whether max. number of epochs since last prune is
          reached.
      - _success: Bool whether desired number of features is obtained.
      - _prune: Bool whether pruning criterion is met or not.
      - _pruned: Bool whether pruned for at least one time.
      - _stop: Bool whether to stop early or not.

    Parameters
    ----------
    logs : dict, optional
      A dictionary of the current epoch's metric. The default is {}.

    Returns
    -------
    None.

    """
    # class variables needed for the threshold trigger criteria:
    self.value_l = [] # value list
    self.value = None # current value

    self.gradient_l = [] # graident list
    self.gradient = None # current gradient

    self.window = 10 # needed for the calculation of the gradient

    self.thresh_l = [self.params.thresh] # threshold list
    self.thresh = self.thresh_l[0] # current threshold

    # class variables needed for the consistency trigger criteria:
    self.epochs = 0

    self.d = 0
    self.d_hit = 0
    self.d_prune = 0

    # early-stopping criteria:
    self._epoch_max = False
    self._success = False

    self._prune = False
    self._pruned = False
    self._stop = False

  def update(self, logs, n_features):
    """
    Updates many class variables initialized by initialize() after every
    epoch.

    Parameters
    ----------
    loss : ndarray
      The loss values for each feature.

    Returns
    -------
    None.

    """
    # stores loss or accuracy value:
    self.value_l.append(logs[f'{self.params.eval_type}']) # list of values
    self.value = self.value_l[-1] # current value

    # stores gradient of loss (or val_loss) value:
    self.gradient_l.append(self._get_gradient(logs, self.window)) # list
    self.gradient = self.gradient_l[-1] # last value

    self.thresh_l.append(self.thresh)

    # epoch iterator
    self.epochs += 1

    # check early stopping criteria:
    self._stop = self._early_stopping_criteria(n_features)

    # check pruning criteria and update hits or misses:
    self._prune = self._pruning_criteria(logs)

  def _pruning_criteria(self, logs):
    """
    Checks whether the pruning conditions are met. There are different possible
    conditions depending on the metric (loss or accuracy).

    Parameters
    ----------
    logs : dict
      Contains default information on the training results (accuracy and
      loss values).

    Returns
    -------
    pruned : bool
      True, if the callback did prune the features for at least one time.

    """
    prune = False

    if self.params.eval_type in ['accuracy', 'val_accuracy']:
      x = self.value # must surpass threshold if accuracy-based

    elif self.params.eval_type in ['loss', 'val_loss']:
      x = self.gradient # must surpass threshold if loss-based

    # threshold criterion:
    if x and x >= self.thresh:
      self._update_hit(logs)

    else:
      self._update_miss(logs)

    # consistency crierion:
    if (self.d >= self.params.d_min) and (not self._success):
      prune = True
      self._update_prune(logs)

    return prune

  def _early_stopping_criteria(self, n_features):
    """
    Updates the boolean stopping criteria values. If the number of desired
    features is attained, it sets the feature criterion True and if the
    algorithm surpassed the maximum number of training epochs without an
    update, it will set the max criterion True.

    Returns
    -------
    stop : bool
      True, if both criteria for the early stopping are met.

    """
    #stopping criterion 1
    if n_features <= self.params.n_features:
      self._success = True

    #stopping criterion 2
    if self.d_prune >= self.params.d_max:
      self._epoch_max = True

    if self._success and self._epoch_max:
      stop = True
    elif self._epoch_max:
      stop = True
    else:
      stop = False

    return stop

  def _update_miss(self, logs):
    """
    Updates all other class variables initialized by initialize() after
    every miss (not meeting the first criterion).

    Parameters
    ----------
    loss : ndarray
      The loss values for each feature.

    Returns
    -------
    None.

    """
    self.d = 0
    self.d_hit += 1
    self.d_prune += 1

    # adaptive thresholding
    d_miss_after_min = self.d_hit - self.params.d_min

    if d_miss_after_min > 0 and self._pruned:
      self.thresh = self.params.thresh - (d_miss_after_min
                                          * self.params.decay)

  def _update_hit(self, logs):
    """
    Updates all other class variables initialized by initialize() after
    every hit (meeting the first criterion).

    Parameters
    ----------
    loss : ndarray
      The loss values for each feature.

    Returns
    -------
    None.

    """
    self.d += 1
    self.d_hit = 0
    self.d_prune += 1

  def _update_prune(self, logs):
    """
    Updates all other class variables initialized by initialize() after
    every time the feature callback prunes features (meeting first and second
    criterion).

    Parameters
    ----------
    loss : ndarray
      The loss values for each feature.

    Returns
    -------
    None.

    """
    self.d = 0
    self.d_hit = 0
    if not self._success:
      self.d_prune = 0
    self.thresh = self.params.thresh
    self._pruned = True # checks whether it is pruned at least once
    self.epochs += 1
    self._epoch_max = False

  def _get_gradient(self, logs, h):
    """
    Calculates the gradient at the current epoch according to the first
    order backward difference quotient.

    Parameters
    ----------
    h : int
      The window size and thus number of epochs that are considered to
      calculate the gradient.

    Returns
    -------
    gradient : float
      The gradient of the loss.

    """
    l = len(self.value_l)

    if l <= h:
      gradient = (self.value_l[-1] - self.value_l[0]) / l
    else:
      gradient = (self.value_l[-1] - self.value_l[-h-1]) / h

    if l > 1:
      return gradient