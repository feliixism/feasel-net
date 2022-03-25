import numpy as np

class CallbackLog:
  def __init__(self, model, layer, params):
    """
    A callback log class that tracks all relevant and interesting data
    during the callback such as the loss values for the decisoion whether
    to keep features or not (class variable: 'loss').

    Parameters
    ----------
    params : CallbackParams
      The parameter object for the feature selection algorithm. It contains all
      relevant initializing parameters.
    model : keras.engine.training.Model
      The neural network architecture plus pre-trained weights.
    layer : str
      The layer name where the feature selection is applied. The layer must be
      of type 'LinearPass'.

    Returns
    -------
    None.

    """
    self.params = params
    self.model = model

    self.layer = layer
    self._correct_layer() # layer must be of type 'LinearPass'

  def __repr__(self):
    return ('Log container for the Feature Selection Callback\n'
            f'{self.__dict__}')

  def _correct_layer(self):
    """
    Checks whether the layer is suited for the callback: must be a 'LinearPass'
    layer type.

    Raises
    ------
    NameError
      Raised if layer is not of type 'LinearPass'.

    Returns
    -------
    None.

    """
    if not self.model.get_layer(self.layer).__class__.__name__ == "LinearPass":
      raise NameError(f"'{self.layer}' is not a 'LinearPass' layer. "
                      "Please choose another layer or make sure that the"
                      " specified layer is part of your model.")

  def initialize(self, logs={}):
    """
    Initializes the Log class with all the values after the very first
    training epoch. This method has to be called after the class
    instantiation because it needs the model information that, at the time
    of instantiation, has not been instantiated yet and thus cannot be
    referred to.

    Initializes the following parameters:
      - m: The current values for the weights in the 'LinearPass' layer. This
          is masking the input and thus considered to be a mask m.
      - m_k: A list with all masks m for keeping the features stored at each
          pruning epoch.
      - m_p: A list of all inverted masks (pruning features) at each pruning
          epoch.
      - e_prune: A list of all epochs e when the callback has been triggered
          for the pruning.
      - e_stop: The last overall epoch of the feature selection algorithm
      - f_n: A list with the number of features after each pruning epoch.
      - f_loss: An array with all the loss values for all samples during
          feature evaluation. This provides an insight into the decision
          process for keeping or pruning features.
      - f_eval: An array with all the final feature loss values for the
          decision process of keeping or pruning features.
      - weights: A list of ndarray matrices with the actual signal masks.
          This is a very interesting class variable to inspect the
          feature selection process, which features have been pruned
          first and which survived longer.
      - n_features: A list with the number of features left after every
          iteration.
      - loss: A list of ndarray matrices with the loss values for every
          feature after the evaluation with the validation set.

    Parameters
    ----------
    logs : dict, optional
      A dictionary of the current epoch's metric. The default is {}.

    Returns
    -------
    None.

    """

    layer = self.model.get_layer(self.layer)
    # initiallized with ones only:
    ones = np.array(np.ones(layer.weights[0].numpy().shape), ndmin=2)
    layer.set_weights(ones)

    self.m = ones # current weights
    self.m_k = ones.astype(bool) # mask with indices to keep
    self.m_p = ~ones.astype(bool) # inversion: mask with indices to delete

    self.e_prune = np.array(0, ndmin=2) # pruning epochs
    self.e_stop = None # stopping epoch

    self.f_n = np.sum(ones, keepdims=True) # list of number of features
    self.f_loss = None # loss array for features and samples
    self.f_eval = [] # loss for feature evaluation

    self._pruned = False

  def update(self, epoch, loss, converged=False):
    """
    Updates all class variables initialized by initialize().

    Parameters
    ----------
    epoch : int
      The current training epoch.
    loss : ndarray
      The loss values for each feature.

    Returns
    -------
    None.

    """
    layer = self.model.get_layer(self.layer)

    self.m = np.array(layer.weights[0].numpy(), ndmin=2)

    if not converged:
      self.f_n = np.append(self.f_n, np.sum(self.m, keepdims=True), axis=0)

      self.e_prune = np.append(self.e_prune, np.array(epoch, ndmin=2), axis=0)

      self.m_k = np.append(self.m_k, self.m.astype(bool), axis=0)
      self.m_p = np.append(self.m_p, ~self.m.astype(bool), axis=0)

    else:
      self.e_stop = epoch

    self._pruned = True