import numpy as np

class CallbackLog:
  def __init__(self, params, model, layer_name):
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
      The neural network architecture plus the pre-trained weights.
    layer_name : str
      The layer where the feature selection is applied.

    Returns
    -------
    None.

    """
    self.params = params
    self.model = model

    self.layer_name = layer_name
    self._check_layer()

    self._pruned = False

  def __repr__(self):
    return ('Log container for the Feature Selection Callback\n'
            f'{self.__dict__}')

  def _check_layer(self):
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
    if (self.model.get_layer(self.layer_name).__class__.__name__
        == "LinearPass"):
      self.layer = self.model.get_layer(self.layer_name)

    else:
      raise NameError(f"'{self.layer}' is not a 'LinearPass' layer. "
                      "Please choose another layer or make sure that the"
                      " specified layer is part of your model.")

  def initialize(self, logs):
    """
    Initializes the Log class with all the values after the very first
    training epoch. This method has to be called after the class
    instantiation because it needs the model information that, at the time
    of instantiation, has not been instantiated yet and thus cannot be
    referred to.

    Initializes the following parameters:
      - pruning_epochs: A list of all epochs when the callback has been
          triggered.
      - index: A list of ndarray matrices with all the indexes where the
          signal has been masked or pruned.
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

    mask = np.array(self.layer.weights[0].numpy(), ndmin=2)

    self.weights = mask
    self.n_features = np.sum(mask, keepdims=True)
    self.pruning_epochs = np.array(0, ndmin=2)

    self.loss_f = None
    self.loss_o = None

    # indices for visualization purposes:
    self.mask_k = mask.astype(bool)
    self.mask_p = ~mask.astype(bool)

  def update(self, epoch, loss):
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
    self.layer = self.model.get_layer(self.layer_name)
    mask = np.array(self.layer.weights[0].numpy(), ndmin=2)

    self.weights = np.append(self.weights, mask, axis=0)
    self.n_features = np.append(self.n_features,
                                np.sum(mask, keepdims=True),
                                axis=0)
    self.pruning_epochs = np.append(self.pruning_epochs,
                                    np.array(epoch, ndmin=2), axis=0)

    self.mask_k = np.append(self.mask_k, mask.astype(bool), axis=0)
    self.mask_p = np.append(self.mask_p, ~mask.astype(bool), axis=0)

    self._pruned = True