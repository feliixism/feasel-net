from tensorflow.keras.layers import Input, Dense
from .tfcustom.callbacks import FeatureSelection
from .tfcustom.layers import LinearPass
from .parameters import Params

from .architectures import DenseDNN

class FeaselDNN(DenseDNN):
  def __init__(self, X, y, layer_name=None, n_features=None, **kwargs):
    """
    The feature selection class object for Dense type deep neural networks
    (DNNs).

    Parameters
    ----------
    X : ndarray
      Input array for training (and validation) data.
    y : ndarray
      Input array for training (and validation) targets.
    layer_name : str, optional
      Layer name where the feature selection is applied. The default is None.
    n_features : int, optional
      Number of features that shall be remaining. If None, a compression ratio
      of 10 % is used. The default is None.
    **kwargs : kwargs
      All other accepted parameters. They can be inspected by using the method
      show_params().

    Returns
    -------
    None.

    """
    super().__init__(X, y, **kwargs)

    self.layer_name = layer_name
    self.params.callback.set_n_features(n_features)
    self.n_features = self.get_n_features()
    self.callback = None

  def __str__(self):
    return "FeaselDNN"

  def __repr__(self):
    return ("FeaselDNN generic model object"
            f"(Size of Dataset: {self._data.X.shape}, "
            f"Number of Samples: {self._data.n_samples}, "
            f"Number of Classes: {self._data.n_classes})")

  # HELPERS:
  def _get_params(self, **kwargs):
    """
    Automatically checks the kwargs for valid parameters for each parameter
    object and updates them.

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
    self.params = Params()

    for key in kwargs:
      containers = {'build': self.params.build,
                    'callback': self.params.callback,
                    'data': self.params.data,
                    'train': self.params.train}

      # updates all values that are summarized in an extra container:
      if key in containers:
        for sub_key in kwargs[key]:
          if sub_key in containers[key].__dict__.keys():
            containers[key].update(sub_key)(kwargs[key][sub_key])

      # updates keys if they are not defined in an extra container:
      elif key in self.params.build.__dict__.keys():
        self.params.build.update(key)(kwargs[key])

      elif key in self.params.callback.__dict__.keys():
        self.params.callback.update(key)(kwargs[key])

      elif key in self.params.data.__dict__.keys():
        self.params.data.update(key)(kwargs[key])

      elif key in self.params.train.__dict__.keys():
        self.params.train.update(key)(kwargs[key])

      else:
        raise KeyError(f"'{key}' is not a valid key for the generic "
                       "neural network useage.")

  # GETTERS:
  def get_architecture(self):
    """
    Automatically generates the required architecture for a feature selection
    in DNNs. The architecture is sequential.

    Returns
    -------
    None.

    """
    self.input_layer = x = Input(shape=(self.n_in, ),
                                 name="Input")

    x = LinearPass(name="Linear")(x) # layer for the feature selection

    x = self.get_block(x, n_nodes=None, n_layers=self.n_layers,
                       architecture_type=self.params.build.architecture_type)

    self.output_layer = Dense(self.data.n_classes,
                              activation="softmax",
                              name="Output")(x)

  def get_n_features(self):
    """
    Provides the number of remaining features after the selection. If it is not
    defined by the callback parameters, it will calculate the features using a
    compression rate.

    Returns
    -------
    n_features : TYPE
      DESCRIPTION.

    """
    if not self.params.callback.n_features:
      n_features = int(self.data.feature_shape.size
                    * self.params.callback.compression_rate)

    else:
      n_features = self.params.callback.n_features

    return n_features

  def fit_model(self):
    """
    The standard model.fit() method by keras with the feature selection
    callback integrated.

    Returns
    -------
    history : model.history
      The keras model training history.

    """
    if not self.callback:
      self.callback = self.get_callback(layer_name=self.layer_name,
                                        n_features=self.n_features,
                                        **self.params.callback.dictionary)

    history = self.model.fit(x=self.data.X_train, y=self.data.y_train,
                             epochs=self.params.train.epochs,
                             shuffle=False, # data is shuffled already
                             batch_size=self.params.train.batch_size,
                             validation_data=(self.data.X_test,
                                              self.data.y_test),
                             callbacks=[self.callback],
                             verbose=False) # only FS log is shown

    self.params.train._trained = True

    return history

  def set_callback(self, layer_name, n_features=None, **kwargs):

    if not layer_name:
      raise KeyError("Please provide the layer for the feature "
                     "selection algorithm.")

    else:
      print("Feature Selection Callback is instantiated.\n"
            f"The algorithm tries to find the {self.n_features} "
            "most important features.\n")

    eval_data = (self.data.X_train, self.data.y_train)

    self.callback = FeatureSelection(evaluation_data=eval_data,
                                     layer_name=layer_name,
                                     n_features=self.n_features,
                                     **kwargs)
    return self.callback

  def get_callback(self, layer_name, n_features=None, **kwargs):

    if not self.callback:
      callback = self.set_callback(layer_name, n_features, **kwargs)

    else:
      callback = self.callback

    return callback

  def get_mask(self):
    return self.callback.log.weights[-1].astype(bool)