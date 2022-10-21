import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from ..data import Classification
from ..plots import FeaselVisualizer as Visualizer
from ..parameters import Params
from ...utils.time import get_timestamp

import os

class ModelContainer:
  def __init__(self,
               X, y=None, features=None,
               name=None, **kwargs):
    self.X = X
    self.y = y
    self.history = None

    if features is not None:
      self.features = np.array(features)
    else:
      self.features = features

    self.set_name(name)

    self._get_params(**kwargs)

    #data container
    self._data = Classification(X, y, features,
                                sample_axis=self.params.data.sample_axis,
                                normalization=self.params.data.normalization,
                                input_layer=self.params.data.input_layer)

    self.timestamp = get_timestamp()

  def __str__(self):
    return 'ModelContainer for Neural Networks'

  def __repr__(self):
    return ("ModelContainer generic model object"
            f"(Size of Dataset: {self._data.X.shape}, "
            f"Number of Samples: {self._data.n_samples}, "
            f"Number of Classes: {self._data.n_classes})")

  @property
  def data(self):
    return self._data

  @property
  def plot(self):
    return self._plot

  def set_name(self, name):
    """
    Sets the name of the container.

    Parameters
    ----------
    name : str
      Name of the container. If None, it will use the class' name. The default
      is None.

    Returns
    -------
    None.

    """
    if not name:
      self.name = str(self)
    else:
      self.name = name

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

      elif key in self.params.data.__dict__.keys():
        self.params.data.update(key)(kwargs[key])

      elif key in self.params.train.__dict__.keys():
        self.params.train.update(key)(kwargs[key])

      else:
        raise KeyError(f"'{key}' is not a valid key for the generic "
                       "neural network useage.")

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
      return self.params
    else:
      try:
        return self.params[f"{type}"]
      except:
        raise NameError(f"'{type}' is an invalid argument for 'type'."
                        " Try 'train', 'build' or 'data' instead.")

  # SETTERS:
  def set_epochs(self, epochs):
    """
    Sets number of epochs.

    Parameters
    ----------
    epochs : int
      Number of epochs.

    Returns
    -------
    None.

    """
    self.params.train.set_epochs(epochs)

  def set_batch_size(self, batch_size):
    """
    Sets size of batch.

    Parameters
    ----------
    batch_size : int
      Size of batch.

    Returns
    -------
    None.

    """
    self.params.train.set_batch_size(batch_size)

  def set_validation_split(self, validation_split):
    """
    Sets ratio between validation and training data.

    Parameters
    ----------
    validation_split : float
      Amount of validation data.

    Returns
    -------
    None.

    """
    self.params.train.set_validation_split(validation_split)

  def set_dropout_rate(self, dropout_rate):
    """
    Sets ratio between activated and inactive nodes.

    Parameters
    ----------
    dropout_rate : float
      Amount of inactive nodes.

    Returns
    -------
    None.

    """
    self.params.train.set_dropout_rate(dropout_rate)

  def set_learning_rate(self, eta):
    """
    Sets learning rate of the optimizer.

    Parameters
    ----------
    learning_rate : float
      Learning rate.

    Returns
    -------
    None.

    """
    self.params.train.set_eta(eta)

  def set_optimizer(self, optimizer):
    """
    Sets the optimizer name. The actual optimizer function is instantiated when
    the training process starts and takes the name and learning rate as
    arguments.

    Parameters
    ----------
    optimizer : str
      The optimizer used for the training.

    Returns
    -------
    None.

    """
    try:
      if optimizer.__module__ == 'keras.optimizers':
        return
    except:
      self.params.train.set_optimizer(optimizer)


  # HELPERS:
  def encode(self, y):
    """
    Automatically encodes the target values in the correct shape. The function
    does not do anything, if the encoding is correct already.

    As an additional benefit, it stores the encoding in a dictionary that can
    be used for e.g. model test purposes.

    Parameters
    ----------
    y : ndarray
      Target array.

    Returns
    -------
    y_train : ndarray
      Target array in correct encoding.

    """
    if y is None:
      return

    if not self.data._label_encoding_map:
      # virtual black box with y_train as return value:
      if self.params.train.loss == "sparse_categorical_crossentropy":
        y_train = self.data.categorical_labels(y)
      elif self.params.train.loss == "categorical_crossentropy":
        y_train = self.data.one_hot_labels(y)

      map = self.get_label_encoding_map(y, y_train)
      self.data._label_encoding_map = map

    else:
      map = self.data._label_encoding_map
      y_train = self._map_labels(y, map)

    return y_train

  def _map_labels(self, y, map):
    """
    Maps the set of targets onto another encoded set of targets.

    Example:
      y = [1., 0., 0., 1.]
      map = {0.: [1,0], 1.: [0,1]}
      y_e = [[0,1], [1,0], [1,0], [0,1]]

    Parameters
    ----------
    y : ndarray
      Original targets.
    map : dict
      Mapping dictionary.

    Returns
    -------
    y_e : ndarray
      Encoded targets.

    """
    # starting with a list because we do not know the dimensionality of y_e
    y = np.array(y, ndmin=1)
    y_e = []
    for i, label in enumerate(y):
      y_e.append(map[label])

    y_e = np.array(y_e)

    return y_e

  # GETTERS:
  def get_layer(self, layer_name):
    """
    Requests the keras layer given by 'layer_name'.

    Parameters
    ----------
    layer_name : str
      Defines name of layer.

    Raises
    ------
    Exception
      If layer is not found.

    Returns
    -------
    layer : keras.layer
      The requested keras layer.

    """
    try:
      layer = self.model.get_layer(layer_name)

    except NameError:
      print(f"Could not find layer '{layer_name}'.")

    return layer

  def get_weights(self, layername, type=None):
    """
    Requests all weights and biases of specified layer.

    Parameters
    ----------
    layername : str
      Defines name of layer.
    type : str
      Defines the type of weights ('weights', 'bias' or None), that is
      returned. If None, both types will be returned as a tuple. The deafult is
      None.

    Returns
    -------
    weights : tuple
      A tuple of weights and biases.
    weights : ndarray
      Weights of trained model at given layer.
    bias : ndarray
      Bias of trained model at given layer.

    """
    layer = self.get_layer(layername)

    if not type:
      return layer.get_weights()

    elif type == 'bias':
      return layer.get_weights[0]

    elif type == 'weights':
      return layer.get_weights[1]

  def get_model(self):
    """
    Getter function for the generic model.

    Returns
    -------
    model : Model
      Keras model of the generic neural network.

    """

    inputs = self.input_layer
    outputs = self.output_layer
    name = self._building_params['architecture_type']
    self.model = Model(inputs=inputs,
                       outputs=outputs,
                       name=name + self.timestamp)
    #sets plot functions (only available when model instantiated)
    self._plot = Visualizer(self)
    return self.model

  def get_feature_maps(self, X, layername):
    """
    Provides the feature maps at a specified layer.

    Parameters
    ----------
    X : ndarray
      The test dataset.
    layername : str
      The layer of the feature map.

    Returns
    -------
    feature_maps : ndarray
      The extracted feature maps.

    """
    test_model = Model(inputs=self.model.inputs,
                       outputs=self.model.get_layer(layername).output)
    feature_maps = self.test(X, model=test_model)

    return feature_maps

  def get_label_encoding_map(self, y, y_e):
    """
    Defines a label encoding map by comparing y and y_e.

    Parameters
    ----------
    y : ndarray
      Array with all targets for the training.
    y_e : ndarray
      Array with all encoded targets for the training.

    Returns
    -------
    map : dict
      A dictionary with the encoding.

    """
    # get encoding of this black box:
    idx = np.unique(y, return_index=True)[1] # gets first idx of unique entry
    classes = y[idx] # gets unique labels in correct order
    classes_encoded = y_e[idx] # gets unique encoded labels in same order

    map = {}
    for i, class_ in enumerate(classes):
      map[class_] = classes_encoded[i]

    return map

  # MODEL METHODS:
  def compile_model(self):
    """
    Compiles the generically built model.

    Returns
    -------
    None.

    """

    self.get_model()
    # uses train parameters 'optimizer' and 'eta' to build the optimizer func:
    optimizer = self.params.train.get_optimizer_function()

    self.model.compile(optimizer=optimizer,
                       loss=self.params.train.loss,
                       metrics=[self.params.train.metric])

  def fit_model(self):
    """
    Fits the generically built model.

    Returns
    -------
    history : Model history
      Training history of the neural network.

    """
    self.compile_model()
    params = self.params.train

    X = self.data.X_train
    y = self.data.y_train

    history = self.model.fit(x=X, y=y, epochs=params.epochs,
                             batch_size=params.batch_size,
                             validation_split=params.validation_split)

    self.params.train._trained = True

    return history

  def train(self):
    """
    Trains the generically built model. Same functionality as
    'fit_model()'.

    Returns
    -------
    history : Model history
      Training history of the neural network.

    """
    if isinstance(self.history, type(None)):
      self.model = self.get_model()
      self.compile_model()
      self.history = self.fit_model()
    return self.history

  def test(self, X_test, y_test=None, model=None):
    """
    Tests the generically built model.

    Parameters
    ----------
    X_test : ndarray
      The test dataset.
    y_test : ndarray, optional
      The label dataset. The default is None.
    model : Model, optional
      The model that is tested. The default is None.

    Returns
    -------
    y_pred : ndarray
      The guessed labels.

    """
    if model is None:
      model = self.model

    X_test, y_test = self.data.prepare(X_test, y_test)
    y_true = self.encode(y_test)
    y_pred = model.predict(X_test)

    return y_pred, y_true

  def save(self):
    model_path = (f"models/{self.model_type}/{self.architecture_type}/"
                  f"{self.epochs}_epochs_{len(self.x_train[0])}_datasize")
    path, filename = os.path.split(model_path)

    if not os.path.exists(path):
      os.makedirs(path)
      print(f"Directory '{path}' created.")

    else:
      print(f"Directory '{path}' already exists")

    model_json = self.model.to_json()

    with open(path + filename + ".json", "w") as jsoile:
      json_file.write(model_json)

    # serialize weights to HDF5
    self.model.save_weights(path + "/" + filename + ".h5")
    print("Saved model to disk.")

  def load(self, model_path):
    self.model_type = model_path.split("/")[1]
    self.architecture_type = model_path.split("/")[2]
    self.model = keras.models.load_model(model_path)