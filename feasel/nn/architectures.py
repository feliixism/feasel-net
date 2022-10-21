"""
feasel.nn.architectures
=======================
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from ..data import classification
from ..plot import FeaselVisualizer as Visualizer
from ..parameter import ParamsNN
from .tfcustom.callbacks import FeatureSelection
from .tfcustom.layers import LinearPass
from ..utils.time import get_timestamp

import os

class Base:
  """
  A model container, that stores all the information needed for a baseline
  generic neural network builder.

  Attributes
  ----------
  X : ndarray (*float* or *int*)
    The input data for training and validation purposes.
  y : ndarray (*str*, *float* or *int*)
    The target data. During the training process, it will either be one-hot or
    sparsely encoded, depending on what has been specified in the builder.
  history : *None*
    Placeholder for history information during training.
  name : *str*
    The name of the model container.
  timestamp : *str*
    The timestamp as additional information, when the current model has been
    initialized.

  """
  def __init__(self, X, y=None, features=None, name=None, **kwargs):
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
    self._data = classification.NN(X, y, features,
                                   sample_axis=self.params.data.sample_axis,
                                   input_layer=self.params.data.input_layer)

    self.timestamp = get_timestamp()

  def __str__(self):
    return 'ModelContainer for Neural Networks'

  def __repr__(self):
    return ("Base generic model object"
            f"(Size of Dataset: {self._data.X.shape}, "
            f"Number of Samples: {self._data.n_samples}, "
            f"Number of Classes: {self._data.n_classes})")

  @property
  def params(self):
    """
    A parameter container :obj:*feasel.parameters.Params* that has all
    information on *build*, *callback*, *data*, and *train* processes.
    """
    return self._params

  @property
  def data(self):
    """
    `feasel.data.classification.NN`
      The data container used for the generic neural networks.
    """
    return self._data

  @property
  def plot(self):
    """
    `feasel.plot.FeaselVisualizer`
      The plot container used for plotting the feature selection results.
    """
    return self._plot

  def set_name(self, name):
    """
    Sets the name of the model container.

    Parameters
    ----------
    name : str
      Name of the container. If `None`, it will use the class' name. The
      default is `None`.
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
    self._params = ParamsNN()

    containers = {'build': self.params.build,
                  'data': self.params.data,
                  'train': self.params.train,
                  'callback': self.params.callback}

    for key in kwargs:
      # updates all values that are summarized in an extra container:
      if key in containers:
        for sub_key in kwargs[key]:
          if sub_key in containers[key].__dict__.keys():
            containers[key].update(sub_key)(kwargs[key][sub_key])

      # updates keys if they are not defined in an extra container:
      for C in containers:
        params = containers[C]
        if key in params.__dict__.keys():
          params.update(key)(kwargs[key])

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

  def set_test_split(self, validation_split):
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
    self.params.train.set_test_split(validation_split)

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
  def compile(self):
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

  def fit(self):
    """
    Fits the generically built model.

    Returns
    -------
    history : Model history
      Training history of the neural network.

    """
    self.compile()
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
    'fit()'.

    Returns
    -------
    history : Model history
      Training history of the neural network.

    """
    if isinstance(self.history, type(None)):
      self.model = self.get_model()
      self.compile()
      self.history = self.fit()
    return self.history

  def test(self, X_test, y_test=None, model=None):
    """
    Tests the generically built model and returns probabilities.

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
      The probabilities.

    """
    if model is None:
      model = self.model

    X_test, y_test = self.data.prepare(X_test, y_test)
    y_true = self.encode(y_test)
    y_pred = model.predict(X_test)

    return y_pred, y_true

  def predict(self, X_test, y_test=None, model=None):
    """
    Tests the generically built model and returns classes.

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

    y_p = np.zeros(y_pred.shape)

    for i, j in enumerate(np.argmax(y_pred, axis=1)):
      y_p[i,j] = 1

    y_pred = y_p

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

    with open(path + filename + ".json", "w") as json_file:
      json_file.write(model_json)

    # serialize weights to HDF5
    self.model.save_weights(path + "/" + filename + ".h5")
    print("Saved model to disk.")

  def load(self, model_path):
    self.model_type = model_path.split("/")[1]
    self.architecture_type = model_path.split("/")[2]
    self.model = keras.models.load_model(model_path)

class FCDNN(Base):
  def __init__(self, X, y, **kwargs):
    """
    Builds an DNN with only dense layers. Subclass of 'SpectralNeuralNet'.

    Parameters
    ----------
    X : np-array
      Input for the DNN.
    y : np-array
      Output of the DNN.

    Returns
    -------
    Class object.

    """
    super().__init__(X, y, **kwargs)

    self._BLOCK_MAP = {"linear": self.get_block_down,
                       "rhombus": self.get_block_up_down,
                       "const": self.get_block_const,
                       "exponential": self.get_block_exp_down
                       }

    # parameter container
    self._building_params = {"architecture_type": "linear",
                             "n_layers": 3,
                             "n_nodes": None}

    for key in kwargs:
      if key in self._building_params:
        self._building_params[f"{key}"] = kwargs[f"{key}"]

    # encoding of target
    self.data.y_train = self.encode(self.data.y_train)

    if self.params.train.test_split:
      # random state is set constant for reproducability:
      split = self.data.train_test_split(self.data.X_train, self.data.y_train,
                                         self.params.train.test_split,
                                         random_state=42)
      self.data.X_train, self.data.X_test = split[0:2]
      self.data.y_train, self.data.y_test = split[2:4]

    self.n_in = np.prod(self.data.feature_shape)
    self.set_n_layers(self._building_params["n_layers"])
    self.get_architecture()

    # self._params = {"data": self.params.data,
    #                 "training": self.params.train,
    #                 "building": self.params.build}

  def __str__(self):
    return 'DenseDNN'

  def __repr__(self):
    return ("DenseDNN generic model object"
            f"(Size of Dataset: {self._data.X.shape}, "
            f"Number of Samples: {self._data.n_samples}, "
            f"Number of Classes: {self._data.n_classes})")

  def set_n_layers(self,
                   n_layers):
    """
    Sets the number of layers in the neural network core.

    Parameters
    ----------
    n_layers : int
      Number of layers in the neural network.

    Returns
    -------
    None.

    """
    self.params.build.n_layers = n_layers
    self.get_architecture()

  def get_block(self, x, n_nodes, n_layers, architecture_type):
    """
    Secures that the given 'architecture_type' is available and provides
    the specific architecture block.

    Parameters
    ----------
    x : tf.Tensor
      Input tensor.
    n_nodes : int
      Number of nodes to start the block with.
    n_layers : int
      Number of layers for the neural network core.
    architecture_type : str
      Defines the architecture type.

    Raises
    ------
    TypeError
      If the 'architecture_type' is not implemented yet.

    Returns
    -------
    architecture : model
      The core architecture for the generically built model.

    """
    if architecture_type not in self._BLOCK_MAP.keys():
      raise TypeError(f"The type '{architecture_type}' is not "
                      "implemented. Try another 'architecture_type'. "
                      "Possible options are 'down', 'up-down', 'const' "
                      "or 'exp-down'.")

    else:
        return self._BLOCK_MAP[f'{architecture_type}'](x, n_nodes, n_layers)

  def get_architecture(self):
    """
    The framework for building the generic neural network. It uses the
    'get_block()' method to fill the inner structure between input and
    output layer.

    Returns
    -------
    None.

    """
    architecture_type = self.params.build.architecture_type
    n_nodes = self.params.build.n_nodes
    n_layers = self.params.build.n_layers

    self.input_layer = x = Input(shape=(self.n_in, ),
                                 name="Input")

    x = self.get_block(x,
                       n_nodes=n_nodes,
                       n_layers=n_layers,
                       architecture_type=architecture_type)

    self.output_layer = Dense(self.data.n_classes,
                              activation="softmax",
                              name="Output")(x)

  def get_dropout(self, x, idx):
    if self.params.train.dropout_rate:
      x = Dropout(self.params.train.dropout_rate,
                  name=f'Dropout{idx}')(x)
    return x

  def get_block_exp_down(self, x, n_nodes=None, n_layers=3):
    """
    Generates a generic neural network architecture with an exponential
    decline of layer nodes. The first layer defines the number of nodes
    in the first dense layer, if 'n_nodes' is None. Otherwise, the number
    of 'n_nodes' will be reduced exponentially:

      n_nodes(i) = n_nodes * 0.5**i

    Parameters
    ----------
    x : tf.Tensor
      The tensor of the previous layer.
    n_nodes : int, optional
      Number of nodes in the first dense layer. If None, it will be as
      much as the input nodes. The default is None.
    n_layers : int, optional
      Number of layers in the network. The default is 3.

    Returns
    -------
    x : tf.Tensor
      The tensor of this layer.

    """
    if n_nodes is None:
      n_nodes = self.n_in

    n_out = self.data.n_classes

    p = (n_out / n_nodes)**(1 / n_layers)

    for i in range(0, n_layers):
      nodes = int(np.round(n_nodes * p**i, 0))
      x = Dense(nodes,
                activation=self.params.train.activation,
                name=f"Dense{i+1}")(x)
      x = self.get_dropout(x, i)

    return x

  def get_block_down(self, x, n_nodes=None, n_layers=3):
    """
    Generates a generic neural network architecture with a linear
    decline of layer nodes. The first layer defines the number of nodes
    in the first dense layer, if 'n_nodes' is None. Otherwise, the number
    of 'n_nodes' will be reduced linearly:

      n_nodes(i) = n_nodes / (2*i)

    Parameters
    ----------
    x : tf.Tensor
      The tensor of the previous layer.
    n_nodes : int, optional
      Number of nodes in the first dense layer. If None, it will be as
      much as the input nodes. The default is None.
    n_layers : int, optional
      Number of layers in the network. The default is 3.

    Returns
    -------
    x : tf.Tensor
      The tensor of this layer.

    """
    if n_nodes is None:
      n_nodes = self.n_in

    n_out = self.data.n_classes

    for i in range(0, n_layers):
      x = Dense(int(n_nodes - (n_nodes - n_out) / n_layers * i),
                activation=self.params.train.activation,
                name=f"Dense{i}")(x)
      x = self.get_dropout(x, i)

    return x

  def get_block_const(self, x, n_nodes=None, n_layers=3):
    """
    Generates a generic neural network architecture with contant numbers
    of layer nodes. The first layer defines the number of nodes
    all dense layer, if 'n_nodes' is None. Otherwise, the number
    of 'n_nodes' will be used:

      n_nodes(i) = n_nodes

    Parameters
    ----------
    x : tf.Tensor
      The tensor of the previous layer.
    n_nodes : int, optional
      Number of nodes in the first dense layer. If None, it will be as
      much as the input nodes. The default is None.
    n_layers : int, optional
      Number of layers in the network. The default is 3.

    Returns
    -------
    x : tf.Tensor
      The tensor of this layer.

    """
    if n_nodes is None:
      n_nodes = self.n_in

    for i in range(0, n_layers):
      x = Dense(n_nodes,
                activation=self.params.train.activation,
                name=f"Dense{i+1}")(x)
      x = self.get_dropout(x, i)

    return x

  def get_block_up_down(self, x, n_nodes=None, n_layers=3):
    """
    Generates a generic neural network architecture with a linear increase
    and decline of layer nodes afterwards. The first layer defines the
    number of nodes in the first dense layer, if 'n_nodes' is None.
    Otherwise, the number of 'n_nodes' will be reduced linearly:

        n_nodes(i) = n_nodes * (2*i) for all i from 0 to n_layers and
        n_nodes(i) = n_nodes(n_layers) / (2*i)

    Parameters
    ----------
    x : tf.Tenso
      The tensor of the previous layer.
    n_nodes : int, optional
      Number of nodes in the first dense layer. If None, it will be as
      much as the input nodes. The default is None.
    n_layers : int, optional
      Number of layers in the network. The default is 3.

    Returns
    -------
    x : tf.Tensor
      The tensor of this layer.

    """
    if n_nodes is None:
      n_nodes = self.n_in

    n_layers1 = int(np.floor(n_layers / 2))
    n_layers2 = int(np.ceil(n_layers / 2))

    for i in range(1, n_layers1 + 2):
      x = Dense(int(n_nodes + n_nodes * i),
                activation=self.params.train.activation,
                name=f"Dense{i}")(x)
      x = self.get_dropout(x, i)

    n_peak = n_nodes + n_nodes * i
    n_out = self.data.n_classes

    for j in range(n_layers2 - 1):
      i += 1
      x = Dense(int(n_peak - (n_peak - n_out) / n_layers2 * (j+1)),
                activation=self.params.train.activation,
                name=f"Dense{i}")(x)
      x = self.get_dropout(x, i)

    return x

  def fit(self):
    """
    Fits the generically built model.

    Returns
    -------
    history : Model history
      Training history of the neural network.

    """
    history = self.model.fit(x=self.data.X_train,
                             y=self.data.y_train,
                             epochs=self.params.train.epochs,
                             shuffle=False,
                             batch_size= self.params.train.batch_size,
                             validation_data=(self.data.X_test,
                                              self.data.y_test),
                             verbose=True)
    return history

class FSDNN(FCDNN):
  """
  The feature selection class object for Dense type deep neural networks
  (DNNs).

  Parameters
  ----------
  X : ndarray (*float* or *int*)
    Input array for training (and validation) data.
  y : ndarray (*str*, *float* or *int*)
    Input array for training (and validation) targets.
  layer_name : *str*
    Layer name where the feature selection is applied.
  n_features : int, optional
    Number of features that shall be remaining. If None, a compression ratio
    of 10 % is used. The default is None.
  **kwargs : kwargs
    All other accepted parameters. They can be inspected by using the method
    :func:*feasel.nn.architectures.Base.show_params()*.

  """
  def __init__(self, X, y, layer_name, n_features=None, **kwargs):
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
    self._params = ParamsNN()

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

    x = self.get_block(x, n_nodes=None, n_layers=self.params.build.n_layers,
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
      n_features = int(np.product(self.data.feature_shape)
                       * self.params.callback.compression_rate)

    else:
      n_features = self.params.callback.n_features

    return n_features

  def fit(self):
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
                                        **self.params.callback.settings)
    if not isinstance(self.data.X_test, type(None)):
      validation_data = (self.data.X_test, self.data.y_test)
    else:
      validation_data = None

    history = self.model.fit(x=self.data.X_train, y=self.data.y_train,
                             epochs=self.params.train.epochs,
                             shuffle=False, # data is shuffled already
                             batch_size=self.params.train.batch_size,
                             validation_data=validation_data,
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

  def get_params():
    return

  def reset(self):
    self.history = None
