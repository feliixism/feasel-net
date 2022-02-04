from .base import ModelContainer
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
import numpy as np

class DenseDNN(ModelContainer):
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

    self._BLOCK_MAP = {"down": self.get_block_down,
                       "up-down": self.get_block_up_down,
                       "const": self.get_block_const,
                       "exp-down": self.get_block_exp_down
                       }

    # parameter container
    self._building_params = {"architecture_type": "down",
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
                                         random_state = 42)
      self.data.X_train, self.data.X_test = split[0:2]
      self.data.y_train, self.data.y_test = split[2:4]

    self.n_in = np.prod(self.data.feature_shape)
    self.set_n_layers(self._building_params["n_layers"])
    self.get_architecture()

    self._params = {"data": self.params.data,
                    "training": self.params.train,
                    "building": self.params.build}

  def __repr__(self):
    return 'DenseDNN generic model object'

  def encode(self, y): # ist das integrierbar in den data parameter block? <-- hier weitermachen
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
    idx = np.unique(y, return_index=True)[1]
    classes = y[idx]

    # virtual black box with y_train as return value:
    if self.params.train.loss == "sparse_categorical_crossentropy":
      y_train = self.data.categorical_labels(y)
    elif self.params.train.loss == "categorical_crossentropy":
      y_train = self.data.one_hot_labels(y)

    classes_encoded = y_train[idx]

    self.data._label_encoding_map = {}
    for i, class_ in enumerate(classes):
      self.data._label_encoding_map[class_] = classes_encoded[i]

    return y_train


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
    self.n_layers = n_layers
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

  def get_block_exp_down(self, x, n_nodes = None, n_layers = 3):
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

  def get_block_down(self, x, n_nodes = None, n_layers = 3):
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

    for i in range(1, n_layers+1):
      x = Dense(int(n_nodes / (2 * i)),
                activation = self.params.train.activation,
                name = f"Dense{i}")(x)
      x = self.get_dropout(x, i)

    return x

  def get_block_const(self, x, n_nodes = None, n_layers = 3):
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
                activation = self.params.train.activation,
                name = f"Dense{i+1}")(x)
      x = self.get_dropout(x, i)

    return x

  def get_block_up_down(self, x, n_nodes = None, n_layers = 3):
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

    if n_layers%2 == 0:
      n_layers1, n_layers2 = int(n_layers / 2), int(n_layers / 2)
    else:
      n_layers1 = int((n_layers + 1) / 2)
      n_layers2 = int((n_layers - 1) / 2)

    for i in range(1, n_layers1 + 1):
      x = Dense(n_nodes * (2*i),
                activation = self.params.train.activation,
                name = f"Dense{i}")(x)
      x = self.get_dropout(x, i)

    n_nodes = n_nodes * (2*i)

    for j in range(1, n_layers2 + 1):
      i += 1
      x = Dense(int(n_nodes / (2*j)),
                activation = self.params.train.activation,
                name = f"Dense{i}")(x)
      x = self.get_dropout(x, i)

    return x

  def fit_model(self):
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