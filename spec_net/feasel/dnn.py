from tensorflow.keras.layers import Input, Dense
from ..tfcustom.callbacks import FeatureSelection
from ..tfcustom.layers import LinearPass

from ..architectures.dense import DenseDNN

class FeaselDNN(DenseDNN):
  def __init__(self, X, y,
               layer_name,
               n_features=None,
               callback=None,
               **kwargs):
    super().__init__(X, y, **kwargs)

    if not n_features:
      if 'compression_rate' in kwargs:
        self.compression_rate = kwargs['compression_rate']
      else:
        self.compression_rate = 0.1 # default is 10 %
      n_features = int(self.n_in * self.compression_rate)

    else:
      self.compression_rate = self.n_in / n_features

    self.n_features = n_features
    self._callback = callback
    self.layer_name = layer_name

  def get_architecture(self):
    self.input_layer = x = Input(shape=(self.n_in, ),
                                 name="Input")
    x = LinearPass(name="Linear")(x)
    x = self.get_block(x,
                       n_nodes=None,
                       n_layers=self.n_layers,
                       architecture_type=self.params.build.architecture_type)
    self.output_layer = Dense(self.data.n_classes,
                              activation="softmax",
                              name="Output")(x)

  def fit_model(self, **kwargs):
    if not hasattr(self, 'callback'):
      self.set_callback(layer_name = self.layer_name,
                        n_features = self.n_features,
                        callback = self._callback)

    history = self.model.fit(x=self.data.X_train,
                             y=self.data.y_train,
                             epochs=self.params.train.epochs,
                             shuffle=False,
                             batch_size=self.params.train.batch_size,
                             validation_data=(self.data.X_test,
                                              self.data.y_test),
                             callbacks=[self.callback],
                             verbose = True)
    try:
      self._params["callback"] = self.callback[0].trigger_params
    except:
      self._params["callback"] = None

    self.params.train._trained = True

    return history

  def set_callback(self,
                   layer_name,
                   n_features=None,
                   **kwargs):

    if not layer_name:
      raise KeyError("Please provide the layer for the feature "
                     "selection algorithm.")

    else:
      print("Feature Selection Callback is instantiated.\n"
            f"The algorithm tries to find the {self.n_features} "
            "most important features.\n")

    val_data = (self.data.X_train, self.data.y_train)

    self.callback = FeatureSelection(evaluation_data=val_data,
                                     layer_name=layer_name,
                                     n_features=n_features,
                                     callback=kwargs)

  def get_callback(self, layer_name,
                   n_features=None,
                   **kwargs):

    if not hasattr(self, 'callback'):
      callback = self.set_callback(layer_name, n_features=None, **kwargs)

    else:
      callback = self.callback

    return callback

  def get_mask(self):
    return self.callback.log.weights[-1].astype(bool)