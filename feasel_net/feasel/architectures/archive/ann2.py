from .base import Base
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from ..tfcustom import losses, callbacks
from ..tfcustom.layers import BinarizeDense, LinearPass
from spec_net.data import classification
import numpy as np

class DenseDNN(Base):
    def __init__(self, X, y, **kwargs):
        """
        Builds an ANN with only dense layers. Subclass of 'SpectralNeuralNet'.

        Parameters
        ----------
        X : np-array
            Input for the ann.
        y : np-array
            Output of the ann.

        Returns
        -------
        Class object.

        """
        super().__init__(X, y, **kwargs)
        
        # parameter container
        self._building_params = {"architecture_type": "down",
                                 "n_layers": 3,
                                 "n_nodes": None}
        
        for key in kwargs:
            if key in self._building_params:
                self._building_params[f"{key}"] = kwargs[f"{key}"]
                
        # encoding of target
        if self.params.train.loss == "sparse_categorical_crossentropy":
            self.data.sparse_labels()
        elif self.params.train.loss == "categorical_crossentropy":
            self.data.one_hot_labels()
        
        if self.params.train.test_split:
            self.data.train_test_split(self.params.train.test_split, 
                                       random_state = 42)
        
        self._information = self.get_information()
        self.n_in = np.prod(self.data.feature_shape)
        self.set_n_layers(self._building_params["n_layers"])
        self.get_architecture()
        
        self._params = {"data": self.params.data,
                        "training": self.params.train,
                        "building": self.params.build}
    
    def get_information(self):
        dict = {"architecture_types": {"down": self.get_block_down, 
                                       "up-down": self.get_block_up_down, 
                                       "const": self.get_block_const, 
                                       "exp-down": self.get_block_exp_down}
                }
        return dict
    
    def set_n_layers(self, 
                     n_layers = 3):
        
        self.n_layers = n_layers
        self.get_architecture()
        
    def get_block(self, x, n_nodes, n_layers, architecture_type):
        architecture_types = self._information["architecture_types"]
        
        if architecture_type not in architecture_types:
            raise TypeError(f"The type '{architecture_type}' is not " 
                            "implemented. Try another 'architecture_type'.")
        
        else:
            return architecture_types.get(architecture_type)(x, 
                                                             n_nodes, 
                                                             n_layers)
        
    def get_architecture(self):
        architecture_type = self._building_params["architecture_type"]
        n_nodes = self._building_params["n_nodes"]
        n_layers = self._building_params["n_layers"]
        
        self.input_layer = x = Input(shape = (self.n_in, ), 
                                     name = "Input")
        
        x = self.get_block(x, 
                           n_nodes = n_nodes, 
                           n_layers = n_layers, 
                           architecture_type = architecture_type)
        
        self.output_layer = Dense(self.data.n_classes, 
                                  activation = "softmax", 
                                  name = "Output")(x)
    
    def get_block_exp_down(self, x, n_nodes = None, n_layers = 3):
        
        if n_nodes is None:
            n_nodes = self.n_in
        
        for i in range(n_layers):
            x = Dense(int(n_nodes / 2 ** (1 + i)), 
                      activation = self.params.train.activation, 
                      name = f"Dense{i}")(x)
            # x = BatchNormalization(momentum = 0.2, 
            # name = f"ClassifierBatchNorm{i}")(x)
            # x = Dropout(0.25)(x)
        
        return x
    
    def get_block_down(self, x, n_nodes = None, n_layers = 3):
        
        if n_nodes is None:
            n_nodes = self.n_in
        
        for i in range(n_layers):
            x = Dense(int(n_nodes / (2 * (i + 1))), 
                      activation = self.params.train.activation, 
                      name = f"Dense{i}")(x)
            # x = BatchNormalization(momentum = 0.2, 
            # name = f"ClassifierBatchNorm{i}")(x)
            # x = Dropout(0.5)(x)
        
        return x
    
    def get_block_const(self, x, n_nodes = None, n_layers = 3):
        
        if n_nodes is None:
            n_nodes = self.n_in
        
        for i in range(0, n_layers):
            x = Dense(n_nodes, 
                      activation = self.params.train.activation, 
                      name = f"Dense{i}")(x)
            # x = Dropout(0.25)(x)
        
        return x
    
    def get_block_up_down(self, x, n_nodes = None, n_layers = 3):
        
        if n_nodes is None:
            n_nodes = self.n_in
        
        for i in range(n_layers):
            nodes = n_nodes * (2 * (i + 1))
            x = Dense(nodes, 
                      activation = self.params.train.activation, 
                      name = f"Dense{i}")(x)
            # x = Dropout(0.5)(x)
        
        for i in range(n_layers):
            x = Dense(nodes / (2 * (i + 1)),
                      activation = self.params.train.activation, 
                      name = f"Dense{i}")
            # x = Dropout(0.5)(x)
        
        return x
    
    def fit_model(self):
        
        history = self.model.fit(x = self.data.X_train, 
                                 y = self.data.y_train, 
                                 epochs = self.params.train.epochs, 
                                 shuffle = False, 
                                 batch_size =  self.params.train.batch_size, 
                                 validation_data = (self.data.X_test, 
                                                    self.data.y_test),
                                 verbose = True)
        
        return history

class FeaSelDNN(DenseDNN):
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
                                 verbose = False)
        try:
            self._params["callback"] = self.callback[0].trigger_params
        except:
            self._params["callback"] = None
        
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
        
        self.callback = callbacks.FeatureSelection(layer_name=layer_name, 
                                                   n_features=n_features,
                                                   callback=kwargs)
    
    def get_callback(self, layer_name, 
                     n_features=None,
                     **kwargs):
        
        if not hasattr(self, 'callback'):
            callback = callbacks.FeatureSelection(layer_name=layer_name, 
                                                  n_features=n_features,
                                                  callback=kwargs)
        
        else:
            callback = self.callback
        
        return callback
    
    def get_mask(self):
        return