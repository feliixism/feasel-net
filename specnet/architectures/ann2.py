from spec_net.utils.syntax import update_kwargs
from .base import Base
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from ..tfcustom import losses, callbacks
from ..tfcustom.layers import BinarizeDense, LinearPass
from spec_net.data import classification
import numpy as np

class ANN(Base):
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
        
        #parameter container
        self._building_params = {"architecture_type": "down",
                                 "n_layers": 3,
                                 "n_nodes": None}
        
        for key in kwargs:
            if key in self._building_params:
                self._building_params[f"{key}"] = kwargs[f"{key}"]
                
        #encoding of target
        if self._training_params["loss"] == "sparse_categorical_crossentropy":
            self.data.sparse_labels()
        elif self._training_params["loss"] == "categorical_crossentropy":
            self.data.one_hot_labels()
        
        if self._training_params["test_split"]:
            self.data.train_test_split(self._training_params["test_split"], random_state = 42)
        
        self._information = self.get_information()
        self._n_features = np.prod(self.data.feature_shape)
        self.set_n_layers(self._building_params["n_layers"])
        self.get_architecture()
        
        self._params = {"data": self.data._data_params,
                        "training": self._training_params,
                        "building": self._building_params}
    
    def get_information(self):
        dict = {"architecture_types": {"down": self.get_block_down, 
                                       "up-down": self.get_block_up_down, 
                                       "const": self.get_block_const, 
                                       "exp-down": self.get_block_exp_down}
                }
        return dict
    
    def set_n_layers(self, n_layers = 3):
        self.n_layers = n_layers
        self.get_architecture()
        
    def get_block(self, x, n_nodes, n_layers, architecture_type):
        
        if self._building_params["architecture_type"] not in self._information["architecture_types"]:
            raise TypeError(f"The type '{self._building_params['architecture_type']}' is not implemented. ",
                            "Try another 'architecture_type'.")
        
        else:
            return self._information["architecture_types"].get(architecture_type)(x, n_nodes, n_layers)
        
    def get_architecture(self):
        
        self.input_layer = x = Input(shape = (self._n_features, ), 
                                     name = "Input")
        
        x = self.get_block(x, n_nodes = self._building_params["n_nodes"], 
                           n_layers = self._building_params["n_layers"], 
                           architecture_type = self._building_params["architecture_type"])
        
        self.output_layer = Dense(self.data.n_classes, activation = "softmax", 
                                  name = "Output")(x)
    
    def get_block_exp_down(self, x, n_nodes = None, n_layers = 3):
        
        if n_nodes is None:
            n_nodes = self._n_features
        
        for i in range(n_layers):
            x = Dense(int(n_nodes / 2 ** (1 + i)), 
                      activation = self._training_params["activation_function"], 
                      name = f"Dense{i}")(x)
            # x = BatchNormalization(momentum = 0.2, name = f"ClassifierBatchNorm{i}")(x)
            # x = Dropout(0.25)(x)
        
        return x
    
    def get_block_down(self, x, n_nodes = None, n_layers = 3):
        
        if n_nodes is None:
            n_nodes = self._n_features
        
        for i in range(n_layers):
            x = Dense(int(n_nodes / (2 * (i + 1))), 
                      activation = self._training_params["activation_function"], 
                      name = f"Dense{i}")(x)
            # x = BatchNormalization(momentum = 0.2, name = f"ClassifierBatchNorm{i}")(x)
            # x = Dropout(0.5)(x)
        
        return x
    
    def get_block_const(self, x, n_nodes = None, n_layers = 3):
        
        if n_nodes is None:
            n_nodes = self._n_features
        
        for i in range(0, n_layers):
            x = Dense(n_nodes, 
                      activation = self._training_params["activation_function"], 
                      name = f"Dense{i}")(x)
            # x = Dropout(0.25)(x)
        
        return x
    
    def get_block_up_down(self, x, n_nodes = None, n_layers = 3):
        
        if n_nodes is None:
            n_nodes = self._n_features
        
        for i in range(n_layers):
            nodes = n_nodes * (2 * (i + 1))
            x = Dense(nodes, 
                      activation = self._training_params["activation_function"], 
                      name = f"Dense{i}")(x)
            # x = Dropout(0.5)(x)
        
        for i in range(n_layers):
            x = Dense(nodes / (2 * (i + 1)),
                      activation = self._training_params["activation_function"], 
                      name = f"Dense{i}")
            # x = Dropout(0.5)(x)
        
        return x
    
    def fit_model(self):
        
        history = self.model.fit(x = self.data.X_train, y = self.data.y_train, 
                                 epochs = self._training_params["epochs"], 
                                 shuffle = False, 
                                 batch_size = self._training_params["batch_size"], 
                                 validation_data = (self.data.X_test, self.data.y_test),
                                 verbose = True)
        
        return history

class FSANN(ANN):
    def __init__(self, X, y, n_features = None, callback = None, 
                 layer = "Linear", **kwargs):
        super().__init__(X, y, **kwargs)
        if n_features is None:
            n_features = int(self._n_features / 10)
        self.n_features = n_features
        self.callback = callback
        self.layer = layer
    
    def get_architecture(self):
        self.input_layer = x = Input(shape = (self._n_features, ), 
                                     name = "Input")
        x = LinearPass(name = "Linear", trainable=True, kernel_regularizer="l1")(x)
        x = self.get_block(x, n_nodes = None, n_layers = self.n_layers, 
                           architecture_type = self._building_params["architecture_type"])
        self.output_layer = Dense(self.data.n_classes, activation = "softmax", 
                                  name = "Output")(x)
    
    def fit_model(self, **kwargs):
        
        if self.callback is None:
            self.callback = self.set_callback(layer = self.layer, 
                                              n_features = self.n_features)
        else:
            self.callback = self.set_callback(layer = self.layer, 
                                              n_features = self.n_features,
                                              callback = self.callback)
        
        history = self.model.fit(x = self.data.X_train, y = self.data.y_train, 
                                 epochs = self._training_params["epochs"], 
                                 shuffle = False, 
                                 batch_size = self._training_params["batch_size"], 
                                 validation_data = (self.data.X_test, self.data.y_test),
                                 callbacks = self.callback, verbose = False)
        try:
            self._params["callback"] = self.callback[0].trigger_params
        except:
            self._params["callback"] = None
        
        return history
    
    def set_callback(self, layer = "Linear", n_features = None, **kwargs):
        
        if layer is None:
            return
        
        else:
            print("Feature Selection Callback is instantiated.",
                  f"The algorithm tries to find the {self.n_features}",
                  "most important features.")
            self.callback = [callbacks.FeatureSelection(layer = layer, 
                                                        n_features = n_features,
                                                        **kwargs)]
        
        return self.callback