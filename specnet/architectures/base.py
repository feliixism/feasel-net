import numpy as np
import keras
from keras.models import Model
from spec_net.data import classification
from spec_net.tfcustom import plot

import os

class Model(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_layer_names(self):
        self.layer_names = []
        for layer in self.layers:
            self.layer_names.append(layer.name)

class Base:
    def __init__(self, X, y = None, **kwargs):
        self.X = X
        self.y = y
                
        #training parameter container
        self._training_params = {"epochs": 100,
                                 "batch_size": 32,
                                 "validation_split": 0.2,
                                 "dropout_rate": 0.5,
                                 "learning_rate": 1e-4,
                                 "loss": "sparse_categorical_crossentropy",
                                 "metric": "accuracy",
                                 "test_split": 0.2,
                                 "normalization": None,
                                 "sample_axis": None,
                                 "input_layer_type": "Dense",
                                 "activation_function": "relu"}
        
        for key in kwargs:
            if key in self._training_params:
                self._training_params[f"{key}"] = kwargs[f"{key}"]
        
        #data container
        self._data = classification.Classification(X, y,
                                                   sample_axis = self._training_params["sample_axis"],
                                                   normalization = self._training_params["normalization"],
                                                   input_layer_type = self._training_params["input_layer_type"])
        
        self._params = {"data": self.data._data_params,
                        "training": self._training_params}
        
        self._plot = plot.Base(self)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(Size of Dataset: {self._data.X.shape}, Number of Samples: {self._data.n_samples}, Number of Classes: {self._data.n_classes})"
        
    @property
    def data(self):
        return self._data
    
    @property
    def plot(self):
        return self._plot
   
    def get_params(self, type = None):
        if type is None:
            return self._params
        else:
            try:
                return self._params[f"{type}"]
            except:
                raise NameError(f"'{type}' is an invalid argument for 'type'.")
        
    def _get_layer_names(self):
        self.layer_names = []
        for layer in self.model.layers:
            self.layer_names.append(layer.name)    
  
    # setters of training parameters
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
        self._training_params["epochs"] = epochs
    
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
        self._training_params["batch_size"] = batch_size
        
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
        self._training_params["validation_split"] = validation_split
    
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
        self._training_params["dropout_rate"] = dropout_rate
        
    def set_learning_rate(self, learning_rate):
        """
        Sets ratio between activated and inactive nodes.

        Parameters
        ----------
        learning_rate : float
            Learning rate.

        Returns
        -------
        None.

        """
        self._training_params["learning_rate"] = learning_rate

# getters
    def get_layer(self, layer_name):
        """
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
        layer : TYPE
            DESCRIPTION.

        """
        try:
            layer = self.model.get_layer(layer_name)
        
        except NameError:
            print(f"Could not find layer '{layer_name}'.")
        
        return layer

    def get_weights(self, layer_name):
        """
        Parameters
        ----------
        layer_name : str
            Defines name of layer.

        Returns
        -------
        weights : np-array
            Weights of trained model at given layer.
        bias : np-array
            Bias of trained model at given layer.

        """
        layer = self.get_layer(layer_name)
        
        weights = layer.get_weights()[0]
        if layer.use_bias == True:
            bias = layer.get_weights()[1]
            return weights, bias
        else:
            return weights

    def get_model(self):
        inputs = self.input_layer
        outputs = self.output_layer
        self.model = Model(inputs = inputs, outputs = outputs, name = f"ANN ({self._building_params[f'architecture_type']})")
        return self.model
    
    def compile_model(self):
        self.get_model()
        optimizer = keras.optimizers.adam(learning_rate = self._training_params["learning_rate"])
        self.model.compile(optimizer = optimizer, 
                           loss = self._training_params["loss"], 
                           metrics = [self._training_params["metric"]])
        self._get_layer_names()
        
    def fit_model(self):
        self.compile_model()
        x = self.data.X_train
        y = self.data.y_train
        history = self.model.fit(x = x, y = y, 
                                 epochs = self._training_params["epochs"], 
                                 batch_size = self._training_params["batch_size"], 
                                 validation_split = self._training_params["validation_split"])
        return history
    
    def train_model(self):
        if not hasattr(self, 'history'):
            self.model = self.get_model()
            self.compile_model()
            self.history = self.fit_model()
        return self.history
    
    def test_model(self, x_test, y_test = None, model = None):
        if model is None:
            model = self.model
        self.model_type = self.model.name
        self.x_test, self.y_test = self._convert_data(x_test, y_test)
        x_test_length = len(self.x_test)
        if self.model_type == "spectral_net":
            x_test = [self.x_test, self.generator.x_train[0 : x_test_length]]
        y_pred = model.predict(x_test)
        return y_pred

    def save_model(self):
        model_path = f"models/{self.model_type}/{self.architecture_type}/{self.epochs}_epochs_{len(self.x_train[0])}_datasize"
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
        print("Saved model to disk")
    
    def load_model(self, model_path):

        self.model_type = model_path.split("/")[1]
        self.architecture_type = model_path.split("/")[2]
        self.model = keras.models.load_model(model_path)