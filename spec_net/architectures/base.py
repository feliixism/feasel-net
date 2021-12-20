import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from ..preprocess.classification import DataContainer
from ..plot import DNN
from ..tfcustom.utils._params import Params
from ..utils.time import get_timestamp

import os

class Model(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_layer_names(self):
        self.layer_names = []
        for layer in self.layers:
            self.layer_names.append(layer.name)

class Base:
    def __init__(self,
                 X, y = None, 
                 features = None,
                 name = None,
                 **kwargs):
        self.X = X
        self.y = y
        if features is not None:
            self.features = np.array(features)
        else:
            self.features = features
        
        self.name = name
                
        # parameter container with train, build and data parameters
        self.params = Params()
        
        for key in kwargs:
            if key in self.params.train.__dict__.keys():
                self.params.train.update(key)(kwargs[key])
            elif key in self.params.build.__dict__.keys():
                self.params.build.update(key)(kwargs[key])
            elif key in self.params.data.__dict__.keys():
                self.params.data.update(key)(kwargs[key])
            else:
                raise KeyError(f"'{key}' is not a valid key for the generic "
                               "neural network useage.")
        
        #data container
        self._data = DataContainer(X, y, 
                                   features,
                                   sample_axis=self.params.data.sample_axis,
                                   normalization=self.params.data.normalization,
                                   input_layer=self.params.data.input_layer)
        
        self.time = get_timestamp()
        
    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(Size of Dataset: {self._data.X.shape}, "
                f"Number of Samples: {self._data.n_samples}, "
                f"Number of Classes: {self._data.n_classes})")
        
    @property
    def data(self):
        return self._data
    
    @property
    def plot(self):
        return self._plot
   
    def get_params(self, type = None):
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
        
    def _get_layer_names(self):
        self.layer_names = []
        for layer in self.model.layers:
            self.layer_names.append(layer.name)    
  
    # setters
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
        Sets ratio between activated and inactive nodes.

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
        Sets ratio between activated and inactive nodes.

        Parameters
        ----------
        learning_rate : float
            Learning rate.

        Returns
        -------
        None.

        """
        try:
            if optimizer.__module__ == 'keras.optimizers':
                return
        except:
            self.params.train.set_optimizer(optimizer)

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
                           outputs=outputs)
        #sets plot functions (only available when model instantiated)
        self._plot = DNN(self)
        return self.model
    
    def compile_model(self):
        """
        Compiles the generically built model.

        Returns
        -------
        None.

        """
        
        self.get_model()
        self.model.compile(optimizer=self.params.train.optimizer, 
                           loss=self.params.train.loss, 
                           metrics=[self.params.train.metric])
        self._get_layer_names()
        
    def fit_model(self):
        """
        Fits the generically built model.

        Returns
        -------
        history : Model history
            Training history of the neural network.

        """
        self.compile_model()
        x = self.data.X_train
        y = self.data.y_train
        history = self.model.fit(x=x, 
                                 y=y, 
                                 epochs=self.params.train.epochs, 
                                 batch_size=self.params.train.batch_size, 
                                 validation_split=self.params.train.validation_split)
        return history
    
    def train_model(self):
        """
        Trains the generically built model. Same functionality as 
        'fit_model()'.

        Returns
        -------
        history : Model history
            Training history of the neural network.

        """
        if not hasattr(self, 'history'):
            self.model = self.get_model()
            self.compile_model()
            self.history = self.fit_model()
        return self.history
    
    def test_model(self, X_test, y_test = None, model = None):
        """
        Tests the generically built model.

        Parameters
        ----------
        X_test : ndarray
            The testdataset.
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
        self.model_type = self.model.name
        self.X_test, self.y_test = self._convert_data(X_test, y_test)
        y_pred = model.predict(X_test)
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