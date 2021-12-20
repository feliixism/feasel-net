import numpy as np
from sklearn.model_selection import train_test_split
from ..preprocess import dataformat as df
from ..tfcustom.utils._params import DataParams

class DataContainer:
    def __init__(self, X, y, features = None, **kwargs):
        """
        This is a data object for the classification with neural networks based
        on the tensorflow and keras frameworks.
        Within this class all necessary pre-processing and data-handling tasks
        can be undertaken.

        Parameters
        ----------
        X : nd-array
            Feature array for the training of neural networks.
        y : nd-array
            Target array for the loss estimation.
        features : nd-array, optional
            Defines the feature's names and must have the same size as the 
            feature shape. If None, the features are iterated from 0 to the 
            number of features - 1. The default is None.
        sample_axis : int, optional
            Defines the sample axis. If None, the algorithm tries to find the 
            sample axis on its own. The default is None.
        input_layer : str, optional
            Defines the shape of the input layer and thus the shape of the
            input data. The possible types are given by the official Keras 
            layer names. The default is None.
        normalization : str, optional
            Defines the normalization type. Possible arguments are 
            'standardize' and 'min_max' for a standardization or min-max 
            scaling, respectively. The default is None.
        
        Raises
        ------
        ValueError
            Number of samples given in 'sample_axis' does not match with 
            samples in given in 'y'. Please change your sample axis.

        Returns
        -------
        Data container object.

        """
        self.X = X
        self.y = y
        self.features = features
        
        self.X_train, self.y_train = X, y
        self.X_test, self.y_test = None, None
        
        #params container
        self.data_params = DataParams()
        
        for key in kwargs:
            if key in self.data_params.__dict__.keys():
                self.data_params.update(key)(kwargs[key])
            else:
                raise KeyError(f"'{key}' is not a valid key for the generic "
                               "neural network useage.")
        
        self._params = {"data": self.data_params}
        
        self.prepare_data()
        
    def __repr__(self):
        return (f"Data(Size of Dataset: {self.X.shape}, "
                f"Number of Samples: {self.n_samples}, "
                f"Number of Classes: {self.n_classes})")
    
    def get_params(self, type = None):
        if type is None:
            return self._params
        else:
            try:
                return self._params[f"{type}"]
            except:
                raise NameError(f"'{type}' is an invalid argument for 'type'.")
    
    def prepare_data(self):
        #get sample axis and number of samples
        if self.y is not None:
            self.classes = np.unique(self.y)
            self.n_classes = len(self.classes)
            self.n_samples = len(self.y)
            
            #get feature shape
            if self.data_params.sample_axis is None:
                sample_axis = int(np.argwhere(np.array(self.X.shape) 
                                              == self.n_samples))
            else:
                sample_axis = self.data_params.sample_axis
            self.feature_shape = np.take(self.X, 0, axis = sample_axis).shape
                        
            #check whether sample axis is chosen right by comparing number of 
            #samples in y and in given axis
            if self.n_samples != len(self.y):
                raise ValueError("Number of samples given in 'sample_axis' "
                                 f"({self.n_samples}) does not match with "
                                 f"samples in given in 'y' ({len(self.y)}). "
                                 "Please change your sample axis.")
        
        else:
            if self.data_params.sample_axis:
                self.n_samples = self.X.shape[sample_axis]
                self.feature_shape = np.take(self.X, 
                                             0, 
                                             axis=sample_axis).shape
            else:
                self.n_samples = self.X.shape[0]
                self.feature_shape = self.X.shape[1:]
            self.classes = []
            self.n_classes = 0
        
        shape = (self.n_samples, ) + self.feature_shape
        self.X = self.X.reshape(shape)
    
        #normalize and transform shape to correct input layer shape
        if self.data_params.normalization:
            if self.data_params.normalization == "standardize":
                self.standardize()
            elif self.data_params.normalization == "min-max":
                self.min_max()
            else:
                raise NameError(f"'{self.data_params.normalization}' is "
                                "not implemented as a normalization " 
                                "technique. Try 'standardize' or 'min_max'.")
        
        if self.data_params.input_layer:
            self.convert_to_input_shape(self.data_params.input_layer)
    
    def convert_to_input_shape(self, input_layer):
        """
        Automatically converts the given dataset to match the requirements of
        the input layers from Keras (e.g. Dense or Convolutional Layers).
        If the dimensionality of the original data is too low, it will be 
        expanded at the previous last dimension. On the other hand, if it is
        too high, the dimensionality will be reduced by multiplying the last
        two dimensions in order to get only one remaining dimension with all
        features encoded in this last dimension.
        The process iterates as long as the dimensionalities do not match.

        Parameters
        ----------
        input_layer : str
            Defines the input layer type. The layer types correspond to the 
            expressions used by Keras.

        Raises
        ------
        NameError
            The input layer is either not a possible keras input layer or not 
            implemented yet. Please try another layer.

        Returns
        -------
        None.

        """
        layer_dims = {"Dense": 2,
                      "Conv1D": 3,
                      "Conv2D": 4,
                      "Conv3D": 5}
        try:
            input_layer_dim = layer_dims[f"{input_layer}"]
            
            while True:
                X_dim = self.X.ndim    
                
                # expansion of dimensionality by adding another dimension 
                # after the previous last dimension
                if input_layer_dim > X_dim:
                    self.X = np.expand_dims(self.X, axis = -1)
                
                # reduction of dimensionality by multiplying the last two 
                # dimensionalities as the new reduced dimension 
                elif input_layer_dim < X_dim:
                    shape = ((self.n_samples, ) + self.X.shape[1:-2] 
                             + (self.X.shape[-2] * self.X.shape[-1], ))
                    self.X = self.X.reshape(shape)
                else:
                    break
            self.feature_shape = self.X.shape[1:]
        
        except:
            raise NameError(f"'{input_layer}' is either not a possible keras "
                            "input layer or not implemented yet. Please try "
                            "one of the following layers: "
                            f"{layer_dims.keys()}.")
    
    def categorical_labels(self):
        self.y_train = df.categorical(self.y_train)
        return self.y_train
    
    def one_hot_labels(self):
        self.y_train = df.one_hot(self.y_train)
        return self.y_train
    
    def train_test_split(self, test_split = 0.25, **kwargs):
        (self.X_train, self.X_test, 
         self.y_train, self.y_test) = train_test_split(self.X_train, 
                                                       self.y_train, 
                                                       test_size=test_split,
                                                       **kwargs)
    
    def standardize(self):
        self.X_train = df.standardize(self.X_train, axis = -1)
    
    def min_max(self, a_max = 1, a_min = 0):
        self.X_train = df.min_max(self.X_train, 
                                  axis=-1, 
                                  a_max=a_max, 
                                  a_min=a_min)
    
        