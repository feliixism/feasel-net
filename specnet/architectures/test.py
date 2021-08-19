from .base import Base
from ..tfcustom import losses, callbacks
from ..tfcustom.layers import BinarizeDense, LinearPass

import numpy as np
from keras.layers import Input, Dense, Dropout

class ANN(Base):
    def __init__(self, x_train, y_train, loss_weight = 1.0):
        """
        Builds a generator block with several dense layers. Subclass of 'SpectralNeuralNet'.

        Parameters
        ----------
        x_train : np-array
            Input for the mask_generator.
        y_train : np-array
            Output of the mask_generator.
        loss_weight : float, optional
            Determines the impact of corresponding loss value. The default is None.

        Returns
        -------
        Class object.

        """
        super().__init__(x_train, y_train)
        self.model_type = self.architecture_type = "ann"
        self.threshold = 0.5
        self.loss = "sparse_categorical_crossentropy"
        self.loss_weight = loss_weight
        self.metric = ["accuracy"]
        self.epochs = 20
        self.n_layers = 3
        self.x_train, self.y_train = self._convert_data(x_train, y_train)
        self.n_features = self.x_train.shape[1]
        self.get_architecture()
        self.callback = None
        
        
    def get_architecture(self):        
        self.input_layer = Input(shape = (self.n_features, ), name = "GeneratorIn")
        
        x = LinearPass(name = "Linear")(self.input_layer)
        
        x = Dense(int(self.n_features/2), activation = "relu", name = "Dense1")(x)
        
        x = Dense(int(self.n_features/4), activation = "relu", name = "Dense2")(x)
        
        # x = Dropout(0.5, name = "Dropout")(x)
        
        x = Dense(int(self.n_features/6), activation = "relu", name = "Dense3")(x)
        
        x = Dense(int(self.n_features/8), activation = "relu", name = "Dense4")(x)
        
        self.output_layer = Dense(2, activation = "softmax", name = "Classifier")(x)
    
    def fit_model(self, *class_objects):
        x = [self.x_train]
        y = [self.y_train]
        for i in range(len(class_objects)):
            x.append(class_objects[i].x_train)
            y.append(class_objects[i].y_train)
        self.callback = self.set_callback(layer = "Linear")
        history = self.model.fit(x = x, y = y, epochs = self.epochs, shuffle = True, batch_size = self.batch_size, validation_split = self.validation_split, callbacks = self.callback, verbose = True)
        return history
    
    def set_callback(self, layer = None, n_nodes = 4, acc_threshold = 0.9, min_interval = 25, kill_rate = 0.2, metric = "accuracy"):
        if layer is None:
            self.callback = None
        else:
            self.callback = [callbacks.InputReduction(layer = layer, n_nodes = n_nodes, threshold = acc_threshold, interval = min_interval, kill_rate = kill_rate, metric = metric)]
            return self.callback