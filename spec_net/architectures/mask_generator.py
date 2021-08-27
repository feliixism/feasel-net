from .base import Base
from ..tfcustom import losses
from ..tfcustom.layers import BinarizeDense, LinearPass

import numpy as np
from keras.layers import Input, Dense

class MaskGenerator(Base):
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
        self.model_type = self.architecture_type = "mask_generator"
        self.threshold = 0.05
        self.loss = losses.ReduceWavelengths(self.n_wavelengths, self.threshold)
        self.loss_weight = loss_weight
        self.metric = []
        self.epochs = 20
        self.n_layers = 3
        y_train = np.ones(x_train.shape)
        x_size = int(np.floor(y_train.shape[1] ** (1. / self.n_layers)))
        x_train = np.empty([x_train.shape[0], x_size])
        # x_train[:] = np.random.normal(0.5, scale = 10, size = (1, self.n_labels))
        # x_train[:] = np.ones([1, self.n_labels]) * self.threshold
        # x_train = np.random.normal(self.threshold, 0.1, size = [x_train.shape[0], x_size])
        x_train[:] = np.ones([1, x_size]) * 0.5
        self.x_train, self.y_train = self._convert_data(x_train, y_train)
        self.get_architecture()
        
    def get_architecture(self):        
        self.input_layer = x = Input(shape = (self.x_train.shape[1], ), name = "GeneratorIn")
        
        x = self.get_block(x, self.n_layers)
            
        self.output_layer = Dense(self.y_train.shape[1], activation = "sigmoid", kernel_constraint = "nonneg", name = "Generator")(x)

    def get_block(self, x, n_layers):
        
        for i in range(1, n_layers + 1):
            x = Dense(self.x_train.shape[1] ** i, activation = "sigmoid", kernel_constraint = "nonneg", name = f"GeneratorDense{i}")(x)
        return x