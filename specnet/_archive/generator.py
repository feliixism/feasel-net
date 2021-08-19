import numpy as np
from keras.layers import Input, Dense
from .base import Base
from .layers import BinarizeDense
from .losses import ReduceWavelengths
from .callbacks import FeatureMap

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
        self.metric = []
        self.threshold = 0.5
        self.loss = ReduceWavelengths(n_wavelengths = self.n_wavelengths, threshold = self.threshold)
        self.loss_weight = loss_weight
        self.epochs = 20
        self.n_layers = 3
        y_train = np.zeros(x_train.shape)
        y_train[30]=1
        x_size = int(np.floor(y_train.shape[1] / self.n_layers))
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
        # self.output_layer = Dense(3600, name = "Generator")(x)
        self.output_layer = BinarizeDense(self.y_train.shape[1], threshold = self.threshold, tuning_range = 3, name = "Generator")(x)#), kernel_constraint = MinMaxClip())(x)
        
    def get_block(self, x, n_layers):
        
        for i in range(2, n_layers):
            x = Dense(self.x_train.shape[1] * i, activation = "sigmoid", use_bias = True, name = f"GeneratorDense{i-1}")(x)#, kernel_constraint = MinMaxClip())(x)
        return x
    
    def fit_model(self, *class_objects):
        x = [self.x_train]
        y = [self.y_train]
        for i in range(len(class_objects)):
            x.append(class_objects[i].x_train)
            y.append(class_objects[i].y_train)
        self.fm_callback = FeatureMap(x) 
        history = self.model.fit(x = x, y = y, epochs = self.epochs, batch_size = self.batch_size, callbacks = [self.fm_callback], validation_split = self.validation_split)
        return history