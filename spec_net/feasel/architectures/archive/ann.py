
from .base import Base
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from ..tfcustom import losses, callbacks
from ..tfcustom.layers import BinarizeDense, LinearPass

class ANN(Base):
    def __init__(self, X, y, architecture_type = "down"):
        """
        Builds an ann with only dense layers. Subclass of 'SpectralNeuralNet'.

        Parameters
        ----------
        X : np-array
            Input for the ann.
        y : np-array
            Output of the ann.
        loss_weight : float, optional
            Determines the impact of corresponding loss value. The default is 
            None.

        Returns
        -------
        Class object.

        """
        super().__init__(X, y)
        self.information = self._information()
        self.model_type = "ann"
        self.architecture_type = architecture_type
        self.loss = "sparse_categorical_crossentropy"
        self.metric = "accuracy"
        self.x_train, self.x_test, self.y_train, self.y_test = self._convert_data(X, y, True)
        self.n_samples, self.n_dimensions = self.X.shape
        self.set_n_layers()
        self.get_architecture()
    
    def _information(self):
        dict = {
            "architecture_types": {"down": self.get_block_down, 
                                   "up-down": self.get_block_up_down, 
                                   "const": self.get_block_const, 
                                   "exp-down": self.get_block_exp_down}
            }
        return dict
    
    def set_n_layers(self, n_layers = 3):
        self.n_layers = n_layers
        self.get_architecture()
        
    def get_block(self, x, n_nodes, n_layers, architecture_type):
        if self.architecture_type not in self.information["architecture_types"]:
            raise TypeError(f"The type '{self.architecture_type}' is not implemented. ",
                            "Try another 'architecture_type'.")
        else:
            return self.information["architecture_types"].get(self.architecture_type)(x, n_nodes, n_layers)
        
    def get_architecture(self):
        self.input_layer = x = Input(shape = (self.n_dimensions, ), 
                                     name = "ClassifierIn")
        x = self.get_block(x, n_nodes = None, n_layers = self.n_layers, 
                           architecture_type = self.architecture_type)
        self.output_layer = Dense(self.n_labels, activation = "softmax", 
                                  name = "Classifier")(x)
    
    def get_block_exp_down(self, x, n_nodes = None, n_layers = 3):
        if n_nodes is None:
            n_nodes = self.n_dimensions
        for i in range(n_layers):
            x = Dense(int(n_nodes / 2 ** (1 + i)), activation = "relu", 
                      name = f"ClassifierDense{i}")(x)
            # x = BatchNormalization(momentum = 0.2, name = f"ClassifierBatchNorm{i}")(x)
            # x = Dropout(0.5)(x)
        return x
    
    def get_block_down(self, x, n_nodes = None, n_layers = 3):
        if n_nodes is None:
            n_nodes = self.n_dimensions
        for i in range(n_layers):
            x = Dense(int(n_nodes / (2 * (i + 1))), activation = "relu", 
                      name = f"ClassifierDense{i}")(x)
            # x = BatchNormalization(momentum = 0.2, name = f"ClassifierBatchNorm{i}")(x)
            # x = Dropout(0.5)(x)
        return x
    
    def get_block_const(self, x, n_nodes = None, n_layers = 3):
        if n_nodes is None:
            n_nodes = self.n_dimensions
        for i in range(0, n_layers):
            x = Dense(n_nodes, activation = "relu", 
                      name = f"ClassifierDense{i}")(x)
            # x = Dropout(0.25)(x)
        return x
    
    def get_block_up_down(self, x, n_nodes = None, n_layers = 3):
        if n_nodes is None:
            n_nodes = self.n_dimensions
        for i in range(n_layers):
            nodes = n_nodes * (2 * (i + 1))
            x = Dense(nodes , activation = "relu", 
                      name = f"ClassifierDense{i}")(x)
            # x = Dropout(0.5)(x)
        for i in range(n_layers):
            x = Dense(nodes / (2 * (i + 1)))
            # x = Dropout(0.5)(x)
        return x
    
    def fit_model(self, *class_objects):
        x = [self.x_train]
        y = [self.y_train]
        for i in range(len(class_objects)):
            x.append(class_objects[i].x_train)
            y.append(class_objects[i].y_train)
        history = self.model.fit(x = self.x_train, y = self.y_train, epochs = self.epochs, 
                                 shuffle = False, batch_size = self.batch_size, 
                                 validation_data = (self.x_test, self.y_test),
                                 verbose = True)
        return history

class FeatureSelectionANN(ANN):
    def __init__(self, X, y, n_features = None, thresh = 0.9, 
                 delta = 25, delta_max = 200, architecture_type = "down"):
        super().__init__(X, y, architecture_type)
        if n_features is None:
            n_features = int(self.n_dimensions / 10)
        self.n_features = n_features
        self.threshold = thresh
        self.delta = delta
        self.delta_max = delta_max
    
    def get_architecture(self):
        self.input_layer = x = Input(shape = (self.n_dimensions, ), 
                                     name = "ClassifierIn")
        x = LinearPass(name = "Linear")(x)
        x = self.get_block(x, n_nodes = None, n_layers = self.n_layers, 
                           architecture_type = self.architecture_type)
        self.output_layer = Dense(self.n_labels, activation = "softmax", 
                                  name = "Classifier")(x)
    
    def fit_model(self, *class_objects):
        X = [self.X]
        y = [self.y]
        for i in range(len(class_objects)):
            X.append(class_objects[i].X)
            y.append(class_objects[i].y)
        self.callback = self.set_callback(layer = "Linear")
        history = self.model.fit(x = self.x_train, y = self.y_train, epochs = self.epochs, 
                                 shuffle = False, batch_size = self.batch_size, 
                                 validation_data = (self.x_test, self.y_test),
                                 callbacks = self.callback, verbose = False)
        return history
    
    def set_callback(self, layer = None):
        if layer is None:
            self.callback = None
        else:
            print("Feature Selection Callback is instantiated.",
                  f"The algorithm tries to find the {self.n_features} most important features.")
            self.callback = [callbacks.FeatureSelection(layer = layer, 
                                                        n_features = self.n_features, 
                                                        threshold = self.threshold, 
                                                        delta = self.delta,
                                                        delta_max = self.delta_max,
                                                        kill_rate = 0.2,
                                                        reduction_type = "linear",
                                                        metric = "accuracy")]
        return self.callback