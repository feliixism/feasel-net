from .base import Base
from keras.layers import Input, Dense, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Dropout

class CNN_1D(Base):
    def __init__(self, x_train, y_train, loss_weight = 1.0):
        """
        Builds an 1d_cnn with several convolutional layers. Subclass of 'SpectralNeuralNet'.

        Parameters
        ----------
        x_train : np-array
            Input for the 1d_cnn.
        y_train : np-array
            Output of the 1d_cnn.
        loss_weight : float, optional
            Determines the impact of corresponding loss value. The default is None.

        Returns
        -------
        Class object.

        """
        super().__init__(x_train, y_train)
        self.model_type = self.architecture_type = "1d_cnn"
        self.loss = "sparse_categorical_crossentropy"
        self.loss_weight = loss_weight
        self.metric = "accuracy"
        self.x_train, self.y_train = self._convert_data(x_train, y_train)
        self.get_architecture()
        
    def get_architecture(self):
        self.input_layer = x = Input(shape = (self.x_train.shape[1], self.x_train.shape[2]), name = "classifier_input")
        
        x = self.get_block(x)

        self.output_layer = Dense(self.n_labels, activation = "softmax", name = "classifier")(x)
    
    def get_block(self, x, n_layers = 3):
        for i in range(1, n_layers + 1):
            x = Conv1D(int(32 / i), 32 * i, activation = "relu", padding = "same", name = f"classifier_conv1d_{i}")(x)
            x = BatchNormalization(momentum = 0.0, name = f"classifier_batch_norm_{i}")(x)
            x = MaxPooling1D(name = f"classifier_maxPool1d_{i}")(x)
            # x = Dropout(self.dropout_rate, name = f"classifier_dropout_{i}")(x)
            
        x = Flatten(name = "classifier_flatten")(x)
        x = Dropout(self.dropout_rate, name = f"classifier_dropout_0")(x)
        x = Dense(self.n_labels * 2, activation='relu', name = "classifier_dense_0")(x)
        return x