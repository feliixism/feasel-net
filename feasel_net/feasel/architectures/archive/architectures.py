import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Input, Dense, BatchNormalization, Lambda, Concatenate, Reshape, Multiply
from keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os
import spec_net.neural_net.callbacks as cb
from sklearn.utils import shuffle
# from . import activations
from ..preprocess import dataformat as df
from . import layers

class Base:
    def __init__(self, x_train, y_train):
        self.epochs = 20
        self.batch_size = 256 
        self.validation_split = 0.2
        self.dropout_rate = 0.25
        self.labels = np.unique(y_train)
        self.n_labels = len(self.labels)
        self.n_wavelengths = 4
        self.optimizer = keras.optimizers.Adam(lr = 0.001)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(Size of Dataset: {len(self.x_train)}, Number of Labels: {self.n_labels})"

# helpers    
    def _convert_data(self, x, y = 0):
        """
        Converts data shape such that it fits the required input shape for each neural network architecture.

        Parameters
        ----------
        x : np-array
            Input data.
        y : np-array, optional
            Ground truth data, for supervised learning algorithms only. The default is None.

        Returns
        -------
        x, (y) : np-array
            Converted np-array(s).

        """
        if len(x.shape) < 2:
            x = np.expand_dims(x, 0)
        if y is None:
            y = np.empty(len(x))
        else:
            if self.loss == "sparse_categorical_crossentropy":
                y = df.sparse_labels(y)
            # if len(y.shape) < 2:
            #     y = np.expand_dims(y, 0)


        if self.architecture_type == "ann" or self.architecture_type == "mask_generator":
            x = df.shape_ann(x, y)
        elif self.architecture_type == "1d_cnn":
            x = df.shape_1d_cnn(x, y)
        else:
            pass
        if y is not None:
            return x, y
        else:
            return x
    
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
        self.epochs = epochs
    
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
        self.batch_size = batch_size
        
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
        self.validation_split = validation_split
    
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
        self.dropout_rate = dropout_rate

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

# model operations    
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
        bias = layer.get_weights()[1]
        
        return weights, bias 

    def get_model(self, *class_objects):
        inputs = [self.input_layer]
        outputs = [self.output]
        for i in range(len(class_objects)):
            inputs.append(class_objects[i].input_layer)
            outputs.append(class_objects[i].output)
        self.model = Model(inputs = inputs, outputs = outputs, name = self.model_type)
        return self.model
    
    def compile_model(self, *class_objects):
        optimizer = self.optimizer
        losses = [self.loss]
        loss_weights = [self.loss_weight]
        metrics = [self.metric]
        for i in range(len(class_objects)):
            losses.append(class_objects[i].loss)
            loss_weights.append(class_objects[i].loss_weight)
            metrics.append(class_objects[i].metric)
        loss_weights[:] = [i / sum(loss_weights) for i in loss_weights]
        self.model.compile(optimizer = optimizer, loss = losses, loss_weights = loss_weights, metrics = metrics)
        self._get_layer_names()
        
    def fit_model(self, *class_objects):
        x = [self.x_train]
        y = [self.y_train]
        for i in range(len(class_objects)):
            x.append(class_objects[i].x_train)
            y.append(class_objects[i].y_train)
        history = self.model.fit(x = x, y = y, epochs = self.epochs, batch_size = self.batch_size, validation_split = self.validation_split)
        return history
    
    def train_model(self, *class_objects):
        try:
            self.history = self.history
        except:
            self.model = self.get_model(*class_objects)
            self.compile_model(*class_objects)
            self.history = self.fit_model(*class_objects)
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
        self.loss = self.test_loss
        self.loss_weight = loss_weight
        self.metric = []
        self.threshold = 0.05
        self.epochs = 20
        self.n_layers = 3
        y_train = np.zeros(x_train.shape)
        x_size = int(np.floor(y_train.shape[1] ** (1. / self.n_layers)))
        x_train = np.empty([x_train.shape[0], x_size])
        # x_train[:] = np.random.normal(0.5, scale = 10, size = (1, self.n_labels))
        # x_train[:] = np.ones([1, self.n_labels]) * self.threshold
        x_train = np.random.normal(0.5, 0.1, size = [x_train.shape[0], x_size])
        # x_train[:] = np.ones([1, x_size]) * 0.5
        self.x_train, self.y_train = self._convert_data(x_train, y_train)
        self.get_architecture()
        
    def get_architecture(self):        
        self.input_layer = x = Input(shape = (self.x_train.shape[1], ), name = "generator_input")
        
        x = self.get_block(x, self.n_layers)
            
        self.output = layers.QuantizedDense(self.y_train.shape[1], name = "generator")(x)

    def get_block(self, x, n_layers):
        
        for i in range(1, n_layers + 1):
            x = Dense(self.x_train.shape[1] ** i, activation = "sigmoid", name = f"generator_dense_{i}")(x)
        return x    
    
    def test_loss(self, y_true, y_pred):
        #constants (watch out for batch size: loss calculated over whole batch)
        n_wavelengths = tf.constant(self.n_wavelengths, dtype = float) * self.threshold
        max_sd = tf.constant(self.y_train.shape[1], dtype = float)

        def sum_above_threshold(x):
            mask = tf.greater_equal(x, self.threshold)
            
            zeros = tf.zeros(mask.shape)
            ones = tf.zeros(mask.shape)
            
            sat = tf.reduce_sum(tf.where(mask, x, zeros)) * self.threshold
            
            # comp = n_wavelengths - sat
            # 
            # original = tf.abs(1.0 - n_wavelengths / sat)
            
            sat = tf.abs(sat - n_wavelengths)
            sat = sat / tf.abs(max_sd - n_wavelengths)
            return sat
        
        sat = sum_above_threshold(y_pred[0])
        
        return sat
    
    def show_mask(self, i):
        plt.figure(f"Predicted Mask {i}")
        i = np.random.randint(0, len(self.x_train[0]))
        mask = np.array(self.x_train[i], ndmin = 2)
        feature_map = self.model.predict(mask)
        pred = feature_map[0]
        # plt.plot([0, self.y_train.shape[1]], [self.threshold, self.threshold])
        plt.plot(pred)
        # plt.text(10, self.threshold + 0.01, f"Threshold: {self.threshold}")
        # plt.text(10, self.threshold - 0.05, f"Sum: {np.sum(pred)}")        
        plt.ylim(0, 1)
        # plt.xlim(0, self.y_train.shape[1])
        # plt.grid(True)
        plt.show()
    
    def fit_model(self, *class_objects):
        x = [self.x_train]
        y = [self.y_train]
        for i in range(len(class_objects)):
            x.append(class_objects[i].x_train)
            y.append(class_objects[i].y_train)
        callback = cb.MaskCallback(x)
        history = self.model.fit(x = x, y = y, epochs = self.epochs, batch_size = self.batch_size, validation_split = self.validation_split, callbacks = [callback])
        return history

class ANN(Base):
    def __init__(self, x_train, y_train, loss_weight = 1.0):
        """
        Builds an ann with only dense layers. Subclass of 'SpectralNeuralNet'.

        Parameters
        ----------
        x_train : np-array
            Input for the ann.
        y_train : np-array
            Output of the ann.
        loss_weight : float, optional
            Determines the impact of corresponding loss value. The default is None.

        Returns
        -------
        Class object.

        """
        super().__init__(x_train, y_train)
        self.model_type = self.architecture_type = "ann"
        self.loss = "sparse_categorical_crossentropy"
        self.loss_weight = loss_weight
        self.metric = "accuracy"
        self.x_train, self.y_train = self._convert_data(x_train, y_train)
        self.get_architecture()
        
    def get_architecture(self):
        
        self.input_layer = x = Input(shape = (self.x_train.shape[1], ), name = "classifier_input")
        
        x = self.get_block(x)
    
        self.output = Dense(self.n_labels, activation = "softmax", name = "classifier")(x)
    
    def get_block(self, x, n_layers = 3):
        for i in range(1, n_layers + 1):
            x = Dense(int(self.x_train.shape[1] / 2 ** (2*i)), activation = "relu", name = f"classifier_dense_{i}")(x)
            x = BatchNormalization(momentum = 0.0, name = f"classifier_batch_norm_{i}")(x)
        return x
    
from keras.layers import Conv1D, Flatten, MaxPooling1D, Dropout

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

        self.output = Dense(self.n_labels, activation = "softmax", name = "classifier")(x)
    
    def get_block(self, x, n_layers = 3):
        for i in range(1, n_layers + 1):
            x = Conv1D(int(12 / i), 3 * i, activation = "relu", padding = "same", name = f"classifier_conv1d_{i}")(x)
            x = BatchNormalization(momentum = 0.0, name = f"classifier_batch_norm_{i}")(x)
            x = MaxPooling1D(name = f"classifier_maxPool1d_{i}")(x)
            # x = Dropout(self.dropout_rate, name = f"classifier_dropout_{i}")(x)
            
        x = Flatten(name = "classifier_flatten")(x)
        x = Dropout(self.dropout_rate, name = f"classifier_dropout_0")(x)
        x = Dense(self.n_labels * 2, activation='relu', name = "classifier_dense_0")(x)
        return x

class SpectralNet(Base):
    def __init__(self, x_train, y_train, classifier_type, tuning_range = 3, classifier_loss = 1.0, generator_loss = 1.0):
        """
        Builds a classifier network with the given classifier type. Its inputs are masked by a trainable automatically generated binary mask.
        Subclass of 'SpectralNeuralNet'.

        Parameters
        ----------
        x_train : np-array
            Input for the mask_generator.
        y_train : np-array
            Output of the mask_generator.
        classifier_type : str
            States the classifier architecture type.
        classifier_loss : float, optional
            Determines the impact of corresponding classifier loss value. The default is 1.0.
        generator_loss : float, optional
            Determines the impact of corresponding generator loss value. The default is 1.0.

        Returns
        -------
        Class object.

        """
        super().__init__(x_train, y_train)
        
        self.model_type = "spectral_net"
        self.generator = MaskGenerator(x_train, y_train, generator_loss)
        self.tuning_range = tuning_range
        
        if classifier_type == "ann":
            self.classifier = ANN(x_train, y_train, classifier_loss)
        elif classifier_type == "1d_cnn":
            self.classifier = CNN_1D(x_train, y_train, classifier_loss)
        
        self.x_train, self.y_train = self.classifier.x_train, self.classifier.y_train
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        
        self.loss = "sparse_categorical_crossentropy"
        self.loss_weight = classifier_loss
        self.metric = "accuracy"
        self.architecture_type = self.classifier.architecture_type
        
        self.get_architecture()        
    
    def get_architecture(self):
        
        self.input_layer = self.classifier.input_layer
        mask = self.generator.output
        if len(self.input_layer.shape) == 3:
            mask = Reshape((mask.shape[1], 1), name = "reshape_mask")(mask)
        self.test = Multiply(name = "mask_classifier_input")([self.input_layer, mask])
        
        
        self.output = Dense(self.n_labels, activation = "softmax", name = "classifier")(self.test)
    
    def fit_model(self, *class_objects):
        x = [self.x_train]
        y = [self.y_train]
        for i in range(len(class_objects)):
            x.append(class_objects[i].x_train)
            y.append(class_objects[i].y_train)
        callback = cb.FeatureMapCallback(x, "generator")
        history = self.model.fit(x = x, y = y, epochs = self.epochs, batch_size = self.batch_size, validation_split = self.validation_split, callbacks = [callback])
        return history
    
    def show_generated_mask(self):
        random_pos = np.random.randint(0, len(self.x_train))
        
        label = self.labels[self.y_train[random_pos]]
        
        if self.architecture_type == "ann":
            signal = np.array(self.x_train[random_pos], ndmin = 2)
        else:
            signal = np.array(self.x_train[random_pos], ndmin = 3)
        
        mask = np.array(self.generator.x_train[random_pos], ndmin = 2)
        
        try:
            model = Model(inputs = self.model.inputs, outputs = self.model.get_layer("mask_classifier_input").output)
            feature_map = model.predict(x = [signal, mask])
        except:
            raise NameError(f"Layer 'mask_classifier_input' is not a part of model '{self.model_type}'")
        
        # masked_signal = feature_map[0][:, 0]
        
        try:
            model = Model(inputs = self.model.inputs, outputs = self.model.get_layer("generator").output)
            feature_map = model.predict(x = [signal, mask])
        except:
            raise NameError(f"Layer 'generator' is not a part of model '{self.model_type}'")
        
        mask = feature_map[0]
        
        x = np.round(np.linspace(4000, 400, 3600), 0)
        
        plt.figure(f"Masked Signal of {label}")
        
        plt.subplot(2,1,1)
        # plt.plot(x, masked_signal, color = "gray", label = "masked_signal")
        plt.plot(x, signal[0][:, 0], color = "k", label = "signal")
        plt.plot([1500, 1500], [0, 100], color = "r")
        plt.plot([500, 500], [0, 100], color = "r")
        plt.ylim(0, 100)
        plt.xlim(x[0], x[-1])
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2,1,2)
        plt.plot(x, mask, color = "gray", label = "mask")
        plt.plot([x[0], x[-1]], [self.generator.threshold, self.generator.threshold], linestyle = "-.", color = "k", label = "threshold")
        plt.plot([1500, 1500], [0, 1], color = "r")
        plt.plot([500, 500], [0, 1], color = "r")
        plt.text(1000, 0.05, "Fingerprint Region", horizontalalignment='center', color = "r")
        plt.ylim(0, 1.1)
        plt.xlim(x[0], x[-1])
        plt.xlabel("Wavenumber [cm$^{-1}$]")
        plt.ylabel("Transmittance [%]")
        plt.legend()
        plt.grid(True)
        
        return feature_map, signal