from .base import Base
from .mask_generator import MaskGenerator
from .ann import ANN
from .cnn1d import CNN_1D
from keras.layers import Input, Reshape, Dense, Lambda, Concatenate
import tensorflow as tf
from sklearn.utils import shuffle

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
        
        self.input_layer = signal = self.classifier.input_layer
        
        mask = Reshape(self.input_layer.shape[1 : ])(self.generator.output_layer)
        
        if len(self.input_layer.shape) == 2:
            mask = Lambda(lambda x: tf.expand_dims(x, axis = -1))(mask)
            signal = Lambda(lambda x: tf.expand_dims(x, axis = -1))(signal)
        
        x = Concatenate()([signal, mask])
        
        def masked_input(x):
            signal = x[..., 0]
            mask = mask = x[..., 1]
            kernel = Input(tensor = tf.ones((self.tuning_range, 1, 1)))
            x = tf.cast(tf.greater_equal(mask, self.generator.threshold), tf.float32)
            
            x = Reshape([x.shape[1], 1])(x)
            x = tf.nn.conv1d(x, kernel, stride = 1, padding = "SAME", name = "CONV")
            
            x = Reshape([x.shape[1]])(x)
            x = tf.cast(tf.multiply(signal, x), tf.float32)
            if len(self.input_layer.shape) == 3:
                x = tf.expand_dims(x, axis = -1)
            return x
                    
        x = Lambda(masked_input, name = "MaskedSignal")(x)
        
        x = self.classifier.get_block(x)
        
        self.output_layer = Dense(self.n_labels, activation = "softmax", name = "classifier")(x)
    
    def fit_model(self, *class_objects):
        x = [self.x_train]
        y = [self.y_train]
        for i in range(len(class_objects)):
            x.append(class_objects[i].x_train)
            y.append(class_objects[i].y_train)
        # callback = FeatureMapCallback(x, "Generator")
        history = self.model.fit(x = x, y = y, epochs = self.epochs, batch_size = self.batch_size, validation_split = self.validation_split)#, callbacks = [callback])
        return history