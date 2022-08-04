from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, Dense, InputSpec
from tensorflow.keras.constraints import NonNeg
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant
import tensorflow as tf
import numpy as np

class LinearPass(Layer):
    def __init__(self,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='ones',
                 bias_initializer='ones',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=NonNeg(),
                 bias_constraint=NonNeg(),
                 trainable=False,
                 name=None,
                 **kwargs):
        super(LinearPass, self).__init__(trainable = trainable, name = name, **kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, ),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    # @tf.autograph.experimental.do_not_convert
    def call(self, inputs):
        output = inputs * self.kernel
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        return tuple(input_shape)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }
        base_config = super(LinearPass, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinarizeDense(Dense):
    def __init__(self,
                 units,
                 threshold,
                 tuning_range,
                 kernel_initializer='glorot_uniform',
                 kernel_constraint=NonNeg(),
                 bias_initializer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.threshold=threshold
        self.tuning_range=tuning_range
        self.kernel_initializer=kernel_initializer
        self.kernel_constraint=kernel_constraint
        if bias_initializer is None:
            self.bias_initializer=Constant(self.threshold)
        else:
            self.bias_initializer=bias_initializer
        super(BinarizeDense, self).__init__(units,
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_constraint=self.kernel_constraint,
                                            bias_initializer=self.bias_initializer,
                                            **kwargs)

    def call(self, inputs):
        mask=K.dot(inputs, self.kernel)
        if self.use_bias:
            mask = K.bias_add(mask, self.bias, data_format='channels_last')
        mask = self.heaviside(mask)#, threshold=self.threshold, tuning_range=self.tuning_range)
        return mask

    @tf.custom_gradient
    def heaviside(self, x):#, threshold, tuning_range):
        def grad(dy):
            return K.sigmoid(dy) * (1 - K.sigmoid(dy))
        kernel=K.constant(1., shape=[self.tuning_range,1,1])
        binarized_mask=K.expand_dims(K.cast(K.greater_equal(x, self.threshold), dtype=K.floatx()))
        # binarized_mask=K.squeeze(K.conv1d(binarized_mask, kernel, padding='same'), axis=-1)
        binarized_mask=K.squeeze(binarized_mask,axis=-1)
        return binarized_mask, grad

    def get_config(self):
        config = {'threshold': self.threshold,
                 'tuning_range': self.tuning_range}
        base_config = super(BinarizeDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BinaryDense(Dense):
    """Quantized Dense layer
    References:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1"
    [http://arxiv.org/abs/1602.02830]
    """
    def __init__(self,
                 units,
                 H=1.,
                 kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier=None,
                 **kwargs):
        super(BinaryDense, self).__init__(units, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier

        super(BinaryDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot H: {}'.format(self.H))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(
              1. / np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot learning rate multiplier: {}'.format(self.kernel_lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier,
                                   self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.output_dim,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        binary_kernel = self.quantize(self.kernel, H=self.H)
        output = K.dot(inputs, binary_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(BinaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def quantize(self, W, H=[0.,1.], Q=2):
        """The weights' quantization function

        Derived from original binarize
        CHANGES: - Name changed
                 - Now uses quantized_sigmoid() instead of binary_tanh().
                 - New input Q (allowed levels of output)
                   This keeps the weights in [0, H] with Q levels between

        # Reference:
        "BinaryNet: Training Deep Neural Networks with Weights and
        Activations Constrained to +1 or -1, Courbariaux et al. 2016"
        [http://arxiv.org/abs/1602.02830]
        """
        # [0, H] -> 0 or H
        span = H[1] - H[0]
        W = W - H[0]
        if Q == 'cont':
            Wb = span * self.linear_saturation(W / span)
        else:
            Wb = span * self.quantized_linear(W / span, Q)

        return Wb + H[0]

    def linear_saturation(self, x):
        """Based on _hard_sigmoid(). Basically a linear function saturating
        at 0 and 1.

        # Reference:
        "BinaryNet: Training Deep Neural Networks with Weights and
        Activations Constrained to +1 or -1, Courbariaux et al. 2016"
        [http://arxiv.org/abs/1602.02830]
        """
        return K.clip(x, 0, 1)

    def quantized_linear(self,x, Q):
        """Binary hard sigmoid for training binarized neural network.

        # Reference:
        "BinaryNet: Training Deep Neural Networks with Weights and
        Activations Constrained to +1 or -1, Courbariaux et al. 2016"
        [http://arxiv.org/abs/1602.02830]
        """
        return self.quantize_through(self.linear_saturation(x), Q)

    def quantize_through(self, x, Q):
        """Element-wise quantizing to the closest quantization level with
        full gradient propagation.  A trick from [Sergey Ioffe]
        (http://stackoverflow.com/a/36480182)
        """
        x = x * (Q-1)
        # Round to nearest int
        rounded = K.round(x)
        # Return "rounded" in forward prop because stop_gradient lets
        # its input pass but returns real x in backprop because
        # stop_gradient returns zero for gradient calculations.
        return (x + K.stop_gradient(rounded - x)) / (Q-1)