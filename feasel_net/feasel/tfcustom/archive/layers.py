from keras.layers import Layer, Input, Dense, Conv2D, InputSpec
from keras.utils import conv_utils
from keras.constraints import Constraint
import keras.backend as K
from keras.initializers import RandomUniform
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import initializers, regularizers, constraints
import tensorflow as tf
import numpy as np

class MaskDense(Layer):
    def __init__(self, 
                 units,
                 threshold, 
                 tuning_range,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='ones',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MaskDense, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.threshold = threshold
        self.tuning_range = tuning_range
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.supports_masking = True
        # self.input_spec = InputSpec(min_ndim=2)
    
    #could be interesting if loss function of classifier and generator do not work in unison: initiallization of bias maybe?
    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A merge layer should be called '
                              'on a list of at least 2 inputs. '
                              'Got ' + str(len(input_shape)) + ' inputs.')
        batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
        if len(batch_sizes) > 1:
            raise ValueError(
                'Can not merge tensors with different '
                'batch sizes. Got tensors with shapes : ' + str(input_shape))
        if input_shape[0] is None:
            output_shape = None
    
    # from dense layer-class
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `MaskDense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape[0])
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `MaskDense` '
                              'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=False)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        signal, mask = inputs
        kernel = K.constant(1., dtype = 'float32', shape=(self.tuning_range, 1, 1))
        mask = K.expand_dims(K.cast(K.greater_equal(mask, self.threshold), tf.float32))
        mask = K.conv1d(mask, kernel, padding = 'same')   
        mask = K.squeeze(mask, axis = -1)
        masked_signal = mask * signal
        return masked_signal
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = {'threshold': self.threshold,
                 'tuning_range': self.tuning_range}
        base_config = super(MaskDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class MaskConv1D(Layer):
    def __init__(self, threshold, tuning_range, **kwargs):
        super(MaskConv1D, self).__init__(**kwargs)
        self.threshold = threshold
        self.tuning_range = tuning_range
    
    #could be interesting if loss function of classifier and generator do not work in unison: initiallization of bias maybe?
    # def build(self, input_shape):
    #     self.w = self.add_weight(
    #         name = 'w',
    #         shape=(input_shape[-1], self.units),
    #         initializer='random_normal',
    #         trainable=False,
    #         )
    #     self.b = self.add_weight(
    #         name='b', shape=(self.units, ), initializer='zeros', trainable=True)

    def call(self, inputs):
        signal, mask = inputs
        kernel = K.constant(1., dtype = 'float32', shape=(self.tuning_range, 1, 1))
        mask = K.expand_dims(K.cast(K.greater_equal(mask, self.threshold), tf.float32))
        mask = K.conv1d(mask, kernel, padding = 'same')
        masked_signal = mask * signal
        return masked_signal
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = {'threshold': self.threshold,
                 'tuning_range': self.tuning_range}
        base_config = super(MaskDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Clip(Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value
    
    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)
    
    def get_config(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}

class QuantizedDense(Dense):
    """Quantized Dense layer
    References: 
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1"
    [http://arxiv.org/abs/1602.02830]
    """
    def __init__(self,
                 units,
                 H = [0., 1.],
                 kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier = None,
                 **kwargs):
        super(QuantizedDense, self).__init__(units, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        
        super(QuantizedDense, self).__init__(units, **kwargs)
    
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
            
        self.kernel_constraint = Clip(self.H[0], self.H[1])
        self.kernel_initializer = RandomUniform(self.H[0], self.H[1])
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        
        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier,
                                   self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None
        
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
    
    # def call(self, inputs):
    #     quantized_kernel = quantize(self.kernel, H = self.H)
    #     weights_output = K.dot(inputs, quantized_kernel)
    #     true_output = K.dot(inputs, self.kernel)
    #     if self.use_bias:
    #         weights_output = K.bias_add(weights_output, self.bias)
    #         true_output = K.bias_add(true_output, self.bias)
    #     if self.activation is not None:
    #         weights_output = self.activation(weights_output)
    #         true_output = self.activation(true_output)
    #     output = [true_output, weights_output]
    #     return output
    
    def call(self, inputs):
        quantized_kernel = quantize(self.kernel, H = self.H)
        weights_output = K.dot(inputs, quantized_kernel)
        if self.use_bias:
            weights_output = K.bias_add(weights_output, self.bias)
        if self.activation is not None:
            weights_output = self.activation(weights_output)
        output = weights_output
        return output
    
    def heaviside(self, x):
        if x >= 0:
            y = 1
        else:
            y = 0
        return y
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
    
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class QuantizedConv2D(Conv2D):
    """Quantized Convolution2D layer
    
    CHANGES: - New Name
             - Calls quantize() instead of binarize()
             - To accomodate that, it allows for quantization level
               count Q to be passed at init
    
    References: 
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1"
    [http://arxiv.org/abs/1602.02830]
    """
    def __init__(self,
                 filters,
                 kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier=None,
                 weight_range=[0.,1.],
                 weight_levels=2,
                 **kwargs):
        super(QuantizedConv2D, self).__init__(filters, **kwargs)
        try:
            self.hi = weight_range[1]
            self.lo = weight_range[0]
        except TypeError:
            self.hi = weight_range
            self.lo = 0
        except IndexError:
            self.hi = weight_range[0]
            self.lo = 0
            
        self.Q = weight_levels
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
    
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        
        base = self.kernel_size[0] * self.kernel_size[1]
        if self.hi == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.hi = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
            #print('Glorot H: {}'.format(self.hi))
        
        if self.kernel_lr_multiplier == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.kernel_lr_multiplier = np.float32(
              1. / np.sqrt(1.5 / (nb_input+nb_output)))
            #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))
        
        self.kernel_constraint = Clip(self.lo, self.hi)
        self.kernel_initializer = RandomUniform(self.lo, self.hi)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier,
                                   self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.filters,), 
                                        initializer=self.bias_initializer, 
                                        name='bias', 
                                        regularizer=self.bias_regularizer, 
                                        constraint=self.bias_constraint)
        
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True
    
    def call(self, inputs):
        quantized_kernel = quantize(self.kernel,
                                    H=[self.lo, self.hi],
                                    Q=self.Q)
        outputs = K.conv2d(inputs,
                           quantized_kernel,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            outputs = K.bias_add(outputs,
                                 self.bias,
                                 data_format=self.data_format)
        
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    
    def get_config(self):
        config = {'weight_range': [self.hi,self.lo],
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# -- additional supporting functions --
# Original source: Ke Ding
# [https://github.com/DingKe/nn_playground/tree/master/binarynet]

def round_through(x):
    """Element-wise rounding to the closest integer with full gradient
    propagation.  A trick from [Sergey Ioffe]
    (http://stackoverflow.com/a/36480182)
    """
    # Round to nearest int.
    rounded = K.round(x)
    # Return "rounded" in forward prop because stop_gradient lets its
    # input pass but returns real x in backprop because stop_gradient
    # returns zero for gradient calculations.
    return x + K.stop_gradient(rounded - x)

def quantize_through(x, Q):
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


def quantize(W, H = [0., 1.], Q = 2):
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
        Wb = span * linear_saturation(W / span)
    else:
        Wb = span * quantized_linear(W / span, Q)
    
    return Wb + H[0]

def _hard_sigmoid(x):
    """Hard sigmoid different from the more conventional form (see
    definition of K.hard_sigmoid).
    
    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)

def linear_saturation(x):
    """Based on _hard_sigmoid(). Basically a linear function saturating
    at 0 and 1.

    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    return K.clip(x, 0, 1)

def binary_sigmoid(x):
    """Binary hard sigmoid for training binarized neural network.

    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    return round_through(_hard_sigmoid(x))


def quantized_linear(x, Q):
    """Binary hard sigmoid for training binarized neural network.

    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    return quantize_through(linear_saturation(x), Q)

def binary_tanh(x):
    """Binary hard sigmoid for training binarized neural network.
    The neurons' activations binarization function
    It behaves like the sign function during forward propagation
    And like:
      hard_tanh(x) = 2 * _hard_sigmoid(x) - 1 
      clear gradient when |x| > 1 during back propagation

    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    return 2 * round_through(_hard_sigmoid(x)) - 1
