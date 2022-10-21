"""
feasel.nn.tfcustom.layers
=========================
"""

from tensorflow.keras import activations, initializers, regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.constraints import NonNeg
import tensorflow.keras.backend as K

class LinearPass(Layer):
  """
  This is a custom *Tensorflow* layer, where all the inputs are handed over
  without any non-linear function, weight or bias application. Eventhough it
  does not do much, it is very powerful for the recursive pruning of input
  nodes, since its weights are able to be set manually and only manually.

  """
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
    super(LinearPass, self).__init__(trainable=trainable, name=name, **kwargs)
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

  def call(self, inputs):
    """
    The function that calls the layer.

    Parameters
    ----------
    inputs : *tf.Tensor*
      The input tensor.

    Returns
    -------
    output : *tf.Tensor*
      The output tensor.

    """
    output = inputs * self.kernel
    if self.use_bias:
      output = K.bias_add(output,
                          self.bias,
                          data_format='channels_last')
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
      'activity_regularizer': regularizers.serialize(self.activity_regularizer),
      'kernel_constraint': constraints.serialize(self.kernel_constraint),
      'bias_constraint': constraints.serialize(self.bias_constraint)
      }
    base_config = super(LinearPass, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))