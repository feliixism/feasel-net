from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf
from keras.layers import Activation

@keras_export('keras.activations.scaled_softsign')
def scaled_softsign(x, alpha=1.0):
    return 1/2 * (1 + alpha * K.softsign(x))

#activation functions as methods:
@keras_export('keras.activations.heaviside')
@tf.custom_gradient
def heaviside(x, threshold = 0.5):
    """
    Heaviside or Binary Step activation function.

    Parameters
    ----------
    x : tensor
        Input tensor.
    threshold : int
        Input value at which the unit step is being executed.
    tuning_range : int
        Sets the number of neighbored values that are set to one as well.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    tf_threshold = tf.constant(threshold)
    binarized_mask=K.expand_dims(K.cast(K.greater_equal(x, tf_threshold), dtype=K.floatx()))
    # @tf.RegisterGradient("heaviside")
    def grad(dy):
        grad = K.sigmoid(dy) * (1 - K.sigmoid(dy))
        return grad
    return binarized_mask#, grad

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"heaviside": Activation(heaviside), "scaled_softsign": Activation(scaled_softsign)})