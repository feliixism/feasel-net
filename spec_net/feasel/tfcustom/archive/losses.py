from keras.losses import LossFunctionWrapper
from keras.utils import losses_utils
import keras.backend as K
from tensorflow.python.util.tf_export import keras_export
import six
import numpy as np

@keras_export('keras.losses.ReduceWavelengths')
class ReduceWavelengths(LossFunctionWrapper):
    def __init__(self,
                 n_wavelengths,
                 threshold=0.5,
                 name='reduce_wavelengths'):
        super(ReduceWavelengths, self).__init__(
            reduce_wavelengths, n_wavelengths=n_wavelengths, threshold=threshold, name=name)
        
    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if K.is_tensor(v) or K.is_variable(v) else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.losses.reduce_wavelengths')
def reduce_wavelengths(y_true, y_pred, n_wavelengths, threshold=0.5):
    n_wavelengths = K.constant(n_wavelengths)
    threshold = K.constant(threshold)
    measured_points = K.constant(y_pred.shape[-1])
    
    mask = K.cast(K.greater_equal(y_pred, threshold), dtype=float)
    mask_values = mask * y_pred
    
    loss = K.abs(K.sum(mask_values)-n_wavelengths)
    return loss

# def test_loss(self, y_true, y_pred):
#      #constants (watch out for batch size: loss calculated over whole batch)
#      n_wavelengths = tf.constant(self.n_wavelengths, dtype = float)
#      max_sd = tf.constant(self.y_train.shape[1], dtype = float)

#      def sum_above_threshold(x):
#          mask = tf.greater_equal(x, self.threshold)
         
#          zeros = tf.zeros(mask.shape)
#          ones = tf.zeros(mask.shape)
         
#          sat = tf.reduce_sum(tf.where(mask, x, zeros))
         
#          # comp = n_wavelengths - sat
#          # 
#          # original = tf.abs(1.0 - n_wavelengths / sat)
         
#          sat = tf.abs(sat - n_wavelengths)
#          sat = sat / tf.abs(max_sd - n_wavelengths)
#          return sat
     
#      sat = sum_above_threshold(y_pred[0])
     
#      return sat

# @keras_export('keras.losses.MorseLoss')
# class MorseLoss(LossFunctionWrapper):
#     def __init__(self,
#                  n_wavelengths,
#                  threshold=0.5,
#                  reduction=losses_utils.Reduction.SUM,
#                  name='reduce_wavelengths'):
#         super(MorseLoss, self).__init__(
#             morse_loss, n_wavelengths=n_wavelengths, threshold=threshold, reduction=reduction, name=name)        
    
#     def get_config(self):
#         config = {}
#         for k, v in six.iteritems(self._fn_kwargs):
#             config[k] = K.eval(v) if K.is_tensor(v) or K.is_variable(v) else v
#         base_config = super(LossFunctionWrapper, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

# @keras_export('keras.losses.morse_loss')
# def morse_loss(y_true, y_pred, n_wavelengths, threshold=0.5):
#     n_wavelengths = K.constant(n_wavelengths)
#     threshold = K.constant(threshold)
    
#     mask = K.cast(K.greater_equal(y_pred, threshold), dtype=float)
#     mask_values = mask*y_pred
    
#     sum_mask_values = K.sum(mask_values) / 64
    
#     loss = (1-K.exp(-(sum_mask_values-n_wavelengths)))**2
#     return loss
