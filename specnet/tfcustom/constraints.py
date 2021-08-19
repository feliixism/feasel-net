from keras.constraints import Constraint
import keras.backend as K

class MinMaxClip(Constraint):
    """Constrains the weights to be in between two values.
    """
    
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value
    
    def __call__(self, w):
        clipped = w * K.cast(K.greater_equal(w, self.min_value), K.floatx())
        clipped = w * K.cast(K.less_equal(clipped, self.max_value), K.floatx()) + K.cast(K.greater_equal(clipped, self.max_value), K.floatx()) * self.max_value
        return clipped
    
    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

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