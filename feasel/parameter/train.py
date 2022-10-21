"""
feasel.parameter.train
======================
"""

from tensorflow import keras

from .base import BaseParams

class TrainParamsNN(BaseParams):
  def __init__(self,
               batch_size=32,
               epochs=100,
               eta=1e-4,
               optimizer='adam',
               loss='categorical_crossentropy',
               activation='relu',
               dropout_rate=None,
               test_split=0.2,
               metric='accuracy'):

    super().__init__()

    self.batch_size = batch_size
    self.epochs = epochs
    self.eta = eta
    self.optimizer = optimizer
    self.loss = loss
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.test_split = test_split
    self.metric = metric
    self._trained = False

    self._MAP = {'batch_size': self.set_batch_size,
                 'epochs': self.set_epochs,
                 'eta': self.set_eta,
                 'optimizer': self.set_optimizer,
                 'loss': self.set_loss,
                 'activation': self.set_activation,
                 'dropout_rate': self.set_dropout_rate,
                 'test_split': self.set_test_split}

    self._OPTIMIZER_MAP = {'adam': keras.optimizers.Adam,
                           'sgd': keras.optimizers.SGD,
                           'rmsprop': keras.optimizers.RMSprop,
                           'adadelta': keras.optimizers.Adadelta,
                           'adagrad': keras.optimizers.Adagrad,
                           'adamax': keras.optimizers.Adamax,
                           'nadam': keras.optimizers.Nadam
                           }

  def __repr__(self):
    return ('Parmeter container for the generic train process\n'
            f'{self.__dict__}')

  def set_batch_size(self, batch_size):
    self.batch_size = batch_size

  def set_epochs(self, epochs):
    self.epochs = epochs

  def set_eta(self, eta):
    self.eta = eta

  def set_optimizer(self, optimizer):
    self.optimizer = optimizer

  def set_loss(self, loss):
    self.loss = loss

  def set_activation(self, activation):
    self.activation = activation

  def set_dropout_rate(self, dropout_rate):
    self.dropout_rate = dropout_rate

  def set_test_split(self, test_split):
    self.test_split = test_split

  def get_optimizer_function(self):
    try:
      func = self._OPTIMIZER_MAP[f'{self.optimizer}'](learning_rate=self.eta)

    except:
      NameError(f"Optimizer '{self.optimizer}' is not implemented. See "
                "'show_optimizers()' for possible optimizers.")

    return func

  def show_optimizers(self, optimizer):
    print(self._OPTIMIZER_MAP.__dict__.keys())