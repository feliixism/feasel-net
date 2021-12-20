class Params:
    def __init__(self):
        self.train = TrainingParams()
        self.build = BuildingParams()
        self.data = DataParams()

class BaseParams:
    def __init__(self):
        self._TRAINING_MAP = {}
    
    def update(self, key):
        return self._TRAINING_MAP[key]        

class BuildingParams(BaseParams):
    def __init__(self,
                 architecture_type='down',
                 n_layers=3,
                 n_features=None,
                 n_nodes=None):
        super().__init__()
        self.architecture_type = architecture_type
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_nodes = n_nodes
        
        self._TRAINING_MAP = {'architecture_type': self.set_architecture_type,
                              'n_layers': self.set_n_layers,
                              'n_features': self.set_n_features}
    
    def __repr__(self):
        return ('Parmeter container for the generic building process\n'
                f'{self.__dict__}')
    
    def set_architecture_type(self, architecture_type):
        self.architecture_type = architecture_type
    
    def set_n_layers(self, n_layers):
        self.n_layers = n_layers
    
    def set_n_features(self, n_features):
        self.n_features = n_features
        
    def set_n_nodes(self, n_nodes):
        self.n_nodes = n_nodes
        
class DataParams(BaseParams):
    def __init__(self, 
                 normalization=None, 
                 sample_axis=None, 
                 input_layer=None):
        super().__init__()
        self.normalization = normalization
        self.sample_axis = sample_axis
        self.input_layer = input_layer
        
        self._TRAINING_MAP = {'normalization': self.set_normalization,
                              'sample_axis': self.set_sample_axis,
                              'input_layer': self.set_input_layer}
        
    def __repr__(self):
        return ('Parmeter container for generic data processing\n'
                f'{self.__dict__}')
    
    def set_normalization(self, normalization):
        self.normalization = normalization
    
    def set_sample_axis(self, sample_axis):
        self.sample_axis = sample_axis
    
    def set_input_layer(self, input_layer):
        self.input_layer = input_layer

from tensorflow import keras

class TrainingParams(BaseParams):
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
        
        self._TRAINING_MAP = {'batch_size': self.set_batch_size,
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
        return ('Parmeter container for the generic training process\n'
                f'{self.__dict__}')
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_epochs(self, epochs):
        self.epochs = epochs
        
    def set_eta(self, eta):
        self.eta = eta
    
    def set_optimizer(self, optimizer):
        try:
            self.optimizer = (self._OPTIMIZER_MAP(f'{optimizer}')
                              (leraning_rate=self.eta))
        except:
            NameError(f"Optimizer '{optimizer}' is not implemented. See "
                      "'show_optimizers()' for possible optimizers.")
            
    def show_optimizers(self, optimizer):
        print(self._OPTIMIZER_MAP.__dict__.keys())
    
    def set_loss(self, loss):
        self.loss = loss
    
    def set_activation(self, activation):
        self.activation = activation
    
    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
    
    def set_test_split(self, test_split):
        self.test_split = test_split
