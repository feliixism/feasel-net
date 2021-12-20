import numpy as np
from spec_net.feasel import DNN
from spec_net.utils.log import _tf_warnings

_tf_warnings('3')

data_path = "data/wine/"
X = np.load(data_path + "data.npy")
y = np.load(data_path + "labels.npy")
features = np.load(data_path + "features.npy").squeeze()

feasel = DNN(X, y,
    layer_name='Linear',
    n_features=3, 
    callback={'metric': "accuracy", 
              'd_min': 30,
              'd_max': 300,
              'n_samples': None,
              'thresh': 0.98,
              'decay': 0.001,
              'pruning_type':'linear',
              },
    features=features, 
    architecture_type='const',
    normalization='min-max', 
    activation='relu',
    loss = 'categorical_crossentropy')

feasel.set_n_layers(2)
feasel.set_learning_rate(0.0003)
feasel.set_batch_size(16)
feasel.set_epochs(2000)

feasel.train_model()

feasel.plot.input_reduction('both', highlight = True)
feasel.plot.history()
feasel.plot.pruning_history()
feasel.plot.model()
feasel.model.summary()

#too random --> the solution is maybe in callback decay?
feasel.data.y_train
