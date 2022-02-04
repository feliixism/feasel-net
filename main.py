import numpy as np
from spec_net.feasel import DNN

data_path = "data/wine/"
X = np.load(data_path + "data.npy")
y = np.load(data_path + "labels.npy")
features = np.load(data_path + "features.npy").squeeze()

FS = DNN(X, y,
         layer_name='Linear',
         n_features=3,
         callback={'eval_metric': 'accuracy',
                   'd_min': 30,
                   'd_max': 300,
                   'n_samples': None,
                   'thresh': 0.70,
                   'decay': 0.0005,
                   'pruning_type':'linear',
                   },
         features=features,
         architecture_type='exp-down',
         normalization='min-max',
         activation='relu',
         loss = 'categorical_crossentropy')

FS.set_n_layers(3)
FS.set_learning_rate(0.002)
FS.set_batch_size(32)
FS.set_epochs(200)

FS.train_model()

FS.callback

FS.callback.trigger._converged

FS.time
FS._building_params['architecture_type']
# y_pred, y_true = feasel.test_model(X, y)

FS.name
# # feasel.plot.input_reduction('both', highlight = True)
FS.plot.feature_entropy()
FS.plot.feature_history()
FS.plot.pruning_history()
FS.plot.history()
# FS.plot.predict(X[1], y[1])
# # feasel.plot.predict_set(X, y, 'all')
# # feasel.plot.predict_set(X, y, 'accuracy')
# # feasel.plot.predict_set(X, y, 'recall')
# # feasel.plot.predict_set(X, y, 'precision')
# feasel.model.summary()
FS.plot.confusion_matrix(X, y)
