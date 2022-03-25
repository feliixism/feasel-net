import numpy as np
from spec_net.feasel import DNN

data_path = "data/wine/npy/"
X = np.load(data_path + "data.npy")
y = np.load(data_path + "labels.npy")
features = np.load(data_path + "features.npy")

# the FeaSel model container for Dense type neural networks:
FS = DNN(X, y,
         layer_name='Linear',
         n_features=3,
         # 'data', 'train', 'build' or 'callback' parameters can also be set
         # within dictionaries:
         callback={'eval_type': 'accuracy',
                   'd_min': 10,
                   'd_max': 300,
                   'n_samples': 0,
                   'thresh': 0.98,
                   'decay': 0.0005,
                   'pruning_type':'exp.',
                   'scale': False,
                   },
         features=features,
         architecture_type='exp-down',
         normalization='min-max',
         activation='relu',
         loss='categorical_crossentropy')

# sets some parameters outside of class instantiation:
FS.set_n_layers(3)
FS.set_learning_rate(0.002)
FS.set_batch_size(32)
FS.set_epochs(100)

# starts the training process:
FS.train_model()

def plot():
  # Performance Evaluations:
  FS.plot.ROC(X, y)
  FS.plot.predict(X[1], y[1])
  FS.plot.predict_set(X, y, ['accuracy', 'sensitivity', 'specificity'])
  FS.plot.predict_set(X, y, 'precision')
  FS.plot.predict_set(X, y, 'f1-score')
  FS.plot.confusion_matrix(X, y)

  # Layer Insights:
  FS.plot.weights('both')
  FS.plot.feature_maps(X[1], 'Dense1')

  # Training History:
  FS.plot.history()

  # Feature Selection:
  FS.plot.feature_entropy()
  FS.plot.pruning_history()
FS.plot.input_reduction('both', highlight=True)