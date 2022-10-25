import numpy as np
from feasel.nn import FSDNN
from feasel.linear import PCA
from feasel.data.normalize import min_max

#load wine classification data:
data_path = "data/wine/npy/"
X = np.load(data_path + "data.npy")
y = np.load(data_path + "labels.npy")
features = np.load(data_path + "features.npy")

#min-max scale:
X = min_max(X, axis=0)

PC = PCA(X, y, features, n_components=3)
PC.plot.scores()

#instantiation of feasel-net:
FS = FSDNN(X, y,
           layer_name='Linear',
           n_features=3,

           #'data', 'train', 'build' or 'callback' parameters can also be set
           #within dictionaries:
           callback={'eval_type': 'accuracy',
                     'd_min': 20,
                     'd_max': 100,
                     'n_samples': None,
                     'thresh': 0.98,
                     # 'decay': 0.0005,
                     'pruning_type': 'exp.',
                     'scale': True,
                     'remove_outliers': True,
                     'accelerate': True
                   },
           test_split=0.2,
           architecture_type='exp-down',
           normalization=None,
           activation='relu',
           loss='categorical_crossentropy')

# sets some parameters outside of class instantiation:
FS.set_n_layers(3)
FS.set_learning_rate(0.005)
FS.set_batch_size(16)
FS.set_epochs(1000)

# starts the training process:
FS.train()

def plot():
  # Performance Evaluations:
  FS.plot.ROC(X, y)
  FS.plot.predict(X[1], y[1])
  FS.plot.predict_set(X, y, ['accuracy', 'sensitivity', 'specificity'])
  FS.plot.predict_set(X, y, 'precision')
  FS.plot.predict_set(X, y, 'f1-score')
  FS.plot.confusion_matrix(X, y)

  # Layer Insights:
  # FS.plot.weights('both')
  # FS.plot.feature_maps(X[1], 'Dense1')

  # Training History:
  FS.plot.history()
  FS.plot.FOI(pruning_step=0)

  # Feature Selection:

  FS.plot.pruning_history()
  FS.plot.mask_history(highlight=True)
  FS.plot.input_reduction('both', highlight=True)

plot()
