import numpy as np
from feasel.nn import FSDNN
from feasel.data.normalize import min_max

data_path = "data/wine/npy/"
X = np.load(data_path + "data.npy")
y = np.load(data_path + "labels.npy")
features = np.load(data_path + "features.npy")

X = min_max(X, axis=0)

FS = FSDNN(X, y,
            layer_name='Linear',
            n_features=3,

            # 'data', 'train', 'build' or 'callback' parameters can also be set
            # within dictionaries:
            callback={'eval_type': 'accuracy',
                      'd_min': 20,
                      'd_max': 100,
                      'n_samples': None,
                      'thresh': 0.98,
                      'pruning_type': 'exp.',
                      'scale': True,
                      'normalization': 'min-max',
                      },

            features=features,
            architecture_type='exp-down',
            normalization='standardize',
            activation='relu',
            loss='categorical_crossentropy')

# sets some parameters outside of class instantiation:
FS.set_n_layers(3)
FS.set_learning_rate(0.001)
FS.set_batch_size(16)
FS.set_epochs(1000)

# FS.compile_model()

# starts the training process:
FS.train()

# FS.plot.predict_set(X, y, ['accuracy', 'sensitivity', 'specificity'])
# FS.plot.input_reduction('both', highlight=True)
FS.plot.information_richness(pruning_step=-1)
FS.plot.mask_history(highlight=True)

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
  FS.plot.information_richness(pruning_step=0)

  # Feature Selection:

  FS.plot.pruning_history()
  FS.plot.mask_history(highlight=True)
  FS.plot.input_reduction('both', highlight=True)

plot()

