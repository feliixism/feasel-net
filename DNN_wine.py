import numpy as np
from spec_net.architectures import DenseDNN

data_path = "data/wine/"
X = np.load(data_path + "data.npy")
y = np.load(data_path + "labels.npy")
features = np.load(data_path + "features.npy").squeeze()
features = np.arange(600, 600+len(features))

dnn = DenseDNN(X, y,
    features=features, 
    architecture_type='exp-down',
    normalization='min_max', 
    activation='relu')

dnn.set_n_layers(2)
dnn.set_learning_rate(0.0003)
dnn.set_batch_size(16)
dnn.set_epochs(10)

dnn.train_model()

# feasel.plot.input_reduction('both', highlight = True)
dnn.plot.history()
# feasel.plot.pruning_history()
dnn.plot.model()
dnn.model.summary()
