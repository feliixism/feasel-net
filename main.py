import numpy as np
import matplotlib.pyplot as plt
from spec_net.analysis import pca, lda
import sklearn.datasets as datasets
from spec_net.preprocess import dataformat as df
from spec_net.tfcustom import plot
from sklearn.model_selection import train_test_split
from spec_net.data.classification import Classification
from spec_net.architectures import ann2, base

from keras import Model

a = Model()


test = np.array([5,2,3,4,9])
np.gradient(test, 5)

wine = datasets.load_wine()

location = "U:/privat/Promotion_GewebedifferenzierungmittelsIR-Spektroskopie/D_Dataset/Wine/"

data = wine.data
labels = wine.target
attributes = wine.feature_names

X = data
y = labels.reshape([len(labels), 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# wine.feature_names
# # x = data

# nn = ann2.ANN(X, y, features = attributes, architecture_type = "down",
#               normalization = "standardize", activation_function = "relu")

# # nn.compile_model()
# # nn.fit_model()
# # nn.model.summary()
# # nn.train_model()
# # print(nn.callback[0].trigger_params.__dict__)

# # nn.__dict__.keys()


# # nn.get_params("training")
# # plt.imshow(nn.data.X_train)
# nn.set_n_layers(2)
# nn.set_learning_rate(0.0005)
# nn.set_batch_size(16)
# nn.set_epochs(1000)
# nn.compile_model()
# nn.model.get_layer("Dense0").kernel_regularizer = "l1"
# nn.train_model()
# plt.plot(np.sum(np.abs(nn.model.get_layer("Dense0").kernel.numpy()), axis = -1))
# nn.model.get_layer("Dense0").__dict__

# # nn.callback.__dict__
# nn = ann2.ANN(X, y, architecture_type="const")
# nn = ann2.FSANN(X, y, architecture_type = "down")
# nn = ann2.FSANN(X, y, features = attributes, 
#                 n_features = 3, architecture_type = "down", 
#                 normalization = "standardize", 
#                 threshold = 0.90, 
#                 metric = "accuracy", activation_function = "relu", delta = 50)

nn = ann2.FSANN(X, y, layer = 'Linear', n_features = 3, 
                callback = {"metric": "accuracy", 
                            "d_min": 20,
                            "d_max": 300,
                            "n_samples": 5,
                            'decay': 0.001},
                features = attributes, architecture_type = "down",
                normalization = "standardize", activation_function = "relu")

nn.compile_model()
# nn.fit_model()
# nn.model.summary()

# print(nn.callback[0].trigger_params.__dict__)

# nn.__dict__.keys()


# nn.get_params("training")
# plt.imshow(nn.data.X_train)
nn.set_n_layers(2)
nn.set_learning_rate(0.0005)
nn.set_batch_size(16)
nn.set_epochs(1000)
nn.model.summary()

# nn.set_callback(delta_max = 250, delta = 5)
nn.train_model()
nn.plot.history()
# nn.model.get_layer("Linear").trainable
plt.plot(np.abs(nn.model.get_layer("Linear").kernel.numpy()))

nn.callback[0].log.weights
# nn.callback[0].log
# a = nn.x_train
# a - nn.x_train
# test = np.array([[1,1,1,1,1,1,1,1,1], 
#                   [1,1,1,1,0,1,1,1,1], 
#                   [1,1,1,1,0,0,1,1,1]])

# x = np.arange(len(test[0]))

# cmap = plt.get_cmap("rainbow")

# fig = plt.figure("Test")
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# for i in range(len(test)):
#     ax1.bar(x, test[i], align = "center", width = 1, color = cmap(i / len(test)))
# ax1.plot()
# ax1.set_ylim(0, 1)
# ax1.set_xlim(0.5, x[-1] - 0.5)

# set([1,2,3,4]) | set([1,4,5])
