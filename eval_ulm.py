import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

indices = [[1,3,6,7,8], [2,4,5,9,10,11]] # background, tumor
idx_labels = ['background', 'tumor']

directory = Path("U:/privat/Promotion_GewebedifferenzierungmittelsIR-Spektroskopie/M_Measurements/IABC_Ulm/20211008/raw/2cmSampling/")
files = list(directory.rglob('*.txt'))
spectra = []
labels = []

# data files and info
for file in files:
    labels.append(file.stem)
    data = np.array(pd.read_csv(file, sep='\t'))[:,1:] 
    data = np.char.replace(data.astype(str), ',', '.')
    data = data.astype(float)
    x = data[:, 0]
    spectra.append(data[:, -1])
spectra = np.array(spectra)

def get_absorbance(array, idx_bg=0, sigma=0):
    bg = array[idx_bg]
    absorbance, label_array = [], []
    for i in range(len(array)):
        if i == idx_bg:
            pass
        data = np.log(bg/array[i])
        if sigma:
            data = gaussian_filter1d(data, sigma = sigma)
        absorbance.append(data)
        label_array.append(labels[i])
    return np.array(absorbance), np.array(label_array)

absorbance, labels = get_absorbance(spectra, sigma = 2)
labels = np.empty(len(labels))
labels[indices[0]] = 0
labels[indices[1]] = 1


import sklearn.datasets as datasets
import numpy as np
from sklearn.model_selection import train_test_split
from spec_net.architectures.feasel import FeaSelDNN
from spec_net.architectures.dense_dnn import DenseDNN

from scipy.signal import savgol_filter

# path = "U:/privat/Promotion_GewebedifferenzierungmittelsIR-Spektroskopie/M_Measurements/Lucas/Cisplatin/npy/"

# labels = np.load(path + "Nuclei48h/base_y.npy")
# concentration = 1
# mask_cisplatin = np.argwhere(labels[:, 1] == f"{concentration}").squeeze()
# mask_control = np.argwhere(labels[:, 1] == "0").squeeze()
# mask = np.append(mask_cisplatin, mask_control)

# X = np.load(path + "Nuclei48h/base_X.npy")[mask].squeeze()[:,1:]
# X = savgol_filter(X, 7, 1)
# features = np.linspace(402, 3003, X.shape[1])
# y = labels[mask, 0]

feasel = FeaSelDNN(absorbance[1:], labels[1:], 'Linear', 
                    callback={
                              'metric': "accuracy", 
                              'd_min': 30,
                              'd_max': 500,
                              'n_samples': 5,
                              'thresh': 0.98,
                              'decay': 0.005,
                              },
                    epochs = 3000,
                    n_layers = 5,
                    n_features=5,
                    features=x,
                    architecture_type = "exp-down", batch_size=4)
feasel.train_model()
feasel.model.summary()
feasel.plot.history()
feasel.plot.mask_history()
feats = np.array(x[feasel.get_mask()], ndmin=2)
