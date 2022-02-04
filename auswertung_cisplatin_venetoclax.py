import pandas as pd
import numpy as np
from spec_net.feasel import DNN
import matplotlib.pyplot as plt

df = pd.read_excel('U:/privat/Promotion/D_Dataset/Raman/RT112_CisplatinVenetoclax_Mitochondria_24h_48h.xlsx')
df1 = df[df['Drug'] == 'Cisplatin']
df2 = df[df['Drug'] == 'Venetoclax']

df1 = df1[df1['Duration'] == 24]
# df1 = df1[df1['Concentration'] == 1]
df1 = df1.append(df1)

df0 = df[df['Drug'] == 'Control']

x = df1.append(df0)

X = np.array(x)[:,6:].astype(float)
y = np.array(x['Concentration']).astype('str')
features = np.array(x.columns[6:]).astype(float)

feasel = DNN(X, y,
             layer_name='Linear',
             n_features=10,
             callback={'metric': 'accuracy',
                       'd_min': 15,
                       'd_max': 300,
                       'n_samples': 10,
                       'thresh': 1.,
                       # 'decay': 0.00005,
                       'pruning_type':'exp',
                       },
             features=features,
             architecture_type='down',
             normalization='standardize',
             activation='relu',
             loss = 'categorical_crossentropy')

feasel.set_n_layers(4)
feasel.set_learning_rate(0.00005)
feasel.set_batch_size(2**7)
feasel.set_epochs(300)

feasel.train_model()

feasel.plot.input_reduction('both', highlight = True)
feasel.plot.history()
# feasel.plot.pruning_history()
# feasel.plot.model()
# feasel.model.summary()
# features[feasel.get_mask()]
plt.plot(features, feasel.data.X_train[0])

np.amax(feasel.data.X_train[0])
