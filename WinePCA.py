import numpy as np
from spec_net.linear_transformation import PCA
data_path = "data/wine/npy/"
X = np.load(data_path + "data.npy")
y = np.load(data_path + "labels.npy")
features = np.load(data_path + "features.npy")

PC = PCA(X, y, features=features,
         n_components=2,
         normalization='min-max',
         solver='svd')

test = PC.data.X_train

def plot():
  PC.plot.loadings([1,2], type='bar')
  PC.plot.loadings([1,2], type='line')
  PC.plot.loadings([1,2], FOI='alcohol', type='scatter')
  PC.plot.scree("both", show_average = True)
  PC.plot.scores(projection=True)
  PC.plot.biplot('alcohol')
  PC.plot.contribution_bar(1, True)
  PC.plot.contribution_circle()
  PC.plot.contribution_heatmap(show_sum=True)
  PC.plot.contribution_bars(components=[1,2], show_average=True)
