import numpy as np
from spec_net.linear_transformation import LDA
data_path = "data/wine/npy/"
X = np.load(data_path + "data.npy")
y = np.load(data_path + "labels.npy")
features = np.load(data_path + "features.npy")

LD = LDA(X, y, features=features,
         n_components=2,
         normalization='standardize',
         solver ='svd')

def plot():
  LD.plot.loadings([1,2], type='bar')
  LD.plot.loadings([1,2], type='line')
  LD.plot.loadings([1,2], type='scatter')
  LD.plot.scree('both', show_average=True)
  LD.plot.scores(1, 2, projection=True, decision_boundaries=True)
  LD.plot.biplot('alcohol')
  LD.plot.contribution_bar(2, True)
  LD.plot.contribution_circle()
  LD.plot.contribution_heatmap(show_sum=True)
  LD.plot.contribution_bars(components=[1,2], show_average=True)


# plot()
pred = LD.predict(X)
decode = LD.decode(X) # decode doesn't really decode right
