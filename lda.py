import numpy as np
from spec_net.linear_transformation import LDA
import sklearn.datasets as datasets

wine = datasets.load_wine()

location = "U:/privat/Promotion_GewebedifferenzierungmittelsIR-Spektroskopie/D_Dataset/Wine/"

X = wine.data
y = wine.target
features = np.array(wine.feature_names)

LD = LDA(X, y, features=features,
         n_components=10,
         normalization='standardize',
         solver ='eigs')

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

pred = LD.predict(X)
decode = LD.decode(X) # decode doesn't really decode right
LD.data._feature_scale
