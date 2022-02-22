import numpy as np
from spec_net.linear_transformation import PCA
import sklearn.datasets as datasets

wine = datasets.load_wine()

location = "U:/privat/Promotion_GewebedifferenzierungmittelsIR-Spektroskopie/D_Dataset/Wine/"

X = wine.data
y = wine.target
features = np.array(wine.feature_names)
# X = df.min_max_scale(data, axis = -1)
# y = labels.reshape([len(labels), 1])
# features = None
PC = PCA(X, y, features=features,
         n_components=2,
         # sample_axis=0,
         normalization='min-max',
         solver='svd')

test = PC.data.X_train

# img = PC.decode()
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

plot()
