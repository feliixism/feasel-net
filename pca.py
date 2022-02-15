import numpy as np
import matplotlib.pyplot as plt
from spec_net.linear_transformation import ModelContainer as PCA
import sklearn.datasets as datasets

wine = datasets.load_wine()

location = "U:/privat/Promotion_GewebedifferenzierungmittelsIR-Spektroskopie/D_Dataset/Wine/"

X = wine.data
y = wine.target
features = np.array(wine.feature_names)
# X = df.min_max_scale(data, axis = -1)
# y = labels.reshape([len(labels), 1])

PC = PCA(X, y, features=features,
         n_components=4,
         sample_axis=0,
         normalization='standardize',
         solver ='eigs')

# PC.plot.scree("both", show_average = True)
# PC.plot.loadings_line()
# PC.plot.loadings(FOI=["flavanoids"])
# PC.plot.scores(projection=True)
# PC.plot.biplot(FOI='alcohol')
# PC.plot.contribution_bar(2, True)
# PC.plot.contribution_circle()
# PC.plot.contribution_heatmap(show_sum=True)
# PC.plot.contribution_bars(components=[1,2], show_average=True)
