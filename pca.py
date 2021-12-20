import numpy as np
import matplotlib.pyplot as plt
from spec_net.analysis import pca, lda
import sklearn.datasets as datasets
from spec_net.preprocess import dataformat as df

wine = datasets.load_wine()

location = "U:/privat/Promotion_GewebedifferenzierungmittelsIR-Spektroskopie/D_Dataset/Wine/"

data = wine.data
labels = wine.target
attributes = wine.feature_names
attributes[11] = "od280/od315"
attributes[7] = "nonflav"
X = df.min_max_scale(data, axis = -1)
y = labels.reshape([len(labels), 1])

pc = pca.PCA(X, y = labels, features = attributes, n_components = 3, rowvar = False, standardize = True, solver ="svd")

pc.plot.scree("both", show_average = True)
pc.plot.loadings_line()
pc.plot.loadings(FOI = ["flavanoids"])
# pc.plot.scores()
# pc.plot.biplot()
# pc.plot.contribution_bar(1, True)
# pc.plot.contribution_circle()
pc.plot.contribution_heatmap(show_sum=True)
# pc.plot.contribution_bars(show_average=True)
