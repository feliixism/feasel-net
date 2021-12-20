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
attributes[7] = "nonflav_phenols"
attributes[11] = "OD280/OD315"

X = df.standardize(data)
y = labels.reshape([len(labels), 1])

ld = lda.LDA(data, y = labels, features = attributes, n_components = 2, rowvar = False, standardize = True)

ld.plot.scree("both", show_average = True)
ld.plot.loadings_line()
ld.plot.loadings(FOI = ["proline"])
ld.plot.scores(decision_boundaries=True)
ld.plot.biplot()
ld.plot.contribution_bar(True)
ld.plot.contribution_circle()
ld.plot.contribution_heatmap()
